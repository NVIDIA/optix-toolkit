//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "CudaCheck.h"

#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>

#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include "Memory/Buffers.h"
#include "Memory/PinnedItemPool.h"
#include "Memory/TilePool.h"
#include "Textures/SparseTexture.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

using namespace demandLoading;
using namespace imageSource;

// A tile request specifies a miplevel and the tile coordinates.
struct TileRequest
{
    SparseTexture* texture;
    unsigned int   mipLevel;
    unsigned int   tileX;
    unsigned int   tileY;
    CUstream       stream;
};


// TileFiller is constructed from a SparseTexture.  Given a TileRequest, it allocates memory from a
// TilePool and calls fillTile() on the texture.
class TileFiller
{
  public:
    TileFiller( const Options& options )
        : m_pool( options.maxTexMemPerDevice )
        , m_pinnedTiles( options.maxPinnedMemory / sizeof( TileBuffer ) )
    {
    }

    void fillTile( const TileRequest& request )
    {
        // Use the CUDA context associated with the stream in the request.
        CUcontext context;
        DEMAND_CUDA_CHECK( cuStreamGetCtx( request.stream, &context ) );
        DEMAND_CUDA_CHECK( cuCtxSetCurrent( context ) );

        // Allocate device memory from the TilePool.
        TileBlockDesc tileLocator = m_pool.allocate( sizeof( TileBuffer ) );
        ASSERT_TRUE( tileLocator.isValid() );

        // Allocate pinned memory for tile.  (We don't bother to initialize it.)
        TileBuffer* pinnedTile = m_pinnedTiles.allocate();

        CUmemGenericAllocationHandle handle;
        size_t                       offset;
        m_pool.getHandle( tileLocator, &handle, &offset );

        try
        {
            // Fill the tile.
            request.texture->fillTile( request.stream, request.mipLevel, request.tileX, request.tileY, pinnedTile->data,
                                       CU_MEMORYTYPE_HOST, sizeof( TileBuffer ), handle, offset );
        }
        catch( ... )
        {
            printf( "Error: mipLevel=%i, tileX=%i, tileY=%i, handle=%llx, offset=%zu (N=%g)\n", request.mipLevel,
                    request.tileX, request.tileY, handle, offset, double( offset ) / sizeof( TileBuffer ) );
            fflush( stdout );
            ++m_numErrors;
        }

        // Free the pinned memory.
        m_pinnedTiles.free( pinnedTile, request.stream );
    }

    int getNumErrors() const { return m_numErrors; }

  private:
    TilePool                   m_pool;
    PinnedItemPool<TileBuffer> m_pinnedTiles;
    std::atomic<int>           m_numErrors{0};
};


class TestTilePool : public testing::Test
{
  public:
    void SetUp()
    {
        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

        // Initialize TextureDescriptor.
        TextureDescriptor desc{};
        desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
        desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
        desc.filterMode       = CU_TR_FILTER_MODE_POINT;
        desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        desc.maxAnisotropy    = 16;

        // Initialize TextureInfo.
        TextureInfo info{};
        info.width        = 2048;
        info.height       = 2048;
        info.format       = CU_AD_FORMAT_HALF;
        info.numChannels  = 4;
        info.numMipLevels = 12;

        // Create a vector of textures.
        m_textures.reserve( 4 );
        for( size_t i = 0; i < 4; ++i )
        {
            m_textures.push_back( SparseTexture() );
            m_textures.back().init( desc, info );
        }

        // Create a vector of streams, which are used in round-robin fashion.
        m_streams.resize( 4 );
        for( CUstream& stream : m_streams )
        {
            DEMAND_CUDA_CHECK( cudaStreamCreate( &stream ) );
        }
    }

    void TearDown()
    {
        for( CUstream stream : m_streams )
        {
            DEMAND_CUDA_CHECK( cudaStreamDestroy( stream ) );
        }
        m_streams.clear();
    }

    std::vector<TileRequest> createRequests()
    {
        // All the textures have the same size.  We iterate based on the dimensions of the first textures.
        const SparseTexture& firstTexture = m_textures.at( 0 );

        // Generate vector of tile requests.
        size_t                   whichTexture = 0;
        size_t                   whichStream  = 0;
        std::vector<TileRequest> requests;
        unsigned int             mipTailFirstLevel = firstTexture.getMipTailFirstLevel();
        for( unsigned int mipLevel = 0; mipLevel < mipTailFirstLevel; ++mipLevel )
        {
            uint2        mipLevelDims = firstTexture.getMipLevelDims( mipLevel );
            unsigned int numTilesX    = mipLevelDims.x / firstTexture.getTileWidth();
            unsigned int numTilesY    = mipLevelDims.y / firstTexture.getTileHeight();

            for( unsigned int tileY = 0; tileY < numTilesY; ++tileY )
            {
                for( unsigned int tileX = 0; tileX < numTilesX; ++tileX )
                {
                    // Fill this tile in one of the textures, using textures and streams in round-robin fashion.
                    TileRequest request{&m_textures.at( whichTexture ), mipLevel, tileX, tileY, m_streams.at( whichStream )};
                    whichTexture = ( whichTexture + 1 ) % m_textures.size();
                    whichStream  = ( whichStream + 1 ) % m_streams.size();
                    requests.push_back( request );
                }
            }
        }
        std::random_shuffle( requests.begin(), requests.end() );
        return requests;
    }


    std::vector<CUstream>      m_streams;
    std::vector<SparseTexture> m_textures;
};

TEST_F( TestTilePool, TestSequentialFill )
{
    // Initialize TileFiller.
    Options options;
    options.maxTexMemPerDevice = 1024 * 1024 * 1024;
    TileFiller filler( options );

    // Generate and fill requests.
    std::vector<TileRequest> requests( createRequests() );
    for( const TileRequest& request : requests )
    {
        filler.fillTile( request );
    }
    EXPECT_EQ( 0, filler.getNumErrors() );
}

// ---------- Parallel fill helper classes ----------

// A thread-safe queue of TileRequests with a condition variable.
class RequestQueue
{
  public:
    bool popOrWait( TileRequest* requestPtr )
    {
        // Wait until the queue is non-empty or destroyed.
        std::unique_lock<std::mutex> lock( m_mutex );
        m_requestAvailable.wait( lock, [this] { return !m_requests.empty() || m_isShutDown; } );

        if( m_isShutDown )
            return false;

        *requestPtr = m_requests.front();
        m_requests.pop_front();

        if( m_requests.empty() )
        {
            m_isEmpty.notify_all();
        }

        return true;
    }

    void push( TileRequest* requests, size_t numRequests )
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        // Don't push requests if the queue is shut down.
        if( m_isShutDown )
            return;

        for( size_t i = 0; i < numRequests; ++i )
        {
            m_requests.push_back( requests[i] );
        }

        // Notify any threads in popOrWait().
        m_requestAvailable.notify_all();
    }

    void waitUntilEmpty()
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_isEmpty.wait( lock, [this] { return m_requests.empty(); } );
    }

    void shutDown()
    {
        {
            std::unique_lock<std::mutex> lock( m_mutex );
            m_isShutDown = true;
        }
        m_requestAvailable.notify_all();
    }

  private:
    std::deque<TileRequest> m_requests;
    std::mutex              m_mutex;
    std::condition_variable m_requestAvailable;
    std::condition_variable m_isEmpty;
    bool                    m_isShutDown = false;
};

// A simple multi-threaded request processor.
class RequestProcessor
{
  public:
    RequestProcessor( TileFiller* filler )
        : m_filler( filler )
    {
    }

    void start()
    {
        unsigned int numThreads = std::max( std::thread::hardware_concurrency(), 1U );

        m_threads.reserve( numThreads );
        for( unsigned int i = 0; i < numThreads; ++i )
        {
            m_threads.emplace_back( &RequestProcessor::worker, this );
        }
    }

    void waitUntilDone() { m_requests.waitUntilEmpty(); }

    void stop()
    {
        // Any threads that are waiting in RequestQueue::popOrWait will be notified when the queue is
        // shut down.
        m_requests.shutDown();
        for( std::thread& thread : m_threads )
        {
            thread.join();
        }
    }

    void addRequests( TileRequest* requests, size_t numRequests ) { m_requests.push( requests, numRequests ); }

  private:
    TileFiller*              m_filler;
    std::vector<std::thread> m_threads;
    RequestQueue             m_requests;

    void worker()
    {
        // Initialize CUDA for this thread.
        cudaFree( nullptr );

        TileRequest request;
        while( true )
        {
            // Pop a request from the queue, waiting if necessary until the queue is non-empty or shut down.
            if( !m_requests.popOrWait( &request ) )
                return;  // Exit thread when queue is shut down.

            m_filler->fillTile( request );
        }
    }
};

TEST_F( TestTilePool, TestParallelFill )
{
    // Initialize tile filler.
    Options options;
    options.maxTexMemPerDevice = 1024 * 1024 * 1024;
    TileFiller filler( options );

    // Initialize multi-threaded request processor.
    RequestProcessor processor( &filler );
    processor.start();

    // Generate requests.
    std::vector<TileRequest> requests( createRequests() );

    // Process requests.
    processor.addRequests( requests.data(), requests.size() );
    processor.waitUntilDone();
    processor.stop();
    EXPECT_EQ( 0, filler.getNumErrors() );
}
