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

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <vector>

#include <Memory/Allocators.h>
#include <Memory/MemoryBlockDesc.h>
#include <Util/Exception.h>

namespace demandLoading {

// MemoryPool is a thread-safe memory pool class that allocates and tracks memory using an allocator and suballocator.
// Memory blocks can be freed either immediately or in stream order.  This class can be used to manage general device memory,
// device memory allocated with cudaMallocAsync/cudaFreeAsync, texture tiles, pinned host memory, and standard host memory.
//
template <class Allocator, class SubAllocator>
class MemoryPool
{
  public:
    /// Constructor specifying both allocator and suballocator
    MemoryPool( Allocator* allocator, SubAllocator* suballocator, uint64_t allocationGranularity = 0, uint64_t maxSize = 0 )
        : m_allocator( allocator )
        , m_suballocator( suballocator )
        , m_allocationGranularity( allocationGranularity )
        , m_maxSize( maxSize ? maxSize : std::numeric_limits<uint64_t>::max() )
    {
        const unsigned int MAX_DEVICES = 32;
        m_eventPool.resize( MAX_DEVICES );
    }

    /// Constructor for when the suballocator is null
    MemoryPool( Allocator* allocator, uint64_t allocationGranularity = 0, uint64_t maxSize = 0 )
        : MemoryPool( allocator, nullptr, allocationGranularity, maxSize )
    {
    }

    /// Constructor for when the allocator is null
    MemoryPool( SubAllocator* suballocator, uint64_t allocationGranularity = 0, uint64_t maxSize = 0 )
        : MemoryPool( nullptr, suballocator, allocationGranularity, maxSize )
    {
    }

    /// Move constructor
    MemoryPool( MemoryPool&& p ) : MemoryPool( p.m_allocator, p.m_suballocator, p.m_allocationGranularity, p.m_maxSize )
    {
       p.m_allocator = nullptr;
       p.m_suballocator = nullptr;
    }
    
    /// Destructor
    ~MemoryPool()
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        for( void* ptr : m_allocations )
            m_allocator->free( ptr );
        delete m_suballocator;
        delete m_allocator;

        // Put all the events back in the event pool
        for( StagedBlock stagedBlock : m_stagedBlocks )
            m_eventPool[stagedBlock.deviceIndex].push_back( stagedBlock.event );
        m_stagedBlocks.clear();

        // Destroy all the events
        for( unsigned int deviceIndex = 0; deviceIndex < m_eventPool.size(); ++deviceIndex )
        {
            if( !m_eventPool[deviceIndex].empty() )
            {
                DEMAND_CUDA_CHECK_NOTHROW( cudaSetDevice( deviceIndex ) );
                for( CUevent event : m_eventPool[deviceIndex] )
                {
                    DEMAND_CUDA_CHECK_NOTHROW( cudaEventDestroy( event ) );
                }
            }
        }
    }

    /// Tell the memory pool to track an address range, bypassing the allocator, which may be null
    void track( uint64_t ptr, uint64_t size ) { m_suballocator->track( ptr, size ); }

    /// Allocate a memory block with (at least) the given size and alignment. Returns BAD_ADDR on failure.
    MemoryBlockDesc alloc( uint64_t size = 0, uint64_t alignment = 1, CUstream stream = 0 )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        freeStagedBlocks();
        size = ( size ) ? size : m_allocationGranularity;

        // If no suballocator, use the allocator directly
        if( !m_suballocator )
            return MemoryBlockDesc{reinterpret_cast<uint64_t>( m_allocator->allocate( size, stream ) ), size};

        // Try to fill the request with the suballocator.  If it fails, allocate additional memory with the
        // allocator and try again.
        MemoryBlockDesc block = m_suballocator->alloc( size, alignment );
        if( ( block.isBad() ) && ( trackedSize() < m_maxSize ) && m_allocator )
        {
            m_allocations.push_back( m_allocator->allocate( m_allocationGranularity ) );
            if( m_allocator->allocationIsHandle() )
            {
                // If the allocator returns handles, they are not pointers in a linear memory space, so
                // construct an artificial linear memory space for the suballocator to use.
                m_suballocator->track( getArenaStartAddress( static_cast<uint64_t>( m_allocations.size() - 1 ) ), m_allocationGranularity );
            }
            else
            {
                m_suballocator->track( reinterpret_cast<uint64_t>( m_allocations.back() ), m_allocationGranularity );
            }
            block = m_suballocator->alloc( size, alignment );
        }
        return block;
    }

    /// Allocate a fixed size item, retaining a pointer to it. Works with FixedSuballocator
    uint64_t allocItem() { return alloc( m_suballocator->itemSize() ).ptr; }

    /// Allocate a number of texture tiles.  Works with TextureTileAllocator.
    TileBlockHandle allocTextureTiles( uint64_t sizeInBytes )
    {
        MemoryBlockDesc block     = alloc( sizeInBytes, TILE_SIZE_IN_BYTES, 0 );
        std::unique_lock<std::mutex> lock( m_mutex );
        const bool      blockGood = block.isGood();

        unsigned int   arenaId  = blockGood ? (unsigned int)getArenaId( block ) : 0;
        unsigned short tileId   = blockGood ? (unsigned short)( getArenaOffset( block ) / TILE_SIZE_IN_BYTES ) : 0;
        unsigned short numTiles = blockGood ? (unsigned short)( block.size / TILE_SIZE_IN_BYTES ) : 0;

        CUmemGenericAllocationHandle handle =
            blockGood ? reinterpret_cast<CUmemGenericAllocationHandle>( m_allocations[arenaId] ) : 0;
        return TileBlockHandle{handle, {arenaId, tileId, numTiles}};
    }

    /// Free block immediately on the specified stream.
    void free( const MemoryBlockDesc& block, CUstream stream = 0 )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        if( m_suballocator )
            m_suballocator->free( block );
        else
            m_allocator->free( reinterpret_cast<void*>( block.ptr ), stream );
    }

    /// Free a fixed size item immediately from a pointer to it
    void freeItem( uint64_t ptr ) { free( MemoryBlockDesc{ptr, m_suballocator->itemSize()} ); }

    /// Free texture tiles immediately
    void freeTextureTiles( const TileBlockDesc& tileBlock )
    {
        uint64_t ptr  = tileBlock.arenaId * getArenaSpacing() + tileBlock.tileId * TILE_SIZE_IN_BYTES;
        uint64_t size = tileBlock.numTiles * TILE_SIZE_IN_BYTES;
        free( MemoryBlockDesc{ptr, size} );
    }

    /// Free block asynchronously, after operations currently in the stream have finished
    void freeAsync( const MemoryBlockDesc& block, unsigned int deviceIndex, CUstream stream )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        freeStagedBlocks();
        m_stagedBlocks.push_back( StagedBlock{block, getEvent( deviceIndex ), deviceIndex} );
        DEMAND_CUDA_CHECK( cudaEventRecord( m_stagedBlocks.back().event, stream ) );
    }

    /// Async free of a fixed size item
    void freeItemAsync( uint64_t ptr, unsigned int deviceIndex, CUstream stream = 0 )
    {
        freeAsync( MemoryBlockDesc{ptr, m_suballocator.itemSize()}, deviceIndex, stream );
    }

    /// Async free of texture tiles
    void freeTextureTilesAsync( const TileBlockDesc& tileBlock, unsigned int deviceIndex, CUstream stream = 0 )
    {
        uint64_t ptr  = tileBlock.arenaId * getArenaSpacing() + tileBlock.tileId * TILE_SIZE_IN_BYTES;
        uint64_t size = tileBlock.numTiles * TILE_SIZE_IN_BYTES;
        freeAsync( MemoryBlockDesc{ptr, size}, deviceIndex, stream );
    }

    /// Return the free space currently tracked in the pool
    uint64_t currentFreeSpace() const { return m_suballocator ? m_suballocator->freeSpace() : 0; }

    /// Return the amount of space that can be allocated in the pool without freeing anything
    uint64_t allocatableSpace() const { return ( maxSize() - trackedSize() ) + currentFreeSpace(); }

    /// Return the amount of memory currently tracked (free or giving out) by the pool
    uint64_t trackedSize() const { return m_suballocator ? m_suballocator->trackedSize() : 0; }

    /// Return the maximum memory that the pool will allocate
    uint64_t maxSize() const { return m_maxSize; }

    /// Return the size of chunks that the allocator will allocate.
    /// Also indicates that largest block that the pool can allocate.
    uint64_t allocationGranularity() const { return m_allocationGranularity; }

    /// Set the max size (maximum size that the pool will allocate)
    void setMaxSize( uint64_t maxSize ) { m_maxSize = maxSize; }

  private:
    // A memory pool is for a single device or the host, but it must
    // be able to wait on all devices because a pinned memory blocks may have to wait
    // on a stream on any device before they are freed.
    struct StagedBlock
    {
        MemoryBlockDesc block;
        CUevent         event;
        unsigned int    deviceIndex;
    };

    Allocator*         m_allocator;
    SubAllocator*      m_suballocator;
    std::vector<void*> m_allocations;
    uint64_t           m_allocationGranularity;
    uint64_t           m_maxSize;
    mutable std::mutex m_mutex;

    std::deque<StagedBlock>           m_stagedBlocks;
    std::vector<std::vector<CUevent>> m_eventPool;

    // Get an event from the internal event pool
    inline CUevent getEvent( unsigned int deviceIndex )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );

        if( m_eventPool[deviceIndex].empty() )
        {
            m_eventPool[deviceIndex].emplace_back();
            DEMAND_CUDA_CHECK( cudaEventCreate( &m_eventPool[deviceIndex].back() ) );
        }
        CUevent event = m_eventPool[deviceIndex].back();
        m_eventPool[deviceIndex].pop_back();

        return event;
    }

    // Free blocks with events that have finished
    inline void freeStagedBlocks()
    {
        while( !m_stagedBlocks.empty() && ( cudaEventQuery( m_stagedBlocks.front().event ) != cudaErrorNotReady ) )
        {
            if( m_suballocator )
                m_suballocator->free( m_stagedBlocks.front().block );
            else
                m_allocator->free( reinterpret_cast<void*>( m_stagedBlocks.front().block.ptr ) );

            m_eventPool[m_stagedBlocks.front().deviceIndex].push_back( m_stagedBlocks.front().event );
            m_stagedBlocks.pop_front();
        }
    }

    // Get the spacing between arenas for handle-based allocations
    uint64_t getArenaSpacing() { return 2 * m_allocationGranularity; }

    // Get the start of an arena in an artificial linear memory space for a handle
    uint64_t getArenaStartAddress( uint64_t arenaId ) { return arenaId * getArenaSpacing(); }

    // Get the arena id for a given memory block when allocations are handles (such as textures)
    uint64_t getArenaId( const MemoryBlockDesc& block ) { return block.ptr / getArenaSpacing(); }

    // Get the offset within a texture tile arena for a given memory block when allocations are handles
    uint64_t getArenaOffset( const MemoryBlockDesc& block ) { return block.ptr % getArenaSpacing(); }

    // Get the cuda memory allocation for a memory block when allocations are handles
    uint64_t getHandle( const MemoryBlockDesc& block ) { return m_allocations[getArenaId( block )]; }
};

/// specialization of MemoryPool with null allocator
template <class Suballocator>
using AddressPool = MemoryPool<HostAllocator, Suballocator>;

}  // namespace demandLoading
