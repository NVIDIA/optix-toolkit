//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

#include <cuda.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <vector>

#define NullAllocator HostAllocator
#define NullSuballocator HeapSuballocator
const uint64_t DEFAULT_HEADROOM = 128ULL << 20;

namespace otk {

// MemoryPool is a thread-safe memory pool class that allocates and tracks memory using an allocator and suballocator.
// Memory blocks can be freed either immediately or in stream order.  This class can be used to manage general device memory,
// device memory allocated with cuMallocAsync/cuFreeAsync, texture tiles, pinned host memory, and standard host memory.
//
template <class Allocator, class SubAllocator>
class MemoryPool
{
  public:
    /// Constructor specifying both allocator and suballocator, one of which could be nullptr
    MemoryPool( Allocator* allocator, SubAllocator* suballocator, uint64_t allocationGranularity = DEFAULT_ALLOC_SIZE,
                uint64_t maxSize = 0, uint64_t headroom = DEFAULT_HEADROOM )
        : m_allocator( allocator )
        , m_suballocator( suballocator )
        , m_allocationGranularity( allocationGranularity )
        , m_maxSize( maxSize ? maxSize : std::numeric_limits<uint64_t>::max() )
        , m_headroom( headroom )
    {
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );
        OTK_ASSERT( m_context != nullptr );
    }

    /// Constructor for when the suballocator has a default constructor
    MemoryPool( Allocator* allocator, uint64_t allocationGranularity = DEFAULT_ALLOC_SIZE,
                uint64_t maxSize = 0, uint64_t headroom = DEFAULT_HEADROOM )
        : MemoryPool( allocator, new SubAllocator(), allocationGranularity, maxSize, headroom )
    {
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );
        OTK_ASSERT( m_context != nullptr );
    }

    /// Constructor for when the allocator has a default constructor
    MemoryPool( SubAllocator* suballocator, uint64_t allocationGranularity = DEFAULT_ALLOC_SIZE,
                uint64_t maxSize = 0, uint64_t headroom = DEFAULT_HEADROOM )
        : MemoryPool( new Allocator(), suballocator, allocationGranularity, maxSize, headroom )
    {
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );
        OTK_ASSERT( m_context != nullptr );
    }

    /// Constructor for when both the allocator and suballocator have default constructors
    MemoryPool( uint64_t allocationGranularity = DEFAULT_ALLOC_SIZE,
                uint64_t maxSize = 0, uint64_t headroom = DEFAULT_HEADROOM )
        : MemoryPool( new Allocator(), new SubAllocator(), allocationGranularity, maxSize, headroom )
    {
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );
        OTK_ASSERT( m_context != nullptr );
    }

    /// Move constructor
    MemoryPool( MemoryPool&& p )
        : MemoryPool( p.m_allocator, p.m_suballocator, p.m_allocationGranularity, p.m_maxSize, p.m_headroom )
    {
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );
        OTK_ASSERT( m_context != nullptr );
        p.m_allocator    = nullptr;
        p.m_suballocator = nullptr;
    }

    /// Destructor
    ~MemoryPool()
    {
        OTK_ERROR_CHECK_NOTHROW( cuCtxPushCurrent( m_context ) );

        std::unique_lock<std::mutex> lock( m_mutex );

        try
        {
            for( void* ptr : m_allocations )
                m_allocator->free( ptr );
            delete m_suballocator;
            delete m_allocator;

            // Destroy events in staged blocks.
            for( StagedBlock stagedBlock : m_stagedBlocks )
                freeEvent( stagedBlock );
            m_stagedBlocks.clear();
        }
        catch(...)
        {
        }

        CUcontext ignored;
        OTK_ERROR_CHECK_NOTHROW( cuCtxPopCurrent( &ignored ) );
    }

    /// Tell the memory pool to track an address range, bypassing the allocator, which may be null
    void track( uint64_t ptr, uint64_t size ) { m_suballocator->track( ptr, size ); }

    /// Allocate a memory block with (at least) the given size and alignment. Returns BAD_ADDR on failure.
    MemoryBlockDesc alloc( uint64_t size = 0, uint64_t alignment = 1, CUstream stream = 0 )
    {
        OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );

        std::unique_lock<std::mutex> lock( m_mutex );
        freeStagedBlocks( false );
        size = ( size ) ? size : m_allocationGranularity;

        // If no suballocator, use the allocator directly
        if( !m_suballocator )
            return MemoryBlockDesc{reinterpret_cast<uint64_t>( m_allocator->allocate( size, stream ) ), size, 0};

        // Try to fill the request with the suballocator. If it fails, allocate more memory and try again.
        MemoryBlockDesc block = m_suballocator->alloc( size, alignment );
        if( ( block.isBad() ) && ( trackedSize() < m_maxSize ) && m_allocator )
        {
            // Make sure there is enough headroom (allocatable space) still on the card
            if( m_headroom > 0 )
            {
                size_t freeMem, totalMem;
                OTK_ERROR_CHECK( cuMemGetInfo( &freeMem, &totalMem ) );
                if( freeMem < m_headroom )
                    return block;
            }

            // Allocate enough memory for the current request at m_allocationGranularity increments.
            size_t allocSize = m_allocationGranularity * ( ( size + m_allocationGranularity - 1 ) / m_allocationGranularity );
            void* ptr = m_allocator->allocate( allocSize );
            if( !ptr )
                return block;
            m_allocations.push_back( ptr );

            if( m_allocator->allocationIsHandle() )
            {
                // If the allocator returns handles, they are not pointers in a linear memory space, so
                // construct an artificial linear memory space for the suballocator to use.
                m_suballocator->track( getArenaStartAddress( static_cast<uint64_t>( m_allocations.size() - 1 ) ), allocSize );
            }
            else
            {
                m_suballocator->track( reinterpret_cast<uint64_t>( m_allocations.back() ), allocSize );
            }
            block = m_suballocator->alloc( size, alignment );
        }

        // If the allocation failed, wait on all the staged blocks and try the suballocator one last time
        if( block.isBad() )
        {
            freeStagedBlocks( true );
            block = m_suballocator->alloc( size, alignment );
        }

        return block;
    }

    /// Allocate a single item. Works with FixedSuballocator.
    uint64_t allocItem( CUstream stream = 0 ) { return alloc( m_suballocator->itemSize(), 1, stream ).ptr; }

    /// Allocate an object of a given type, returning a pointer to it
    template <typename TYPE>
    TYPE* allocObject( CUstream stream = 0 )
    {
        return reinterpret_cast<TYPE*>( alloc( sizeof( TYPE ), alignof( TYPE ), stream ).ptr );
    }

    /// Allocate an array of objects, returning a pointer to the array.
    template <typename TYPE>
    TYPE* allocObjects( size_t numItems, CUstream stream = 0 )
    {
        return reinterpret_cast<TYPE*>( alloc( sizeof( TYPE ) * numItems, alignof( TYPE ), stream ).ptr );
    }


    /// Allocate a number of texture tiles.  Works with TextureTileAllocator.
    TileBlockHandle allocTextureTiles( uint64_t sizeInBytes )
    {
        MemoryBlockDesc              block = alloc( sizeInBytes, TILE_SIZE_IN_BYTES, 0 );
        std::unique_lock<std::mutex> lock( m_mutex );
        const bool                   blockGood = block.isGood();

        unsigned int   arenaId  = blockGood ? (unsigned int)getArenaId( block ) : 0;
        unsigned short tileId   = blockGood ? (unsigned short)( getArenaOffset( block ) / TILE_SIZE_IN_BYTES ) : 0;
        unsigned short numTiles = blockGood ? (unsigned short)( block.size / TILE_SIZE_IN_BYTES ) : 0;

        CUmemGenericAllocationHandle handle =
            blockGood ? reinterpret_cast<CUmemGenericAllocationHandle>( m_allocations[arenaId] ) : 0;

        return TileBlockHandle{handle, {arenaId, tileId, numTiles}};
    }

    // Get the allocation handle backing a block of texture tiles
    uint64_t getAllocationHandle( unsigned int arenaId )
    {
        return reinterpret_cast<CUmemGenericAllocationHandle>( m_allocations[arenaId] );
    }

    /// Free block immediately on the specified stream.
    void free( const MemoryBlockDesc& block, CUstream stream = 0 )
    {
        OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );

        std::unique_lock<std::mutex> lock( m_mutex );
        if( m_suballocator )
            m_suballocator->free( block );
        else
            m_allocator->free( reinterpret_cast<void*>( block.ptr ), stream );
    }

    /// Free a single item (used with FixedSuballocator).
    void freeItem( uint64_t ptr ) { free( MemoryBlockDesc{ptr, m_suballocator->itemSize(), 0} ); }

    /// Free an object (not compatible with RingSuballocator).
    template <typename TYPE>
    void freeObject( TYPE* ptr )
    {
        free( MemoryBlockDesc{reinterpret_cast<uint64_t>( ptr ), sizeof( TYPE ), 0} );
    }

    /// Free an array of objects (not compatible with RingSuballocator)
    template <typename TYPE>
    void freeObjects( TYPE* ptr, uint64_t numObjects )
    {
        free( MemoryBlockDesc{reinterpret_cast<uint64_t>( ptr ), numObjects * sizeof( TYPE ), 0} );
    }

    /// Free texture tiles immediately
    void freeTextureTiles( const TileBlockDesc& tileBlock )
    {
        uint64_t ptr  = tileBlock.arenaId * getArenaSpacing() + tileBlock.tileId * TILE_SIZE_IN_BYTES;
        uint64_t size = tileBlock.numTiles * TILE_SIZE_IN_BYTES;
        free( MemoryBlockDesc{ptr, size, 0} );
    }

    /// Free block asynchronously, after operations currently in the stream have finished
    void freeAsync( const MemoryBlockDesc& block, CUstream stream )
    {
        OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );

        CUcontext context;
        OTK_ERROR_CHECK( cuCtxGetCurrent( &context ) );

        // Create event.
        CUevent event;
        OTK_ERROR_CHECK( cuEventCreate( &event, CU_EVENT_DEFAULT ) );

        std::unique_lock<std::mutex> lock( m_mutex );
        freeStagedBlocks( false );
        m_stagedBlocks.push_back( StagedBlock{context, block, event} );

        // Record event.
        OTK_ERROR_CHECK( cuEventRecord( m_stagedBlocks.back().event, stream ) );
    }

    /// Async free of a single address slot (not compatible with RingSuballocator)
    void freeItemAsync( uint64_t ptr, CUstream stream = 0 )
    {
        freeAsync( MemoryBlockDesc{ptr, m_suballocator->itemSize(), 0}, stream );
    }

    /// Async free an object (not compatible with RingSuballocator)
    template <typename TYPE>
    void freeObjectAsync( TYPE* ptr, CUstream stream = 0 )
    {
        freeAsync( MemoryBlockDesc{reinterpret_cast<uint64_t>( ptr ), sizeof( TYPE ), 0}, stream );
    }

    /// Free an array of objects (not compatible with RingSuballocator)
    template <typename TYPE>
    void freeObjectsAsync( TYPE* ptr, uint64_t numObjects, CUstream stream = 0 )
    {
        freeAsync( MemoryBlockDesc{reinterpret_cast<uint64_t>( ptr ), numObjects * sizeof( TYPE ), 0}, stream );
    }

    /// Async free of texture tiles
    void freeTextureTilesAsync( const TileBlockDesc& tileBlock, CUstream stream = 0 )
    {
        uint64_t ptr  = tileBlock.arenaId * getArenaSpacing() + tileBlock.tileId * TILE_SIZE_IN_BYTES;
        uint64_t size = tileBlock.numTiles * TILE_SIZE_IN_BYTES;
        freeAsync( MemoryBlockDesc{ptr, size, 0}, stream );
    }

    /// Return the free space currently tracked in the pool
    uint64_t currentFreeSpace() const { return m_suballocator ? m_suballocator->freeSpace() : 0; }

    /// Return the amount of space that can be allocated in the pool without freeing anything
    uint64_t allocatableSpace() const { return currentFreeSpace() + maxSize() - std::min( maxSize(), trackedSize() ); }

    /// Return the amount of memory currently tracked (free or given out) by the pool
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
        CUcontext       context;
        MemoryBlockDesc block;
        CUevent         event;
    };

    CUcontext          m_context;
    Allocator*         m_allocator;
    SubAllocator*      m_suballocator;
    std::vector<void*> m_allocations;
    uint64_t           m_allocationGranularity;
    uint64_t           m_maxSize;
    uint64_t           m_headroom; // Leave at least this much memory unallocated
    mutable std::mutex m_mutex;

    std::deque<StagedBlock> m_stagedBlocks;

    // Free blocks with events that have finished
    inline void freeStagedBlocks( bool waitOnEvents )
    {
        while( !m_stagedBlocks.empty() )
        {
            if( cuEventQuery( m_stagedBlocks.front().event ) == CUDA_ERROR_NOT_READY )
            {
                if( waitOnEvents )
                    cuEventSynchronize( m_stagedBlocks.front().event );
                else
                    break;
            }

            if( m_suballocator )
                m_suballocator->free( m_stagedBlocks.front().block );
            else
                m_allocator->free( reinterpret_cast<void*>( m_stagedBlocks.front().block.ptr ) );

            freeEvent( m_stagedBlocks.front() );
            m_stagedBlocks.pop_front();
        }
    }

    void freeEvent( const StagedBlock& stagedBlock )
    {
        OTK_ERROR_CHECK( cuCtxPushCurrent( stagedBlock.context ) );
        OTK_ERROR_CHECK( cuEventDestroy( stagedBlock.event ) );
        CUcontext ignored;
        OTK_ERROR_CHECK( cuCtxPopCurrent( &ignored ) );
    }

    // Get the spacing between arenas for handle-based allocations
    uint64_t getArenaSpacing() { return 2 * m_allocationGranularity; }

    // Get the start of an arena in an artificial linear memory space for a handle
    uint64_t getArenaStartAddress( uint64_t arenaId ) { return arenaId * getArenaSpacing(); }

    // Get the arena id for a given memory block when allocations are handles (such as textures)
    uint64_t getArenaId( const MemoryBlockDesc& block ) { return block.ptr / getArenaSpacing(); }

    // Get the offset within a texture tile arena for a given memory block when allocations are handles
    uint64_t getArenaOffset( const MemoryBlockDesc& block ) { return block.ptr % getArenaSpacing(); }
};

}  // namespace otk
