// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <vector>

namespace otk {

// RingSuballocator is designed to quickly allocate and free memory for transfers,
// similar to a traditional ring buffer, but allowing flexibility in the order in
// which memory blocks are freed.  To make things fast, individual allocations are not tracked.
// Instead, memory is divided into arenas, which are reference counted, making alloc and free
// constant time operations.
//
class RingSuballocator
{
  public:
    RingSuballocator( uint64_t arenaSize = std::numeric_limits<uint64_t>::max() )
        : m_arenaSize( arenaSize ){};
    ~RingSuballocator() {}

    /// Tell the allocator to track a memory segment.
    /// This can be called multiple times to track multiple segments.
    void track( uint64_t ptr, uint64_t size );

    /// Allocate a block from tracked memory. Returns the address to the allocated block.
    /// Returns the arenaId, needed to free the memory. On failure, returns BAD_ADDR
    MemoryBlockDesc alloc( uint64_t size, uint64_t alignment );

    /// Free a block. The size must be correct to ensure correctness.
    void free( const MemoryBlockDesc& memBlock )
    {
        decrementAllocCount( static_cast<unsigned int>( memBlock.description ), 1 );
    }

    /// Decrement the allocation count. Note that this should coorespond to a
    /// block or blocks returned by alloc.
    void decrementAllocCount( unsigned int arenaId, unsigned int inc );

    /// Free all the arenas
    void freeAll();

    /// Untrack memory that is currently tracked by the suballocator.
    /// This only works when the RingSuballocator is not being used.
    void untrack( uint64_t ptr, uint64_t size );

    /// Return the total amount of free space
    uint64_t freeSpace() const { return m_freeSpace; }

    /// Return the total memory tracked by the pool
    uint64_t trackedSize() const { return m_trackedSize; }

  protected:
    struct AllocCountArena
    {
        unsigned int arenaId;
        bool         isActive;
        uint64_t     arenaStart;
        uint64_t     arenaSize;
        uint64_t     startPos;
        uint64_t     numAllocs;
    };

    std::vector<AllocCountArena> m_arenas;
    std::deque<unsigned int>     m_activeArenas;

    uint64_t m_arenaSize   = 0;  // The standard size of arenas (larger allocations logically split)
    uint64_t m_trackedSize = 0;  // Total memory tracked by the pool
    uint64_t m_freeSpace   = 0;  // Current free memory available
};

inline void RingSuballocator::track( uint64_t ptr, uint64_t size )
{
    m_trackedSize += size;
    m_freeSpace += size;

    // Break the single allocation into multiple arenas of m_arenaSize
    while( size > 0 )
    {
        AllocCountArena newArena = {};
        newArena.arenaId         = static_cast<unsigned int>( m_arenas.size() );
        newArena.isActive        = true;
        newArena.arenaStart      = ptr;
        newArena.arenaSize       = std::min( m_arenaSize, size );
        newArena.startPos        = newArena.arenaStart;
        newArena.numAllocs       = 0;

        ptr += newArena.arenaSize;
        size -= newArena.arenaSize;

        m_arenas.push_back( newArena );
        m_activeArenas.push_back( newArena.arenaId );
    }
}

inline MemoryBlockDesc RingSuballocator::alloc( uint64_t size, uint64_t alignment )
{
    alignment = std::max( alignment, static_cast<uint64_t>( 1 ) );

    // Can't allocate 0 size, or something larger than an arena
    if( size == 0 || size > m_arenaSize )
        return MemoryBlockDesc{BAD_ADDR, 0, 0};

    // Find the first arena with space to hold the allocation. Should be current or next arena.
    unsigned int numActiveArenas = static_cast<unsigned int>( m_activeArenas.size() );
    for( unsigned int i = 0; i < numActiveArenas; ++i )
    {
        unsigned int     arenaId = m_activeArenas.front();
        AllocCountArena& arena   = m_arenas[arenaId];
        uint64_t         ptr     = alignVal( arena.startPos, alignment );
        if( ptr + size <= arena.arenaStart + arena.arenaSize )  // If the allocation fits, return it
        {
            m_freeSpace    = m_freeSpace + ( arena.startPos - arena.arenaStart ) - ( ptr + size - arena.arenaStart );
            arena.startPos = ptr + size;
            arena.numAllocs++;
            return MemoryBlockDesc{ptr, size, arenaId};
        }
        else  // If the allocation does not fit, deactivate the arena and go to the next one
        {
            m_activeArenas.pop_front();
            if( arena.numAllocs == 0 )
            {
                arena.startPos = arena.arenaStart;
                m_activeArenas.push_back( arena.arenaId );
            }
            else
            {
                m_freeSpace    = m_freeSpace + ( arena.startPos - arena.arenaStart ) - arena.arenaSize;
                arena.startPos = arena.arenaStart + arena.arenaSize;
                arena.isActive = false;
            }
        }
    }

    return MemoryBlockDesc{BAD_ADDR, 0, 0};
}

inline void RingSuballocator::decrementAllocCount( unsigned int arenaId, unsigned int inc )
{
    AllocCountArena& arena = m_arenas[arenaId];
    OTK_ASSERT_MSG( inc <= arena.numAllocs, "Too many free operations in RingSuballocator." );

    arena.numAllocs -= inc;
    if( arena.numAllocs == 0 )
    {
        m_freeSpace += ( arena.startPos - arena.arenaStart );
        arena.startPos = arena.arenaStart;
        if( !arena.isActive )
            m_activeArenas.push_back( arenaId );
        arena.isActive = true;
    }
}

inline void RingSuballocator::freeAll()
{
    for( unsigned int i = 0; i < m_arenas.size(); ++i )
    {
        if( m_arenas[i].numAllocs > 0 )
            decrementAllocCount( i, static_cast<unsigned int>( m_arenas[i].numAllocs ) );
    }
}

inline void RingSuballocator::untrack( uint64_t ptr, uint64_t size )
{
    freeAll();

    // Remove arenas in region to untrack
    for( int id = m_arenas.size() - 1; id >= 0; --id )
    {
        const AllocCountArena& arena = m_arenas[id];
        if( ( ptr <= arena.arenaStart && ptr + size > arena.arenaStart ) ||
            ( arena.arenaStart <= ptr && arena.arenaStart + arena.arenaSize > ptr ) )
        {
            m_arenas[id] = m_arenas.back();
            m_arenas[id].arenaId = id;
            m_arenas.pop_back();
        }
    }

    // Reset all arenas to active
    m_activeArenas.clear();
    m_trackedSize = 0;
    m_freeSpace = 0;
    for( unsigned int id = 0; id < m_arenas.size(); ++id )
    {
        m_trackedSize += m_arenas[id].arenaSize;
        m_freeSpace += m_arenas[id].arenaSize;
        m_activeArenas.push_back( id );
    }
}

}  // namespace otk
