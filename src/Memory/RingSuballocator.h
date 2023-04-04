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

#include <Memory/MemoryBlockDesc.h>
#include <Util/Exception.h>

#include <algorithm>
#include <deque>
#include <vector>

namespace demandLoading {

// RingSuballocator is designed to quickly allocate and free memory for transfers,
// similar to a traditional ring buffer, but allowing flexibility in the order in
// which memory blocks are freed.  To make things fast, individual allocations are not tracked.
// Instead, memory is divided into arenas, which are reference counted, making alloc and free
// constant time operations.
//
class RingSuballocator
{
  public:
    RingSuballocator( uint64_t arenaSize = DEFAULT_ALLOC_SIZE )
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

    std::vector<AllocCountArena> arenas;
    std::deque<unsigned int>     activeArenas;

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
        newArena.arenaId         = static_cast<unsigned int>( arenas.size() );
        newArena.isActive        = true;
        newArena.arenaStart      = ptr;
        newArena.arenaSize       = std::min( m_arenaSize, size );
        newArena.startPos        = newArena.arenaStart;
        newArena.numAllocs       = 0;

        ptr += newArena.arenaSize;
        size -= newArena.arenaSize;

        arenas.push_back( newArena );
        activeArenas.push_back( newArena.arenaId );
    }
}

inline MemoryBlockDesc RingSuballocator::alloc( uint64_t size, uint64_t alignment )
{
    alignment = std::max( alignment, static_cast<uint64_t>( 1 ) );

    // Can't allocate 0 size, or something larger than an arena
    if( size == 0 || size > m_arenaSize )
        return MemoryBlockDesc{BAD_ADDR};

    // Find the first arena with space to hold the allocation. Should be current or next arena.
    unsigned int numActiveArenas = static_cast<unsigned int>( activeArenas.size() );
    for( unsigned int i = 0; i < numActiveArenas; ++i )
    {
        unsigned int     arenaId = activeArenas.front();
        AllocCountArena& arena   = arenas[arenaId];
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
            activeArenas.pop_front();
            if( arena.numAllocs == 0 )
            {
                arena.startPos = arena.arenaStart;
                activeArenas.push_back( arena.arenaId );
            }
            else
            {
                m_freeSpace    = m_freeSpace + ( arena.startPos - arena.arenaStart ) - arena.arenaSize;
                arena.startPos = arena.arenaStart + arena.arenaSize;
                arena.isActive = false;
            }
        }
    }

    return MemoryBlockDesc{BAD_ADDR};
}

inline void RingSuballocator::decrementAllocCount( unsigned int arenaId, unsigned int inc )
{
    AllocCountArena& arena = arenas[arenaId];
    // This ties the allocators to util/Exception.  Do we want to do that?
    DEMAND_ASSERT_MSG( inc <= arena.numAllocs, "Too many free operations in RingSuballocator." );

    arena.numAllocs -= inc;
    if( arena.numAllocs == 0 )
    {
        m_freeSpace += ( arena.startPos - arena.arenaStart );
        arena.startPos = arena.arenaStart;
        if( !arena.isActive )
            activeArenas.push_back( arenaId );
        arena.isActive = true;
    }
}

inline void RingSuballocator::freeAll()
{
    for( unsigned int i = 0; i < arenas.size(); ++i )
    {
        if( arenas[i].numAllocs > 0 )
            decrementAllocCount( i, static_cast<unsigned int>( arenas[i].numAllocs ) );
    }
}

}  // namespace demandLoading
