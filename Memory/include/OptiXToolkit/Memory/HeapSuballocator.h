// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Memory/MemoryBlockDesc.h>
#include <algorithm>
#include <map>

#include <stdio.h>

namespace otk {

// HeapSuballocator is used to track free blocks from an address space.
//
// Blocks are defined by an address and a size. If a block is
// allocated, it is not tracked.
//
// The track() function tells the suballocator to track a block of memory.
// New blocks can be added at any time, but should not overlap existing
// tracked blocks.
//
// The alloc() function finds a sub-block to fit the allocation, splitting
// blocks as needed to fill the allocation size and alignment.
// The free() function frees a block. When a block is freed, adjacent blocks
// are merged. Alloc and free work in O(log n) time, where n is the number
// of free blocks in the heap.
//
class HeapSuballocator
{
  public:
    HeapSuballocator()  = default;
    ~HeapSuballocator() = default;

    /// Tell the suballocator to track a memory segment.
    /// This can be called multiple times to track multiple segments.
    void track( uint64_t ptr, uint64_t size )
    {
        m_trackedSize += size;
        free( MemoryBlockDesc{ptr, size, 0} );
    }

    /// Allocate a block from tracked memory. Returns the address to the allocated block.
    /// On failure, BAD_ADDR is returned in the memory block.
    MemoryBlockDesc alloc( uint64_t size, uint64_t alignment = 1 );

    /// Free a block. The size must be correct to ensure correctness.
    void free( const MemoryBlockDesc& memBlock );

    /// Untrack memory that is currently tracked by the suballocator.
    void untrack( uint64_t ptr, uint64_t size );

    /// Return the current free space
    uint64_t freeSpace() const { return m_freeSpace; }

    /// Return the total memory tracked by suballocator
    uint64_t trackedSize() const { return m_trackedSize; }

    /// Return true if the internal structure is valid (all blocks have non-zero size and none overlap)
    bool validate();

    /// return the begin map for testing purposes
    const std::map<uint64_t, uint64_t>& getBeginMap() { return m_beginMap; }

  private:
    uint64_t m_trackedSize    = 0;  // Total memory tracked by the suballocator
    uint64_t m_freeSpace      = 0;  // Current free memory available
    uint64_t m_gteLargestFree = 0;  // This is >= (gte) size of largest free block
    uint64_t m_startPos       = 0;  // Where to start the search for a free block in alloc

    std::map<uint64_t, uint64_t> m_beginMap;  // Free blocks indexed by beginning address
};

inline MemoryBlockDesc HeapSuballocator::alloc( uint64_t size, uint64_t alignment )
{
    alignment = std::max( alignment, static_cast<uint64_t>( 1 ) );

    // Can't allocate 0 size, or something larger than the largest free block
    if( size == 0 || size > m_gteLargestFree )
        return MemoryBlockDesc{BAD_ADDR, 0, 0};

    // Start search at the start position
    uint64_t largestFreeSize = 0;
    auto     blockIt         = m_beginMap.lower_bound( m_startPos );

    for( int i = static_cast<int>( m_beginMap.size() ); i > 0; --i )
    {
        // Wrap around to the beginning of the map
        if( blockIt == m_beginMap.end() )
            blockIt = m_beginMap.begin();

        uint64_t blockBegin = blockIt->first;
        uint64_t blockSize  = blockIt->second;
        uint64_t blockEnd   = blockBegin + blockSize;

        //uint64_t usedBegin = alignVal( blockBegin, alignment ); // alloc at beginning of block
        uint64_t usedBegin = alignVal( blockEnd - size - alignment + 1, alignment );  // alloc at end of block (faster)

        if( ( usedBegin >= blockBegin ) && ( usedBegin < blockEnd ) && ( usedBegin + size <= blockEnd ) )
        {
            m_startPos = blockIt->first;
            m_freeSpace -= size;

            if( usedBegin != blockBegin )  // Alignment does not fall on block beginning, so split
            {
                blockIt->second  = usedBegin - blockBegin;
                uint64_t newSize = blockSize - ( size + ( usedBegin - blockBegin ) );
                if( newSize != 0 )
                    m_beginMap[usedBegin + size] = newSize;
            }
            else if( blockSize == size )  // The block size is exactly the right size, so erase it
            {
                m_beginMap.erase( blockIt );
            }
            else  // The block is bigger than needed, so add end as new block
            {
                m_beginMap.erase( blockIt );
                m_beginMap[blockBegin + size] = blockSize - size;
            }
            return MemoryBlockDesc{usedBegin, size, 0};
        }

        largestFreeSize = std::max( largestFreeSize, blockIt->second );
        ++blockIt;
    }

    m_gteLargestFree = largestFreeSize;
    return MemoryBlockDesc{BAD_ADDR, 0, 0};
}

inline void HeapSuballocator::free( const MemoryBlockDesc& memBlock )
{
    const uint64_t start = memBlock.ptr;
    const uint64_t size  = memBlock.size;
    m_freeSpace += size;

    // Special case for empty map
    if( m_beginMap.empty() )
    {
        m_beginMap[start] = size;
        m_gteLargestFree  = size;
        return;
    }

    // Find iterators that straddle free block
    auto nextIt = m_beginMap.lower_bound( start );
    auto prevIt = nextIt;
    if( prevIt != m_beginMap.begin() )
        --prevIt;

    // Merge with previous block or create new block
    if( prevIt->first + prevIt->second == start )
    {
        prevIt->second += size;
        m_gteLargestFree = std::max( m_gteLargestFree, prevIt->second );
    }
    else
    {
        m_beginMap[start] = size;
        prevIt            = m_beginMap.find( start );
        m_gteLargestFree  = std::max( m_gteLargestFree, size );
    }

    // Merge with next block if needed
    if( nextIt != m_beginMap.end() && ( start + size == nextIt->first ) )
    {
        if( nextIt->first == m_startPos )
            m_startPos = prevIt->first;
        prevIt->second += nextIt->second;
        m_beginMap.erase( nextIt );
        m_gteLargestFree = std::max( m_gteLargestFree, prevIt->second );
    }
}

inline void HeapSuballocator::untrack( uint64_t ptr, uint64_t size )
{
    // Remove free blocks that are in the untracked region
    auto it = m_beginMap.lower_bound( ptr );
    while( it != m_beginMap.end() && it->first < ( ptr + size ) )
    {
        auto eraseIt = it;
        ++it;
        m_freeSpace -= eraseIt->second;
        m_beginMap.erase( eraseIt );
    }

    // Reduce the tracked size
    m_trackedSize = ( size <= m_trackedSize ) ? m_trackedSize - size : 0ULL;
}

inline bool HeapSuballocator::validate()
{
    for( auto blockIt = m_beginMap.begin(); blockIt != m_beginMap.end(); blockIt++ )
    {
        auto nextIt = blockIt;
        nextIt++;
        if( ( nextIt != m_beginMap.end() ) && ( blockIt->first + blockIt->second >= nextIt->first ) )
            return false;
    }
    return true;
}

}  // namespace otk
