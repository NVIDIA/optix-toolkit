// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Memory/FixedSuballocator.h>
#include <OptiXToolkit/Memory/HeapSuballocator.h>
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

#include <functional>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <vector>

namespace otk {

// Memory pool allocates and tracks memory using an allocator and suballocator
//
class BinnedSuballocator
{
  public:
    /// Constructor, giving the item sizes for the fixed allocators, as well as how many items
    /// to chunk into a single allocation for each size.
    BinnedSuballocator( const std::vector<uint64_t>& itemSizes, const std::vector<uint64_t>& itemsPerChunk );
    ~BinnedSuballocator() {}

    /// Tell the pool to track a memory segment.
    /// This can be called multiple times to track multiple segments.
    void track( uint64_t ptr, uint64_t size );

    /// Allocate a block from tracked memory. Returns the address to the allocated block.
    /// On failure, BAD_ADDR is returned in the memory block.
    MemoryBlockDesc alloc( uint64_t size, uint64_t alignment = 1 );

    /// Free a block. The size must be correct to ensure correctness.
    void free( const MemoryBlockDesc& memBlock );

    /// Return the total amount of free space
    uint64_t freeSpace() const { return m_freeSpace; }

    /// Return the total memory tracked by pool
    uint64_t trackedSize() const { return m_trackedSize; }

  protected:
    std::vector<uint64_t>          m_itemSizes;
    std::vector<FixedSuballocator> m_fixedSuballocators;
    std::vector<uint64_t>          m_itemsPerChunk;
    HeapSuballocator               m_heapSuballocator;

    uint64_t m_freeSpace   = 0;
    uint64_t m_trackedSize = 0;
};

inline BinnedSuballocator::BinnedSuballocator( const std::vector<uint64_t>& itemSizes, const std::vector<uint64_t>& itemsPerChunk )
    : m_itemsPerChunk( itemsPerChunk )
{
    for( unsigned int i = 0; i < itemSizes.size(); ++i )
    {
        m_fixedSuballocators.emplace_back( itemSizes[i], itemSizes[i] );
        m_itemSizes.push_back( m_fixedSuballocators.back().itemSize() );
    }
}

inline void BinnedSuballocator::track( uint64_t ptr, uint64_t size )
{
    m_trackedSize += size;
    m_freeSpace += size;
    m_heapSuballocator.track( ptr, size );
}

inline MemoryBlockDesc BinnedSuballocator::alloc( uint64_t size, uint64_t alignment )
{
    // Try to allocate from one of the fixed size suballocators
    for( unsigned int i = 0; i < m_itemSizes.size(); ++i )
    {
        // Found the right size
        if( m_itemSizes[i] >= size )
        {
            // Try to get the memory block off the fixedSuballocator
            MemoryBlockDesc block = m_fixedSuballocators[i].alloc();
            if( block.ptr != BAD_ADDR )
            {
                m_freeSpace -= block.size;
                return block;
            }

            // If no block was available from the fixedSuballocator, try to give it a new chunk from the heap
            MemoryBlockDesc chunk = m_heapSuballocator.alloc( m_itemsPerChunk[i] * m_itemSizes[i], m_itemSizes[i] );
            if( chunk.ptr != BAD_ADDR )
            {
                m_fixedSuballocators[i].track( chunk.ptr, chunk.size );
                block = m_fixedSuballocators[i].alloc();
                m_freeSpace -= block.size;
            }
            return block;
        }
    }

    // If the allocation size is too big, use the heapSuballocator
    MemoryBlockDesc block = m_heapSuballocator.alloc( size, alignment );
    if( block.ptr != BAD_ADDR )
        m_freeSpace -= block.size;
    return block;
}

inline void BinnedSuballocator::free( const MemoryBlockDesc& block )
{
    if( block.ptr != BAD_ADDR )
        m_freeSpace += block.size;

    // First try to free from one of the fixed suballocators, then use heap suballocator
    for( unsigned int i = 0; i < m_itemSizes.size(); ++i )
    {
        if( block.size == m_itemSizes[i] )
        {
            m_fixedSuballocators[i].free( block );
            return;
        }
    }
    m_heapSuballocator.free( block );
}
}
