// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <stdio.h>
#include <vector>

#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

namespace otk {

// FixedSuballocator tracks fixed item sizes, which allows alloc and free to be done in
// constant time.
//
class FixedSuballocator
{
  public:
    FixedSuballocator( uint64_t itemSize = 1, uint64_t alignment = 1 )
        : m_itemSize( alignVal( itemSize, std::max( alignment, static_cast<uint64_t>( 1ULL ) ) ) )
        , m_alignment( std::max( alignment, static_cast<uint64_t>( 1ULL ) ) )
    {
    }
    ~FixedSuballocator() = default;

    /// Tell the pool to track a memory segment.
    /// This can be called multiple times to track multiple segments.
    void track( uint64_t ptr, uint64_t size );

    /// Allocate a block from tracked memory. Returns the address to the allocated block.
    /// On failure, returns BAD_ADDR
    MemoryBlockDesc alloc( uint64_t size = 0, uint64_t alignment = 0 );

    /// Allocate an item, and just return a pointer to it
    uint64_t allocItem() { return alloc().ptr; };

    /// Free a block. The size must be correct to ensure correctness.
    void free( const MemoryBlockDesc& memBlock )
    {
        m_freeSpace += memBlock.size;
        m_freeBlocks.push_back( memBlock );
    }

    /// Free an item from just its pointer
    void freeItem( uint64_t ptr ) { free( MemoryBlockDesc{ptr, m_itemSize, 0} ); }

    /// Untrack memory that is currently tracked by the suballocator.
    void untrack( uint64_t ptr, uint64_t size );

    /// Return the size of items in the pool
    uint64_t itemSize() const { return m_itemSize; }

    /// Returns the alignment of items in the pool.
    uint64_t alignment() const { return m_alignment; }

    /// Return the total amount of free space
    uint64_t freeSpace() const { return m_freeSpace; }

    /// Return the total memory tracked by pool
    uint64_t trackedSize() const { return m_trackedSize; }

  protected:
    uint64_t m_trackedSize = 0;
    uint64_t m_freeSpace   = 0;
    uint64_t m_itemSize    = 0;
    uint64_t m_alignment   = 0;

    std::vector<MemoryBlockDesc> m_freeBlocks;
};

inline void FixedSuballocator::track( uint64_t ptr, uint64_t size )
{
    // Align tracked block with item size
    uint64_t p = alignVal( ptr, m_alignment );
    uint64_t s = size - ( p - ptr );
    s -= s % m_itemSize;

    m_trackedSize += s;
    free( MemoryBlockDesc{p, s, 0} );
}

inline MemoryBlockDesc FixedSuballocator::alloc( uint64_t /*size*/, uint64_t /*alignment*/ )
{
    if( m_freeBlocks.empty() )
        return MemoryBlockDesc{BAD_ADDR, 0, 0};

    MemoryBlockDesc block{m_freeBlocks.back().ptr, m_itemSize, 0};
    if( m_freeBlocks.back().size == m_itemSize )
    {
        m_freeBlocks.pop_back();
    }
    else
    {
        m_freeBlocks.back().ptr += m_itemSize;
        m_freeBlocks.back().size -= m_itemSize;
    }
    m_freeSpace -= m_itemSize;
    return block;
}

inline void FixedSuballocator::untrack( uint64_t ptr, uint64_t size )
{
    // Get rid of items in the untracked region
    for( int i = (int)m_freeBlocks.size() - 1; i >= 0; --i )
    {
        const MemoryBlockDesc& b = m_freeBlocks[i];
        if( ( ptr >= b.ptr && ptr < b.ptr + b.size ) || ( b.ptr >= ptr && b.ptr < ptr + size ) )
        {
            m_freeBlocks[i] = m_freeBlocks.back();
            m_freeBlocks.pop_back();
            m_freeSpace -= m_itemSize;
        }
    }

    // Reduce the tracked size
    m_trackedSize = ( size <= m_trackedSize ) ? m_trackedSize - size : 0ULL;
}

}  // namespace optix
