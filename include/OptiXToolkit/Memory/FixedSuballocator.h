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
    void freeItem( uint64_t ptr ) { free( MemoryBlockDesc{ptr, m_itemSize} ); }

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
    free( MemoryBlockDesc{p, s} );
}

inline MemoryBlockDesc FixedSuballocator::alloc( uint64_t size, uint64_t alignment )
{
    if( m_freeBlocks.empty() )
        return MemoryBlockDesc{BAD_ADDR};

    MemoryBlockDesc block{m_freeBlocks.back().ptr, m_itemSize};
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

}  // namespace optix
