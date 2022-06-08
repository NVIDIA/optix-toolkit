//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "Memory/AsyncItemPool.h"
#include "Memory/BulkMemory.h"

namespace demandLoading {

/// BulkPinnedItemPool is a thread-safe, fixed-capacity pool of items that are allocated in pinned
/// memory.  The base class, AsyncItemPool, provides a free() method takes a stream argument, and
/// the freed item is not reused until all operations on that stream have completed.  This allows
/// pool-allocated items to be use as the source or destination of an asynchronous memcpy.
/// BulkPinnedItemPool is similar to PinnedItemPool, but it allocates each item from a chunk of
/// BulkPinnedMemory.  This allows each item to contain pointers into various offsets of the item's
/// allocation.  For example, \see PageMappingsContext, which uses BulkPinnedMemory to allocate
/// separate arrays of page mappings and invalidated pages.
template <typename Item, typename ItemConfig>
class BulkPinnedItemPool : public AsyncItemPool<Item>
{
  public:
    /// Construct a pool of items in pinned memory with the specified capacity.  Each item is
    /// constructed with a BulkPinnedMemory object along with the given configuration parameter.
    BulkPinnedItemPool( size_t capacity, const ItemConfig& config )
    {
        // Reserve BulkMemory for each Items.
        for( size_t i = 0; i < capacity; ++i )
        {
            Item::reserve( &m_memory, config );
        }

        // Construct each item, which allocates a chunk of BulkMemory.
        m_items.resize( capacity );
        for( Item& item : m_items )
        {
            item.allocate( &m_memory, config );
        }

        // Provide array of items to the base class.
        AsyncItemPool<Item>::init( m_items.data(), m_items.size() );
    }

    /// Get the total amount of pinned memory allocated.
    size_t getTotalPinnedMemory() const { return m_memory.capacity(); }
    
  private:
    BulkPinnedMemory  m_memory;
    std::vector<Item> m_items;
};

}  // namespace demandLoading
