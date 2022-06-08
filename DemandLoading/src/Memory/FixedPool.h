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

#include "Util/Exception.h"

#include <mutex>
#include <vector>

namespace demandLoading {

/// FixedPool is a thread-safe, fixed-capacity pool of items.  It is agnostic to the allocation
/// mechanism employed for the backing storage.
template <typename Item>
class FixedPool
{
  public:
    /// The pool is default constructed, then initialized.  This simplifies constructing the backing
    /// storage in a derived class prior to initializing this base class.
    FixedPool() {}

    /// Initialize the item pool with the given items.
    void init( Item* items, size_t numItems )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT( m_items == nullptr );
        m_items = items;
        m_capacity = numItems;
    }

    /// Allocate an item from the pool.  Returns nullptr if no item is available.
    Item* allocate()
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT( m_items != nullptr );

        // Return the most recently freed item, if any.
        if( !m_freedItems.empty() )
        {
            Item* item = m_freedItems.back();
            m_freedItems.pop_back();
            return item;
        }

        // Return the next available item, if any.
        if( m_nextItem < m_capacity )
        {
            return &m_items[m_nextItem++];
        }
        return nullptr;
    }

    /// Return the given item to the pool.
    void free( Item* item )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT( m_items != nullptr );
        DEMAND_ASSERT( item != nullptr );
        DEMAND_ASSERT_MSG( m_items <= item && item < m_items + m_capacity, "Invalid pointer in FixedPool::free()" );

        m_freedItems.push_back( item );
    }

    /// Get the number of items in use.
    size_t size() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        size_t allocated = m_nextItem;
        size_t freed     = m_freedItems.size();
        DEMAND_ASSERT( freed <= allocated );
        return allocated - freed;
    }

    /// Get the total number of items in the pool.
    size_t capacity() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_capacity;
    }

    /// Not copyable.
    FixedPool( const FixedPool& ) = delete;

    /// Not assignable.
    FixedPool& operator=( const FixedPool& ) = delete;

  private:
    mutable std::mutex m_mutex;
    Item*              m_items    = nullptr;
    size_t             m_capacity = 0;
    size_t             m_nextItem = 0;
    std::vector<Item*> m_freedItems;
};

}  // namespace demandLoading
