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

#include "Util/Math.h"

#include <cstddef>
#include <mutex>
#include <vector>

namespace demandLoading {

/// ItemPool is a thread-safe, unlimited capacity pool of fixed-sized items.  In order to reduce
/// allocation frequency, memory is allocated in arenas of approximately 2 MB.  This is important
/// when allocating device memory, because doing so incurs a substantial synchronization cost.
/// ItemPool is intended for use with "plain old data" (POD) types.  Initializing and destroying
/// elements is the client's responsibility.
template <typename Item, typename Allocator>
class ItemPool
{
  public:
    /// Construct item pool using the given allocator (which is copied).
    explicit ItemPool( const Allocator& allocator )
        : m_allocator( allocator )
        , m_itemSize( align( sizeof( Item ), alignof( Item ) ) )
        , m_itemsPerArena( 2 * 1024 * 1024 / m_itemSize )
    {
    }

    /// Destroy the memory pool, reclaiming its resources.
    ~ItemPool()
    {
        for( Item* arena : m_arenas )
        {
            m_allocator.free( arena );
        }
    }

    /// Allocate memory for an item from the pool.  The item is uninitialized.
    Item* allocate()
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        // Return the most recently freed item, if any.
        if( !m_freedItems.empty() )
        {
            Item* item = m_freedItems.back();
            m_freedItems.pop_back();
            ++m_size;
            return item;
        }

        // Create a new arena if necessary.  This reduces allocation frequency compared to
        // allocating items individually.
        if( m_arenas.empty() || m_nextItem >= m_itemsPerArena )
        {
            size_t paddedItemSize = align( sizeof( Item ), alignof( Item ) );
            Item*  arena = reinterpret_cast<Item*>( m_allocator.allocate( m_itemsPerArena * paddedItemSize ) );
            m_arenas.push_back( arena );
            m_nextItem = 0;
        }

        Item* item = m_arenas.back() + m_nextItem++;
        ++m_size;
        return item;
    }

    /// Return the given item to the pool.
    void free( Item* item )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_freedItems.push_back( item );
        --m_size;
    }

    /// Get the number of items in use.
    size_t size() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_size;
    }

    /// Get the pool capacity (number of items).
    size_t capacity() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_arenas.size() * m_itemsPerArena;
    }


    /// Not copyable
    ItemPool( const ItemPool& ) = delete;

    /// Not assignable
    ItemPool& operator=( const ItemPool& ) = delete;

  private:
    mutable std::mutex m_mutex;
    Allocator          m_allocator;
    std::vector<Item*> m_arenas;
    std::vector<Item*> m_freedItems;
    size_t             m_itemSize      = 0;
    size_t             m_itemsPerArena = 0;
    size_t             m_nextItem      = 0;  // offset of next available item in current arena
    size_t             m_size          = 0;
};

}  // namespace demandLoading
