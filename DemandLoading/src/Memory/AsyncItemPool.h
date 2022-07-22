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

#include "Memory/EventPool.h"
#include "Util/Exception.h"
#include "Util/Math.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

namespace demandLoading {

/// AsyncItemPool is a thread-safe, fixed-capacity pool of items that are used asynchronously, for
/// example as the source or destination of memcpyAsync.  The free() method takes a stream argument,
/// and the freed item is not reused until all operations on that stream have completed.  The
/// allocate() method returns an item immediately if one is available.  Otherwise it waits for the
/// oldest freed item to become available.  The shutDown() method should be called before the pool
/// is destroyed.  It wakes any threads that are waiting in allocate() and it waits on all recently
/// freed items, to ensure that any asynchronous operations using the pinned memory have completed.
template <typename Item>
class AsyncItemPool
{
  public:
    /// The pool is default constructed, then initialized.  This simplifies constructing the backing
    /// storage in a derived class prior to initializing this base class.
    AsyncItemPool() {}

    /// Initialize the item pool with the given items.
    void init( Item* items, size_t numItems )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT( m_items == nullptr );
        m_items    = items;
        m_capacity = numItems;
    }

    /// Shut down the pool, waking any threads that are waiting in allocate() and waiting on all
    /// recently freed items, to ensure that any asynchronous operations using the pinned memory
    /// have completed.
    void shutDown()
    {
        {  // We acquire the mutex in a nested scope so it's released before we notify the condition variable.
            std::unique_lock<std::mutex> lock( m_mutex );
            m_isShutDown = true;
        }
        m_freedItemAvailable.notify_all();
    }

    /// Allocate an item from the pool, waiting for one to be freed if necessary.  Returns a null
    /// pointer if the pool has been shut down.
    Item* allocate()
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT( m_items != nullptr );

        // Return the next available item, if any.
        if( m_nextItem < m_capacity )
        {
            return &m_items[m_nextItem++];
        }

        if( oldestFreedItemAvailable() )
        {
            FreedItem freedItem = m_freedItems.front();
            m_freedItems.pop_front();
            getEventPool( freedItem.deviceIndex )->free( freedItem.event );
            return freedItem.item;
        }

        // Wait for an item to be freed (until the pool is shut down).
        m_freedItemAvailable.wait( lock, [this] { return !m_freedItems.empty() || m_isShutDown; } );
        if( m_isShutDown )
            return nullptr;

        // Wait for the oldest freed item to be available.
        FreedItem freedItem = m_freedItems.front();
        m_freedItems.pop_front();
        DEMAND_CUDA_CHECK( cudaSetDevice( freedItem.deviceIndex ) );
        DEMAND_CUDA_CHECK( cuEventSynchronize( freedItem.event ) );

        // Release the event and return the associated item.
        getEventPool( freedItem.deviceIndex )->free( freedItem.event );
        return freedItem.item;
    }

    /// Return the given item to the pool.  An event is recorded on the given stream, and the freed
    /// item is not reused until all preceding operations on the stream have finished (e.g.  the
    /// pinned memory might be the source or target of an asynchronous copy).
    void free( Item* item, unsigned int deviceIndex, CUstream stream )
    {
        {  // We acquire the mutex in a nested scope so it's released before we notify the condition variable.
            std::unique_lock<std::mutex> lock( m_mutex );
            DEMAND_ASSERT( m_items != nullptr );
            DEMAND_ASSERT( item != nullptr );
            DEMAND_ASSERT_MSG( m_items <= item && item < m_items + m_capacity,
                               "Invalid pointer in AsyncItemPool::free()" );

            // Take an event from the pool and record it on the given stream.
            CUevent event = getEventPool( deviceIndex )->allocate();
            DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
            DEMAND_CUDA_CHECK( cuEventRecord( event, stream ) );

            // Push the item and its associated event on the freed items queue and notify any threads
            // waiting in allocate().
            m_freedItems.push_back( FreedItem{item, event, deviceIndex} );
        }
        m_freedItemAvailable.notify_all();
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
    AsyncItemPool( const AsyncItemPool& ) = delete;

    /// Not assignable.
    AsyncItemPool& operator=( const AsyncItemPool& ) = delete;

  private:
    struct FreedItem
    {
        Item*        item;
        CUevent      event;
        unsigned int deviceIndex;
    };

    mutable std::mutex m_mutex;
    Item*              m_items    = nullptr;
    size_t             m_capacity = 0;
    size_t             m_nextItem = 0;

    std::vector<std::unique_ptr<EventPool>> m_eventPools;  // one per device
    std::deque<FreedItem>                   m_freedItems;
    std::condition_variable                 m_freedItemAvailable;
    bool                                    m_isShutDown = false;

    EventPool* getEventPool( unsigned int deviceIndex )
    {
        if( deviceIndex >= m_eventPools.size() )
        {
            m_eventPools.resize( deviceIndex + 1 );
            m_eventPools[deviceIndex].reset( new EventPool( deviceIndex, m_capacity ) );
        }
        else if( !m_eventPools[deviceIndex] )
        {
            m_eventPools[deviceIndex].reset( new EventPool( deviceIndex, m_capacity ) );
        }
        return m_eventPools[deviceIndex].get();
    }

    bool oldestFreedItemAvailable() 
    {
        return !m_freedItems.empty() && ( cudaEventQuery( m_freedItems.front().event ) == cudaSuccess );
    }
};

}  // namespace demandLoading
