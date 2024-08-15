// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/Ticket.h>

#include <cuda.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

namespace demandLoading {

/// A page request contains a page id, which is a index into the page table.  It also holds a shared
/// pointer to a Ticket, which must be notified when the request has been filled.
struct PageRequest
{
    unsigned int pageId{};
    Ticket       ticket;

    // A constructor is necessary for emplace_back.
    PageRequest( unsigned int pageId_, Ticket ticket_ )
        : pageId( pageId_ )
        , ticket( ticket_ )
    {
    }

    // Default constructor
    PageRequest() = default;
};

class RequestQueue
{
  public:
    /// Construct request queue.
    RequestQueue( unsigned int maxQueueSize )
        : m_maxQueueSize( maxQueueSize )
    {
    }

    /// Pop a request, waiting if necessary until the queue is non-empty or shut down.  Returns
    /// false if the queue was shut down.
    bool popOrWait( PageRequest* request );

    /// Push a batch of page requests.  Notifies any threads waiting in popOrWait().  Updates the
    /// given Ticket with the number of requests, and retains it for notifications as requests are
    /// filled.
    void push( const unsigned int* pageIds, unsigned int numPageIds, Ticket ticket );

    /// Shut down the queue, signalling any waiting threads to exit.  Clients must call shutDown()
    /// and join with any waiting threads before invoking the RequestQueue destructor.
    void shutDown();

    /// Not copyable.
    RequestQueue( const RequestQueue& ) = delete;

    /// Not assignable.
    RequestQueue& operator=( const RequestQueue& ) = delete;

  private:
    std::deque<PageRequest> m_requests;
    std::mutex              m_mutex;
    std::condition_variable m_requestAvailable;
    unsigned int            m_maxQueueSize;
    bool                    m_isShutDown = false;
};

}  // namespace demandLoading
