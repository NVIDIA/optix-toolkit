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

#include <DemandLoading/Ticket.h>

#include <cuda.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

namespace demandLoading {

/// A page request contains a page id, which is a index into the page table and the index of the
/// requesting device.  It also holds a shared pointer to a Ticket, which must be notified when
/// the request has been filled.
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
    PageRequest() {}
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

    /// Push a batch of page requests from the specified device.  Notifies any threads waiting in
    /// popOrWait().  Updates the given Ticket with the number of requests, and retains it for
    /// notifications as requests are filled.
    void push( unsigned int deviceIndex, CUstream stream, const unsigned int* pageIds, unsigned int numPageIds, Ticket ticket );

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
