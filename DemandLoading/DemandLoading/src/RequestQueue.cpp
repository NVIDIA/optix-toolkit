// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "RequestQueue.h"
#include "TicketImpl.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <algorithm>

namespace demandLoading {

void RequestQueue::shutDown()
{
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_isShutDown = true;
    }
    m_requestAvailable.notify_all();
}

bool RequestQueue::popOrWait( PageRequest* requestPtr )
{
    // Wait until the queue is non-empty or destroyed.
    std::unique_lock<std::mutex> lock( m_mutex );
    m_requestAvailable.wait( lock, [this] { return !m_requests.empty() || m_isShutDown; } );

    if( m_isShutDown )
        return false;

    *requestPtr = std::move( m_requests.front() );
    m_requests.pop_front();

    return true;
}

void RequestQueue::push( const unsigned int* pageIds, unsigned int numPageIds, Ticket ticket )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Don't push requests if the queue is shut down.
    if( m_isShutDown )
        numPageIds = 0;
    
    // Don't overfill the queue
    if( m_requests.size() >= m_maxQueueSize )
        numPageIds = 0;
    else if( numPageIds + m_requests.size() > m_maxQueueSize )
        numPageIds = static_cast<unsigned int>( m_maxQueueSize - m_requests.size() );

    // Update the ticket, now that the number of tasks is known.
    TicketImpl::getImpl( ticket )->update( numPageIds );

    if( numPageIds == 0 )
        return;

    for( unsigned int i = 0; i < numPageIds; ++i )
    {
        m_requests.emplace_back( pageIds[i], ticket );
    }

    // Notify any threads in popOrWait().
    m_requestAvailable.notify_all();
}

}  // namespace demandLoading
