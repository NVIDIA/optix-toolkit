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

#include "ThreadPoolRequestProcessor.h"

#include "DemandLoaderImpl.h"
#include "RequestHandler.h"
#include "TicketImpl.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>

namespace demandLoading {

ThreadPoolRequestProcessor::ThreadPoolRequestProcessor( std::shared_ptr<PageTableManager> pageTableManager, const Options& options )
    : m_pageTableManager( std::move( pageTableManager ) )
    , m_options( options )
{
    m_requests.reset( new RequestQueue( options.maxRequestQueueSize ) );
}

void ThreadPoolRequestProcessor::start()
{
    if( m_started )
        return;

    m_requests.reset( new RequestQueue( m_options.maxRequestQueueSize ) );
    unsigned int maxThreads = m_options.maxThreads;
    if( maxThreads == 0 )
        maxThreads = std::thread::hardware_concurrency();
    m_threads.reserve( maxThreads );
    for( unsigned int i = 0; i < maxThreads; ++i )
    {
        m_threads.emplace_back( &ThreadPoolRequestProcessor::worker, this );
    }
    m_started = true;
}

void ThreadPoolRequestProcessor::stop()
{
    std::unique_lock<std::mutex> lock( m_ticketsMutex );

    if( !m_started )
        return;

    // Any threads that are waiting in RequestQueue::popOrWait will be notified when the queue is
    // shut down.
    m_requests->shutDown();
    for( std::thread& thread : m_threads )
    {
        thread.join();
    }
    m_requests.reset();
    m_threads.clear();
    m_started = false;
}

void ThreadPoolRequestProcessor::addRequests( CUstream /*stream*/, unsigned int id, const unsigned int* pageIds, unsigned int numPageIds )
{
    std::unique_lock<std::mutex> lock( m_ticketsMutex );
    start();
    
    auto it = m_tickets.find( id );
    OTK_ASSERT( it != m_tickets.end() );
    Ticket ticket = it->second;
    // We won't issue this id again, so we can discard it from the map.
    m_tickets.erase( it );

    // Filter the batch of requests, and add it to the main request list with the ticket to track their progress
    if( numPageIds > 0 && m_requestFilter )
    {
        std::vector<unsigned int> filteredRequests = m_requestFilter->filter( pageIds, numPageIds );
        m_requests->push( &filteredRequests[0], static_cast<unsigned int>( filteredRequests.size() ), ticket );
    }
    else
    {
        m_requests->push( pageIds, numPageIds, ticket );
    }
}

void ThreadPoolRequestProcessor::setTicket( unsigned int id, Ticket ticket )
{
    std::unique_lock<std::mutex> lock( m_ticketsMutex );
    OTK_ASSERT( m_tickets.find( id ) == m_tickets.end() );
    m_tickets[id] = ticket;
}

void ThreadPoolRequestProcessor::worker()
{
    try
    {
        PageRequest request;
        while( true )
        {
            // Pop a request from the queue, waiting if necessary until the queue is non-empty or shut down.
            if( !m_requests->popOrWait( &request ) )
                return;  // Exit thread when queue is shut down.

            // Ask the PageTableManager for the request handler associated with the range of pages in
            // which the request occurred.
            RequestHandler* handler = m_pageTableManager->getRequestHandler( request.pageId );
            OTK_ASSERT_MSG( handler != nullptr, "Invalid page requested (no associated handler)" );

            // Use the CUDA context associated with the stream in the ticket.
            std::shared_ptr<TicketImpl>& ticket = TicketImpl::getImpl( request.ticket );
            CUcontext                    context;
            OTK_ERROR_CHECK( cuStreamGetCtx( ticket->getStream(), &context ) );
            OTK_ERROR_CHECK( cuCtxSetCurrent( context ) );

            // Process the request.  Page table updates are accumulated in the PagingSystem.
            handler->fillRequest( ticket->getStream(), request.pageId );

            // Notify the associated Ticket that the request has been filled.
            ticket->notify();
            ticket.reset();
        }
    }
    catch( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
#ifndef NDEBUG
        std::terminate();
#endif        
    }
}

} // namespace demandLoading
