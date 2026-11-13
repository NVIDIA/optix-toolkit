// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "ThreadPoolRequestProcessor.h"

#include "DemandLoaderImpl.h"
#include "RequestHandler.h"
#include "TicketImpl.h"

#include <OptiXToolkit/Error/ErrorCheck.h>
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
    // Swap threads and requests with local variables
    std::vector<std::thread>      threads;
    std::unique_ptr<RequestQueue> requests;
    {
        std::unique_lock<std::mutex> lock( m_ticketsMutex );
        if( !m_started )
            return;
        threads.swap( m_threads );
        requests.swap( m_requests );
        m_started = false;
    }

    // Shut down the request queue and join the threads.
    requests->shutDown();
    for( std::thread& thread : threads )
        thread.join();
    // Flush the request queue to ensure all tickets are processed.
    requests->flush();
}

void ThreadPoolRequestProcessor::flush()
{
    std::unique_lock<std::mutex> lock( m_ticketsMutex );
    if( m_requests )
        m_requests->flush();
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
            RequestQueue* requests = m_requests.get();
            if( !requests || !requests->popOrWait( &request ) )
                return;  // Exit thread when queue is shut down.

            std::shared_ptr<TicketImpl>& ticket = TicketImpl::getImpl( request.ticket );

            try
            {
                // Ask the PageTableManager for the request handler for the page.
                RequestHandler* handler = m_pageTableManager->getRequestHandler( request.pageId );
                OTK_ASSERT_MSG( handler != nullptr, "Invalid page requested (no associated handler)" );

                // Use the CUDA context associated with the stream in the ticket.
                CUcontext context;
                OTK_ERROR_CHECK( cuStreamGetCtx( ticket->getStream(), &context ) );
                OTK_ERROR_CHECK( cuCtxSetCurrent( context ) );

                // Process the request.  Page table updates are accumulated in the PagingSystem.
                handler->fillRequest( ticket->getStream(), request.pageId );
            }
            catch( const std::exception& e )
            {
                std::cerr << "Error: " << e.what() << std::endl;
            }

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
