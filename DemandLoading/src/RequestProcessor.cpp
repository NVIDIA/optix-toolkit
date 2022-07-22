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

#include "RequestProcessor.h"
#include "DemandLoaderImpl.h"
#include "RequestHandler.h"
#include "TicketImpl.h"

namespace demandLoading {

void RequestProcessor::start( unsigned int maxThreads )
{
    if (maxThreads == 0)
        maxThreads = std::thread::hardware_concurrency();

    m_threads.reserve( maxThreads );
    for( unsigned int i = 0; i < maxThreads; ++i )
    {
        m_threads.emplace_back( &RequestProcessor::worker, this );
    }
}

void RequestProcessor::stop()
{
    // Any threads that are waiting in RequestQueue::popOrWait will be notified when the queue is
    // shut down.
    m_requests.shutDown();
    for( std::thread& thread : m_threads )
    {
        thread.join();
    }
}

void RequestProcessor::addRequests( unsigned int deviceIndex, CUstream stream, const unsigned int* pageIds, unsigned int numPageIds, Ticket ticket )
{
    m_requests.push( deviceIndex, stream, pageIds, numPageIds, ticket );

    // If recording is enabled, write the requests to the trace file.
    if( m_traceFile && numPageIds > 0 )
    {
        m_traceFile->recordRequests( deviceIndex, stream, pageIds, numPageIds );
    }
}

void RequestProcessor::worker()
{
    try
    {
        PageRequest request;
        while( true )
        {
            // Pop a request from the queue, waiting if necessary until the queue is non-empty or shut down.
            if( !m_requests.popOrWait( &request ) )
                return;  // Exit thread when queue is shut down.

            // Ask the PageTableManager for the request handler associated with the range of pages in
            // which the request occurred.
            RequestHandler* handler = m_pageTableManager->getRequestHandler( request.pageId );
            DEMAND_ASSERT_MSG( handler != nullptr, "Invalid page requested (no associated handler)" );

            // Process the request.  Page table updates are accumulated in the PagingSystem.
            std::shared_ptr<TicketImpl>& ticket = TicketImpl::getImpl( request.ticket );
            handler->fillRequest( ticket->getDeviceIndex(), ticket->getStream(), request.pageId );

            // Notify the associated Ticket that the request has been filled.
            ticket->notify();
            ticket.reset();
        }
    }
    catch( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::terminate();
    }
}

} // namespace demandLoading
