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

#include "PagingSystem.h"
#include "DemandLoaderImpl.h"
#include "Memory/PinnedMemoryManager.h"
#include "PagingSystemKernels.h"
#include "RequestProcessor.h"
#include "Util/CudaCallback.h"

#include <algorithm>

namespace demandLoading {

PagingSystem::PagingSystem( unsigned int         deviceIndex,
                            const Options&       options,
                            DeviceMemoryManager* deviceMemoryManager,
                            PinnedMemoryManager* pinnedMemoryManager,
                            RequestProcessor*    requestProcessor )
    : m_options( options )
    , m_deviceIndex( deviceIndex )
    , m_deviceMemoryManager( deviceMemoryManager )
    , m_pinnedMemoryManager( pinnedMemoryManager )
    , m_requestProcessor( requestProcessor )
{
    DEMAND_ASSERT( m_options.maxFilledPages >= m_options.maxRequestedPages );
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );

    // Make the initial pushMappings event (which will be recorded when pushMappings is called)
    m_pushMappingsEvent = std::make_shared<FutureEvent>();

    // Allocate pinned memory for page mappings and invalidated pages (see pushMappings).  Note that
    // it's not necessary to free m_pageMappingsContext in the destructor, since it's pool
    // allocated.  (Nor is it possible, because doing so requires a stream.)
    m_pageMappingsContext = m_pinnedMemoryManager->getPageMappingsContextPool()->allocate();
    m_pageMappingsContext->clear();
}

void PagingSystem::updateLruThreshold( unsigned int returnedStalePages, unsigned int requestedStalePages, unsigned int medianLruVal )
{
    // Don't change the value if no stale pages were requested
    if( requestedStalePages == 0 )
        return;

    // Heuristic to update the lruThreshold. The basic idea is to aggressively reduce the threshold
    // if not enough stale pages are returned, but only gradually increase the threshold if it is too low.
    if( returnedStalePages < requestedStalePages / 2 )
        m_lruThreshold -= std::min( m_lruThreshold - MIN_LRU_THRESHOLD, 4u );
    else if( returnedStalePages < requestedStalePages )
        m_lruThreshold -= std::min( m_lruThreshold - MIN_LRU_THRESHOLD, 2u );
    else if( medianLruVal > m_lruThreshold )
        m_lruThreshold++;
}

// This callback invokes processRequests() after the asynchronous copies in pullRequests have completed.
class ProcessRequestsCallback : public CudaCallback
{
  public:
    ProcessRequestsCallback( PagingSystem* pagingSystem, DeviceContext context, RequestContext* pinnedContext, CUstream stream, Ticket ticket )
        : m_pagingSystem( pagingSystem )
        , m_context( context )
        , m_pinnedContext( pinnedContext )
        , m_stream( stream )
        , m_ticket( ticket )
    {
    }

    void callback() override { m_pagingSystem->processRequests( m_context, m_pinnedContext, m_stream, m_ticket ); }

  private:
    PagingSystem*   m_pagingSystem;
    DeviceContext   m_context;
    RequestContext* m_pinnedContext;
    CUstream        m_stream;
    Ticket          m_ticket;
};

void PagingSystem::pullRequests( const DeviceContext& context, CUstream stream, unsigned int startPage, unsigned int endPage, Ticket ticket )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // The array lengths are accumulated across multiple device threads, so they must be initialized to zero.
    DEMAND_CUDA_CHECK( cudaMemsetAsync( context.arrayLengths.data, 0, context.arrayLengths.capacity * sizeof( unsigned int ), stream ) );

    DEMAND_ASSERT( startPage <= endPage );
    DEMAND_ASSERT( endPage < m_options.numPages );
    m_launchNum++;

    launchPullRequests( stream, context, m_launchNum, m_lruThreshold, startPage, endPage );

    // Get a RequestContext from the pinned memory manager, which will serve as the destination for
    // asynchronous copies of the requested pages, etc.
    RequestContext* pinnedContext = m_pinnedMemoryManager->getRequestContextPool()->allocate();

    // Copy the requested page list from this device.  The actual length is unknown, so we copy the entire capacity
    // and update the length below.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( pinnedContext->requestedPages, context.requestedPages.data,
                                        pinnedContext->maxRequestedPages * sizeof( unsigned int ), cudaMemcpyDeviceToHost, stream ) );

    // Get the stale pages from the device. This may be a subset of the actual stale pages.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( pinnedContext->stalePages, context.stalePages.data,
                                        pinnedContext->maxStalePages * sizeof( StalePage ), cudaMemcpyDeviceToHost, stream ) );

    // Get the sizes of the requested/stale page lists.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( pinnedContext->arrayLengths, context.arrayLengths.data,
                                        pinnedContext->numArrayLengths * sizeof( unsigned int ), cudaMemcpyDeviceToHost, stream ) );

    // Enqueue host function call to process the page requests once the kernel launch and copies have completed.
    CudaCallback::enqueue( stream, new ProcessRequestsCallback( this, context, pinnedContext, stream, ticket ) );
}

// Note: this method must not make any CUDA API calls, because it's invoked via cuLaunchHostFunc.
void PagingSystem::processRequests( const DeviceContext& context, RequestContext* requestContext, CUstream stream, Ticket ticket )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Return device context to pool.  The DeviceContext has been copied, but DeviceContextPool is designed to permit that.
    m_deviceMemoryManager->getDeviceContextPool()->free( const_cast<DeviceContext*>( &context ) );

    // Restore staged requests, and remove them from the request list (second chance algorithm)
    unsigned int numRequestedPages = requestContext->arrayLengths[PAGE_REQUESTS_LENGTH];
    unsigned int numStalePages     = requestContext->arrayLengths[STALE_PAGES_LENGTH];

    for( unsigned int i = 0; i < numRequestedPages; ++i )
    {
        if( restoreMapping( requestContext->requestedPages[i] ) )
        {
            requestContext->requestedPages[i] = requestContext->requestedPages[numRequestedPages - 1];
            --numRequestedPages;
            --i;
        }
    }
    requestContext->arrayLengths[PAGE_REQUESTS_LENGTH] = numRequestedPages;

    // Enqueue the requests for asynchronous processing.
    m_requestProcessor->addRequests( m_deviceIndex, stream, requestContext->requestedPages, numRequestedPages, ticket );

    // Sort and stage stale pages, and update the LRU threshold
    unsigned int medianLruVal = 0;
    if( numStalePages > 0 )
    {
        if( context.lruTable != nullptr )
        {
            std::sort( requestContext->stalePages, requestContext->stalePages + numStalePages,
                       []( StalePage a, StalePage b ) { return a.lruVal < b.lruVal; } );
            medianLruVal = requestContext->stalePages[numStalePages / 2].lruVal;
        }
        else
        {
            std::shuffle(requestContext->stalePages, requestContext->stalePages + numStalePages, m_rng);
        }

        if( m_evictionActive && getNumStagedPages() < m_options.maxStagedPages )
        {
            m_stagedPages.emplace_back( StagedPageList{m_pushMappingsEvent, std::deque<PageMapping>()} );
            stageStalePages( requestContext, m_stagedPages.back().mappings );
        }
    }
    updateLruThreshold( numStalePages, requestContext->maxStalePages, medianLruVal );

    // Return the RequestContext to its pool.
    m_pinnedMemoryManager->getRequestContextPool()->free( requestContext );
}

void PagingSystem::addMapping( unsigned int pageId, unsigned int lruVal, unsigned long long entry )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    addMappingBody( pageId, lruVal, entry );
}

bool PagingSystem::isResident( unsigned int pageId )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    const auto&                  p = m_pageTable.find( pageId );
    return ( p != m_pageTable.end() ) ? p->second.resident : false;
}

unsigned int PagingSystem::pushMappings( const DeviceContext& context, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    const unsigned int numFilledPages = m_pageMappingsContext->numFilledPages;
    if( numFilledPages > 0 )
    {
        DEMAND_CUDA_CHECK( cudaMemcpyAsync( context.filledPages.data, m_pageMappingsContext->filledPages,
                                            numFilledPages * sizeof( PageMapping ), cudaMemcpyHostToDevice, stream ) );
        launchPushMappings( stream, context, numFilledPages );
    }

    const unsigned int numInvalidatedPages = m_pageMappingsContext->numInvalidatedPages;
    if( numInvalidatedPages > 0 )
    {
        DEMAND_CUDA_CHECK( cudaMemcpyAsync( context.invalidatedPages.data, m_pageMappingsContext->invalidatedPages,
                                            numInvalidatedPages * sizeof( unsigned int ), cudaMemcpyHostToDevice, stream ) );
        launchInvalidatePages( stream, context, numInvalidatedPages );
    }

    // Zero out the reference bits
    unsigned int referenceBitsSizeInBytes = idivCeil( context.maxNumPages, 8 );
    DEMAND_CUDA_CHECK( cudaMemsetAsync( context.referenceBits, 0, referenceBitsSizeInBytes, stream ) );

    // Record the event in the stream. pushMappings will be complete when it returns cudaSuccess
    DEMAND_CUDA_CHECK( cudaEventRecord( m_pushMappingsEvent->event, stream ) );
    m_pushMappingsEvent->recorded = true;

    // Make a new event for the next time pushMappings is called
    m_pushMappingsEvent = std::make_shared<FutureEvent>();

    // Free the current PageMappingsContext (it's not reused until the preceding operations on the stream are done)
    // and allocate another one.  Note that we're careful to reserve two contexts per stream in the PinnedMemoryManager.
    m_pinnedMemoryManager->getPageMappingsContextPool()->free( m_pageMappingsContext, m_deviceIndex, stream );
    m_pageMappingsContext = m_pinnedMemoryManager->getPageMappingsContextPool()->allocate();
    m_pageMappingsContext->clear();

    return numFilledPages;
}

void PagingSystem::stageStalePages( RequestContext* requestContext, std::deque<PageMapping>& stagedMappings )
{
    // Mutex acquired in caller (processRequests)

    unsigned int numStalePages = requestContext->arrayLengths[STALE_PAGES_LENGTH];
    size_t       numStaged     = getNumStagedPages();

    // Count backwards to stage the oldest pages first
    for( int i = static_cast<int>( numStalePages - 1 ); i >= 0; --i )
    {
        StalePage sp = requestContext->stalePages[i];
        if( numStaged >= m_options.maxStagedPages || m_pageMappingsContext->numInvalidatedPages >= m_options.maxInvalidatedPages )
            break;

        const auto& p = m_pageTable.find( sp.pageId );
        if( p != m_pageTable.end() && p->second.resident == true && p->second.inStagedList == false )
        {
            // Stage the page
            stagedMappings.emplace_back( PageMapping{sp.pageId, sp.lruVal, p->second.entry} );
            p->second.resident     = false;
            p->second.staged       = true;
            p->second.inStagedList = true;

            // Schedule the page mapping to be invalidated on the device
            m_pageMappingsContext->invalidatedPages[m_pageMappingsContext->numInvalidatedPages++] = sp.pageId;
            numStaged++;
        }
    }
}

bool PagingSystem::freeStagedPage( PageMapping* m )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    while( !m_stagedPages.empty() && ( m_stagedPages[0].event->query() == cudaSuccess ) )
    {
        // Advance to the next list if the beginning list is empty
        if( m_stagedPages[0].mappings.empty() )
        {
            m_stagedPages.pop_front();
            continue;
        }

        // Pop the next staged page off the list
        *m = m_stagedPages[0].mappings.front();
        m_stagedPages[0].mappings.pop_front();

        const auto& p = m_pageTable.find( m->id );
        DEMAND_ASSERT( p != m_pageTable.end() );
        p->second.inStagedList = false;

        // If the page is still staged, return. Otherwise, go around and look for another one
        if( p->second.staged == true )
        {
            m_pageTable.erase( p );
            return true;
        }
    }
    return false;
}

void PagingSystem::addMappingBody( unsigned int pageId, unsigned int lruVal, unsigned long long entry )
{
    // Mutex acquired in caller

    DEMAND_ASSERT_MSG( m_pageMappingsContext->numFilledPages <= m_pageMappingsContext->maxFilledPages,
                       "Maximum number of filled pages exceeded (Options::maxFilledPages)" );
    const auto& p = m_pageTable.find( pageId );
    DEMAND_ASSERT( p == m_pageTable.end() || p->second.resident == false );
    DEMAND_ASSERT( pageId < m_options.numPages );

    m_pageMappingsContext->filledPages[m_pageMappingsContext->numFilledPages++] = PageMapping{pageId, lruVal, entry};
    m_pageTable[pageId] = HostPageTableEntry{entry, true, false, false};
}

bool PagingSystem::restoreMapping( unsigned int pageId )
{
    // Mutex acquired in caller (processRequests).

    const auto& p = m_pageTable.find( pageId );
    if( p != m_pageTable.end() && p->second.staged && !p->second.resident
        && m_pageMappingsContext->numFilledPages < m_pageMappingsContext->maxFilledPages )
    {
        p->second.staged = false;
        addMappingBody( pageId, 0, p->second.entry );
        return true;
    }

    return false;
}

size_t PagingSystem::getNumStagedPages()
{
    size_t numPages = 0;
    for( StagedPageList s : m_stagedPages )
        numPages += s.mappings.size();
    return numPages;
}

void PagingSystem::flushMappings()
{
    std::unique_lock<std::mutex> lock( m_mutex );
    m_pageMappingsContext->numFilledPages = 0;
}

}  // namespace demandLoading
