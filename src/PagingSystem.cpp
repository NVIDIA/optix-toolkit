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
#include "Memory/DeviceMemoryManager.h"
#include "PageMappingsContext.h"
#include "PagingSystemKernels.h"
#include "RequestContext.h"
#include "Util/CudaCallback.h"
#include "Util/Math.h"

#include <OptiXToolkit/DemandLoading/RequestProcessor.h>

#include <algorithm>
#include <set>

namespace demandLoading {

PagingSystem::PagingSystem( unsigned int         deviceIndex,
                            const Options&       options,
                            DeviceMemoryManager* deviceMemoryManager,
                            MemoryPool<PinnedAllocator, RingSuballocator>* pinnedMemoryPool,
                            RequestProcessor*    requestProcessor )
    : m_options( options )
    , m_deviceIndex( deviceIndex )
    , m_deviceMemoryManager( deviceMemoryManager )
    , m_requestProcessor( requestProcessor )
    , m_pinnedMemoryPool( pinnedMemoryPool )
{
    DEMAND_ASSERT( m_options.maxFilledPages >= m_options.maxRequestedPages );
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
    DEMAND_CUDA_CHECK( cudaFree( 0 ) );

    // Make the initial pushMappings event (which will be recorded when pushMappings is called)
    m_pushMappingsEvent = std::make_shared<FutureEvent>();

    // Allocate a pageMappingsContext in pinned memory (see pushMappings). It's not necessary
    // to free it in the destructor, since it's pool allocated.
    m_pageMappingsContextBlock =
        m_pinnedMemoryPool->alloc( PageMappingsContext::getAllocationSize( m_options ), alignof( PageMappingsContext ) );
    m_pageMappingsContext = reinterpret_cast<PageMappingsContext*>( m_pageMappingsContextBlock.ptr );
    m_pageMappingsContext->init( m_options );
}

PagingSystem::~PagingSystem()
{
    for( RequestContext* requestContext : m_pinnedRequestContextPool )
        DEMAND_CUDA_CHECK( cudaFreeHost( requestContext ) );
    m_pinnedRequestContextPool.clear();
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
    ProcessRequestsCallback( PagingSystem* pagingSystem, DeviceContext context, RequestContext* pinnedRequestContext, CUstream stream, unsigned int id )
        : m_pagingSystem( pagingSystem )
        , m_context( context )
        , m_pinnedRequestContext( pinnedRequestContext )
        , m_stream( stream )
        , m_id( id )
    {
    }

    void callback() override { m_pagingSystem->processRequests( m_context, m_pinnedRequestContext, m_stream, m_id ); }

  private:
    PagingSystem*   m_pagingSystem;
    DeviceContext   m_context;
    RequestContext* m_pinnedRequestContext;
    CUstream        m_stream;
    unsigned int    m_id;
};

void PagingSystem::pullRequests( const DeviceContext& context, CUstream stream, unsigned int id, unsigned int startPage, unsigned int endPage )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // The array lengths are accumulated across multiple device threads, so they must be initialized to zero.
    DEMAND_CUDA_CHECK( cuMemsetD8Async( reinterpret_cast<CUdeviceptr>( context.arrayLengths.data ), 0,
                                        context.arrayLengths.capacity * sizeof( unsigned int ), stream ) );

    DEMAND_ASSERT( startPage <= endPage );
    DEMAND_ASSERT( endPage < m_options.numPages );
    m_launchNum++;

    launchPullRequests( stream, context, m_launchNum, m_lruThreshold, startPage, endPage );

    // Get a RequestContext from the pinned memory pool, which will serve as the destination for async copies.
    RequestContext* pinnedRequestContext = nullptr;
    if( !m_pinnedRequestContextPool.empty() )
    {
        pinnedRequestContext = m_pinnedRequestContextPool.back();
        m_pinnedRequestContextPool.pop_back();
    }
    else 
    {
       DEMAND_CUDA_CHECK( cudaMallocHost(&pinnedRequestContext, RequestContext::getAllocationSize( m_options ) ) );
       pinnedRequestContext->init( m_options );
    }

    // Copy the requested page list from this device.  The actual length is unknown, so we copy the entire capacity
    // and update the length below.
    DEMAND_CUDA_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( pinnedRequestContext->requestedPages ),
                                      reinterpret_cast<CUdeviceptr>( context.requestedPages.data ),
                                      pinnedRequestContext->maxRequestedPages * sizeof( unsigned int ), stream ) );

    // Get the stale pages from the device. This may be a subset of the actual stale pages.
    DEMAND_CUDA_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( pinnedRequestContext->stalePages ),
                                      reinterpret_cast<CUdeviceptr>( context.stalePages.data ),
                                      pinnedRequestContext->maxStalePages * sizeof( StalePage ), stream ) );

    // Get the sizes of the requested/stale page lists.
    DEMAND_CUDA_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( pinnedRequestContext->arrayLengths ),
                                      reinterpret_cast<CUdeviceptr>( context.arrayLengths.data ),
                                      pinnedRequestContext->numArrayLengths * sizeof( unsigned int ), stream ) );

    // Enqueue host function call to process the page requests once the kernel launch and copies have completed.
    CudaCallback::enqueue( stream, new ProcessRequestsCallback( this, context, pinnedRequestContext, stream, id ) );
}

// Note: this method must not make any CUDA API calls, because it's invoked via cuLaunchHostFunc.
void PagingSystem::processRequests( const DeviceContext& context, RequestContext* pinnedRequestContext, CUstream stream, unsigned int id )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Return device context to pool.  The DeviceContext has been copied, but DeviceContextPool is designed to permit that.
    m_deviceMemoryManager->freeDeviceContext( const_cast<DeviceContext*>( &context ) );

    // Restore staged requests, and remove them from the request list (second chance algorithm)
    unsigned int numRequestedPages = pinnedRequestContext->arrayLengths[PAGE_REQUESTS_LENGTH];
    unsigned int numStalePages     = pinnedRequestContext->arrayLengths[STALE_PAGES_LENGTH];

    for( unsigned int i = 0; i < numRequestedPages; ++i )
    {
        if( restoreMapping( pinnedRequestContext->requestedPages[i] ) )
        {
            pinnedRequestContext->requestedPages[i] = pinnedRequestContext->requestedPages[numRequestedPages - 1];
            --numRequestedPages;
            --i;
        }
    }
    pinnedRequestContext->arrayLengths[PAGE_REQUESTS_LENGTH] = numRequestedPages;

    // Enqueue the requests for processing.
    // Must do this even when zero pages are requested to get proper end-to-end asynchronous communication via the Ticket mechanism.
    m_requestProcessor->addRequests( m_deviceIndex, stream, id, pinnedRequestContext->requestedPages, numRequestedPages );

    // Sort and stage stale pages, and update the LRU threshold
    unsigned int medianLruVal = 0;
    if( numStalePages > 0 )
    {
        if( context.lruTable != nullptr )
        {
            std::sort( pinnedRequestContext->stalePages, pinnedRequestContext->stalePages + numStalePages,
                       []( StalePage a, StalePage b ) { return a.lruVal < b.lruVal; } );
            medianLruVal = pinnedRequestContext->stalePages[numStalePages / 2].lruVal;
        }
        else
        {
            std::shuffle(pinnedRequestContext->stalePages, pinnedRequestContext->stalePages + numStalePages, m_rng);
        }

        if( m_evictionActive && getNumStagedPages() < m_options.maxStagedPages )
        {
            m_stagedPages.emplace_back( StagedPageList{m_pushMappingsEvent, std::deque<PageMapping>()} );
            stageStalePages( pinnedRequestContext, m_stagedPages.back().mappings );
        }
    }
    updateLruThreshold( numStalePages, pinnedRequestContext->maxStalePages, medianLruVal );

    // Return the RequestContext to its pool.
    m_pinnedRequestContextPool.push_back(pinnedRequestContext);
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
    pushMappingsAndInvalidations( context, stream );

    // Zero out the reference bits
    unsigned int referenceBitsSizeInBytes = idivCeil( context.maxNumPages, 8 );
    DEMAND_CUDA_CHECK( cuMemsetD8Async( reinterpret_cast<CUdeviceptr>( context.referenceBits ), 0, referenceBitsSizeInBytes, stream ) );

    // Record the event in the stream. pushMappings will be complete when it returns cudaSuccess
    DEMAND_CUDA_CHECK( cuEventRecord( m_pushMappingsEvent->event, stream ) );
    m_pushMappingsEvent->recorded = true;

    // Make a new event for the next time pushMappings is called
    m_pushMappingsEvent = std::make_shared<FutureEvent>();

    // Free the current PageMappingsContext when the preceding operations on the stream are done.
    m_pinnedMemoryPool->freeAsync( m_pageMappingsContextBlock, m_deviceIndex, stream);

    // Allocate a new PageMappingsContext for the next pushMappings cycle.
    m_pageMappingsContextBlock =
        m_pinnedMemoryPool->alloc( PageMappingsContext::getAllocationSize( m_options ), alignof( PageMappingsContext ) );
    m_pageMappingsContext = reinterpret_cast<PageMappingsContext*>( m_pageMappingsContextBlock.ptr );
    m_pageMappingsContext->init( m_options );

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

    while( !m_stagedPages.empty() && ( m_stagedPages[0].event->query() == CUDA_SUCCESS ) )
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

void PagingSystem::pushMappingsAndInvalidations( const DeviceContext& context, CUstream stream )
{
    // Mutex acquired in caller 

    // First push any new mappings
    const unsigned int numFilledPages = m_pageMappingsContext->numFilledPages;
    if( numFilledPages > 0 )
    {
        DEMAND_CUDA_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( context.filledPages.data ),
                                          reinterpret_cast<CUdeviceptr>( m_pageMappingsContext->filledPages ),
                                          numFilledPages * sizeof( PageMapping ), stream ) );
        launchPushMappings( stream, context, numFilledPages );
    }
    
    // Next, push the invalidated pages
    const unsigned int numInvalidatedPages = m_pageMappingsContext->numInvalidatedPages;
    if( numInvalidatedPages > 0 )
    {
        DEMAND_CUDA_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( context.invalidatedPages.data ),
                                          reinterpret_cast<CUdeviceptr>( m_pageMappingsContext->invalidatedPages ),
                                          numInvalidatedPages * sizeof( unsigned int ), stream ) );
        launchInvalidatePages( stream, context, numInvalidatedPages );
    }
    
    m_pageMappingsContext->clear();
}

void PagingSystem::invalidatePages( unsigned int              startId,
                                    unsigned int              endId,
                                    PageInvalidatorPredicate* predicate,
                                    const DeviceContext&      context,
                                    CUstream                  stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Remove specified page entries from the page table.
    std::set<unsigned int> stagedInvalidatedPages;
    auto p = m_pageTable.lower_bound( startId );
    while( p != m_pageTable.end() && p->first < endId )
    {
        const unsigned int pageId = p->first;
        const unsigned long long pageVal = p->second.entry;

        if( !predicate || (*predicate)( pageId, pageVal ) )
        {
            m_pageMappingsContext->invalidatedPages[m_pageMappingsContext->numInvalidatedPages++] = pageId;
            if( p->second.inStagedList )
            {
                stagedInvalidatedPages.insert( pageId );
            }
            p = m_pageTable.erase(p);

            // If the buffer for invalidations is about to overflow, push the invalidated pages to clear it. 
            // This should not happen very often.  Usually, the mappings will be pushed from pushMappings.
            if( m_pageMappingsContext->numInvalidatedPages >= m_pageMappingsContext->maxInvalidatedPages )
            {
                pushMappingsAndInvalidations( context, stream );
                cuStreamSynchronize( stream ); // wait for the stream because we will reuse the context
            }
        }
        else 
        {
            ++p;
        }
    }
    
    if( stagedInvalidatedPages.empty() )
    {
        return;
    }

    // Remove invalidated page entries from staged pages list 
    for( StagedPageList& spl : m_stagedPages )
    {
        for( int i=0; i < static_cast<int>( spl.mappings.size() ); ++i )
        {
            const unsigned int pageId = spl.mappings[i].id;
            if( pageId >= startId && pageId < endId )
            {
                if( stagedInvalidatedPages.find(pageId) != stagedInvalidatedPages.end() )
                {
                    spl.mappings[i] = spl.mappings.back();
                    spl.mappings.pop_back();
                    --i;
                }
            }
        }
    }
}

}  // namespace demandLoading
