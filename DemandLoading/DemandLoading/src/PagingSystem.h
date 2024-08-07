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

#include <OptiXToolkit/DemandLoading/DeviceContext.h>  // for PageMapping
#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/Ticket.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/MemoryPool.h>
#include <OptiXToolkit/Memory/RingSuballocator.h>

#include <cuda.h>

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <random>

namespace demandLoading {

struct DeviceContext;
class DeviceMemoryManager;
struct PageMappingsContext;
class PinnedMemoryManager;
struct RequestContext;
class RequestProcessor;
class TicketImpl;

class PageInvalidatorPredicate
{
  public:
    virtual bool operator() ( unsigned int pageId, unsigned long long pageVal, CUstream stream ) = 0;
    virtual ~PageInvalidatorPredicate() {};
};

class PagingSystem
{
  public:
    /// Create paging system, allocating device memory based on the given options.
    PagingSystem( std::shared_ptr<Options> options,
                  DeviceMemoryManager*     deviceMemoryManager,
                  otk::MemoryPool<otk::PinnedAllocator, otk::RingSuballocator>* pinnedMemoryPool,
                  RequestProcessor* requestProcessor );

    virtual ~PagingSystem();
    
    /// Pull requests from device to system memory.
    void pullRequests( const DeviceContext& context, CUstream stream, unsigned int id, unsigned int startPage, unsigned int endPage );

    // Add a page mapping (thread safe). The device-side page table (etc.) is not updated until
    /// pushMappings is called.
    void addMapping( unsigned int pageId, unsigned int lruVal, unsigned long long entry );

    /// Add a page mapping (not thread safe). Exposed for PageInvalidatorPredicate callbacks that
    /// need to map pages.
    void addMappingBody( unsigned int pageId, unsigned int lruVal, unsigned long long entry );

    /// Check whether the specified page is resident (thread safe).
    bool isResident( unsigned int pageId, unsigned long long* entry = nullptr );

    /// Push tile mappings to the device.  Returns the total number of new mappings.
    unsigned int pushMappings( const DeviceContext& context, CUstream stream );

    /// Free a staged page for reuse (thread safe). Return the page mapping in m so resources
    /// it holds can also be freed.
    bool freeStagedPage( PageMapping* m );

    /// Turn eviction on/off, (allows or stops staging stale pages)
    void activateEviction( bool activate ) { m_evictionActive = activate; }

    /// Returns whether eviction is turned on or off
    bool evictionIsActive() { return m_evictionActive; }

    /// Invalidate a half open interval of page ids, from startId up to but not including endId, based on a predicate
    void invalidatePages( unsigned int startId, unsigned int endId, PageInvalidatorPredicate* predicate, const DeviceContext& context, CUstream stream );

  private:
    struct HostPageTableEntry
    {
        unsigned long long entry;
        bool               resident;      // Whether a page is considered resident on the GPU
        bool               staged;        // Pages that are currently staged (and not restored by second chance).
        bool               inStagedList;  // All pages that are in the staged list, whether restored or not.
    };

    std::shared_ptr<Options> m_options{};
    DeviceMemoryManager*     m_deviceMemoryManager{};
    RequestProcessor*        m_requestProcessor{};

    otk::MemoryBlockDesc m_pageMappingsContextBlock;
    PageMappingsContext* m_pageMappingsContext; 
    otk::MemoryPool<otk::PinnedAllocator, otk::RingSuballocator>* m_pinnedMemoryPool;

    std::map<unsigned int, HostPageTableEntry> m_pageTable;  // Host-side. Not copied to/from device. Used for eviction.
    std::mutex m_mutex;  // Guards m_pageTable and filledPages list (see addMapping).

    std::mt19937 m_rng; // Used for randomized eviction when LRU table is not present.

    // Variables related to eviction
    const unsigned int MIN_LRU_THRESHOLD = 2;
    bool               m_evictionActive  = false;
    unsigned int       m_launchNum       = 0;
    unsigned int       m_lruThreshold    = MIN_LRU_THRESHOLD;

    // Synchronization event for pushMappings
    struct FutureEvent
    {
        FutureEvent() { OTK_ERROR_CHECK( cuEventCreate( &event, CU_EVENT_DEFAULT ) ); }
        ~FutureEvent() { OTK_ERROR_CHECK_NOTHROW( cuEventDestroy( event ) ); }
        CUresult query() { return recorded ? cuEventQuery( event ) : CUDA_ERROR_NOT_READY; }

        CUevent event{};
        bool    recorded = false;
    };
    std::shared_ptr<FutureEvent> m_pushMappingsEvent = nullptr;

    // Staged tiles (tiles set as non-resident on the host, which can be freed once they are set
    // as non-resident on the device).
    struct StagedPageList
    {
        std::shared_ptr<FutureEvent> event;
        std::deque<PageMapping>      mappings;
    };
    std::deque<StagedPageList> m_stagedPages;

    // Pool of pinned RequestContext for processRequests function
    std::vector<RequestContext*> m_pinnedRequestContextPool;

    // CUDA module containing the PTX for the paging kernels.
    CUmodule m_pagingKernels{};

    // A host function callback is used to invoke processRequests().
    friend class ProcessRequestsCallback;

    // Process requests, inserting them in the global request queue.
    void processRequests( const DeviceContext& context, RequestContext* pinnedRequestContext, CUstream stream, unsigned int id );

    // Update the lru threshold value
    void updateLruThreshold( unsigned int returnedStalePages, unsigned int requestedStalePages, unsigned int medianLruVal );

    // Stage pages for reuse (Remove their mappings on the host, and schedule removal of thier mappings on the device
    // the next time pushMappings is called.)
    void stageStalePages( RequestContext* requestContext, std::deque<PageMapping>& stagedMappings );

    // Get the number of staged pages (ready to be freed for reuse)
    size_t getNumStagedPages();

    // Allocate a PageMappingsContext in pinned memory.
    void initPageMappingsContext();

    // Restore the mapping for a staged page if possible
    bool restoreMapping( unsigned int pageId );

    // Push invalidated pages to device
    void pushMappingsAndInvalidations( const DeviceContext& context, CUstream stream );
};

}  // namespace demandLoading
