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

#include "Util/Exception.h"

#include <DemandLoading/DeviceContext.h>  // for PageMapping
#include <DemandLoading/Options.h>
#include <DemandLoading/Ticket.h>

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

class PagingSystem
{
  public:
    /// Create paging system, allocating device memory based on the given options.
    PagingSystem( unsigned int         deviceIndex,
                  const Options&       options,
                  DeviceMemoryManager* deviceMemoryManager,
                  PinnedMemoryManager* pinnedMemoryManager,
                  RequestProcessor*    requestProcessor );

    /// Pull requests from device to system memory.
    void pullRequests( const DeviceContext& context, CUstream stream, unsigned int startPage, unsigned int endPage, Ticket ticket );

    /// Get the device index for this paging system.
    unsigned int getDeviceIndex() const { return m_deviceIndex; }

    // Add a page mapping (thread safe).  The device-side page table (etc.) is not updated until
    /// pushMappings is called.
    void addMapping( unsigned int pageId, unsigned int lruVal, unsigned long long entry );

    /// Check whether the specified page is resident (thread safe).
    bool isResident( unsigned int pageId );

    /// Push tile mappings to the device.  Returns the total number of new mappings.
    unsigned int pushMappings( const DeviceContext& context, CUstream stream );

    /// Free a staged page for reuse (thread safe). Return the page mapping in m so resources
    /// it holds can also be freed.
    bool freeStagedPage( PageMapping* m );

    /// Turn eviction on/off, (allows or stops staging stale pages)
    void activateEviction( bool activate ) { m_evictionActive = activate; }

    /// Returns whether eviction is turned on or off
    bool evictionIsActive() { return m_evictionActive; }

    /// Flush accumulated page mappings.  Used during trace file playback.
    void flushMappings();

  private:
    struct HostPageTableEntry
    {
        unsigned long long entry;
        bool               resident;      // Whether a page is considered resident on the GPU
        bool               staged;        // Pages that are currently staged (and not restored by second chance).
        bool               inStagedList;  // All pages that are in the staged list, whether restored or not.
    };

    Options              m_options{};
    unsigned int         m_deviceIndex = 0;
    DeviceMemoryManager* m_deviceMemoryManager{};
    PinnedMemoryManager* m_pinnedMemoryManager{};
    RequestProcessor*    m_requestProcessor{};

    PageMappingsContext* m_pageMappingsContext;  // owned by PinnedMemoryManager::m_pageMappingsContextPool

    std::map<unsigned int, HostPageTableEntry> m_pageTable;  // Host-side. Not copied to/from device. Used for eviction.
    std::mutex m_mutex;  // Guards m_pageTable and filledPages list (see addMapping).

    std::mt19937 m_rng; // Used for randomized eviction when LRU table is not present.

  private:
    // Variables related to eviction
    const unsigned int MIN_LRU_THRESHOLD = 2;
    bool               m_evictionActive  = false;
    unsigned int       m_launchNum       = 0;
    unsigned int       m_lruThreshold    = MIN_LRU_THRESHOLD;

    // Synchronization event for pushMappings
    struct FutureEvent
    {
        FutureEvent() { DEMAND_CUDA_CHECK( cudaEventCreate( &event ) ); }
        ~FutureEvent() { DEMAND_CUDA_CHECK( cudaEventDestroy( event ) ); }
        cudaError_t query() { return recorded ? cudaEventQuery( event ) : cudaErrorNotReady; }

        cudaEvent_t event{};
        bool        recorded = false;
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

    // A host function callback is used to invoke processRequests().
    friend class ProcessRequestsCallback;

    // Process requests, inserting them in the global request queue.
    void processRequests( const DeviceContext& context, RequestContext* requestContext, CUstream stream, Ticket ticket );

    // Update the lru threshold value
    void updateLruThreshold( unsigned int returnedStalePages, unsigned int requestedStalePages, unsigned int medianLruVal );

    // Stage pages for reuse (Remove their mappings on the host, and schedule removal of thier mappings on the device
    // the next time pushMappings is called.)
    void stageStalePages( RequestContext* requestContext, std::deque<PageMapping>& stagedMappings );

    // Get the number of staged pages (ready to be freed for reuse)
    size_t getNumStagedPages();

    // Add mapping function without mutex
    void addMappingBody( unsigned int pageId, unsigned int lruVal, unsigned long long entry );

    // Restore the mapping for a staged page if possible
    bool restoreMapping( unsigned int pageId );
};

}  // namespace demandLoading
