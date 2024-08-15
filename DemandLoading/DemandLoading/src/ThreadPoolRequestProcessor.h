// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/RequestProcessor.h>

#include "RequestQueue.h"

#include <cuda.h>

#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace demandLoading {

class PageTableManager;

class ThreadPoolRequestProcessor : public RequestProcessor
{
  public:
    /// Construct request processor, which uses the given PageTableManager to
    /// find the RequestHandler associated with a range of pages.
    ThreadPoolRequestProcessor( std::shared_ptr<PageTableManager> pageTableManager, const Options& options );
    ~ThreadPoolRequestProcessor() override = default;

    /// Stop processing requests, terminating threads.
    void stop() override;

    /// Add a batch of page requests to the request queue.
    void addRequests( CUstream stream, unsigned id, const unsigned int* pageIds, unsigned int numPageIds ) override;

    /// Add a request filter to preprocess batches of requests
    void setRequestFilter( std::shared_ptr<RequestFilter> requestFilter ) { m_requestFilter = requestFilter; }

    /// Set the ticket that will track requests with the given ticket id
    void setTicket( unsigned int id, Ticket ticket );

private:
    std::shared_ptr<PageTableManager> m_pageTableManager;
    std::unique_ptr<RequestQueue>     m_requests;
    std::vector<std::thread>          m_threads;
    std::map<unsigned int, Ticket>    m_tickets;
    std::mutex                        m_ticketsMutex;
    Options                           m_options;
    bool                              m_started = false;
    std::shared_ptr<RequestFilter>    m_requestFilter;

    /// Start processing requests.
    void start();

    // Per-thread worker function.
    void worker();
};

}  // namespace demandLoading
