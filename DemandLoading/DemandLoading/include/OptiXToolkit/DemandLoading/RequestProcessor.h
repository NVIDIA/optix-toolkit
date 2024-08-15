// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/Ticket.h>
#include <OptiXToolkit/DemandLoading/RequestFilter.h>

#include <cuda.h>

namespace demandLoading {

class RequestProcessor
{
  public:
    virtual ~RequestProcessor() = default;

    /// Add a batch of page requests from the specified device to the request queue.
    virtual void addRequests( CUstream stream, unsigned int id, const unsigned int* pageIds, unsigned int numPageIds ) = 0;

    /// Stop processing requests, waking and joining with worker threads.
    virtual void stop() = 0;
};

}  // namespace demandLoading
