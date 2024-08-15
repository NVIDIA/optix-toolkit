// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "RequestHandler.h"

#include <OptiXToolkit/DemandLoading/Resource.h>

namespace demandLoading {

class DemandLoaderImpl;

/// Page request handler for user-defined resources.
class ResourceRequestHandler : public RequestHandler
{
  public:
    /// Construct resource page request handler.
    ResourceRequestHandler( ResourceCallback callback, void *callbackContext, DemandLoaderImpl* loader )
        : m_callback( callback )
        , m_callbackContext( callbackContext )
        , m_loader( loader )
    {
    }

    /// Fill a request for the specified page using the given stream.
    void fillRequest( CUstream stream, unsigned int pageIndex ) override;

    /// Get the index of the first page table entry allocated to this resource.
    unsigned int getStartPage() const { return m_startPage; }

  private:
    ResourceCallback  m_callback;
    void*             m_callbackContext;
    DemandLoaderImpl* m_loader;
};

}  // namespace demandLoading
