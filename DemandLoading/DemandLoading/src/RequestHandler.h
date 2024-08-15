// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "Util/MutexArray.h"

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

namespace demandLoading {

/// A RequestHandler fills page requests for a particular resource, e.g. a demand-loaded texture.
/// RequestHandlers are associated with a range of pages by the PageTableManager and are invoked by
/// the RequestProcessor.
class RequestHandler
{
  public:
    /// Construct RequestHandler, which shares state with the DemandLoader.
    RequestHandler() = default;

    /// The destructor is virtual.
    virtual ~RequestHandler() = default;

    /// Set the range of pages assigned to this corresponding resource by the PageTableManager.
    /// These values are invariant once established.
    void setPageRange( unsigned int startPage, unsigned int numPages )
    {
        // If a page range has already been set, make sure the new range is a subset
        // of the existing range.
        if( m_mutex.get() != nullptr )
        {
            OTK_ASSERT_MSG( startPage >= m_startPage && startPage + numPages <= m_startPage + m_numPages,
                               "Cannot change request handler page range" );
            return;
        }

        // The MutexArray is used by fillRequest to ensure mutual exclusion on a per-page basis.
        m_startPage = startPage;
        m_numPages  = numPages;
        m_mutex.reset( new MutexArray( numPages ) );
    }

    /// Fill a request for the specified page using the given stream.
    virtual void fillRequest( CUstream /*stream*/, unsigned int /*pageId*/ ) {}

    /// Get the start page for the request handler
    unsigned int getStartPage() { return m_startPage; }

    /// Get the number of pages assigned to the request handler
    unsigned int getNumPages() { return m_numPages; }

  protected:
    unsigned int                m_startPage = 0;
    unsigned int                m_numPages  = 0;
    std::unique_ptr<MutexArray> m_mutex;
};

}  // namespace demandLoading
