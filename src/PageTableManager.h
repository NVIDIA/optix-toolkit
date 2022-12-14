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

#include "RequestHandler.h"
#include "Util/Exception.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <vector>

namespace demandLoading {

class RequestHandler;

/// The PageTableManager is used to reserve a contiguous range of page table entries.  It keeps a
/// mapping that allows the request handler corresponding to a page table entry to be determined in
/// log(N) time.
class PageTableManager
{
  public:
    explicit PageTableManager( unsigned int totalPages )
        : m_totalPages( totalPages )
    {
    }

    unsigned int getAvailablePages() const { return m_totalPages - m_nextPage; }

    unsigned int getHighestUsedPage() const { return m_nextPage - 1; }

    /// Reserve the specified number of contiguous page table entries, associating them with the
    /// specified request handler.  Returns the first page reserved.
    unsigned int reserve( unsigned int numPages, RequestHandler* handler )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT_MSG( getAvailablePages() >= numPages, "Insufficient pages in demand loading page table" );

        unsigned int firstPage = m_nextPage;
        handler->setPageRange( firstPage, numPages );

        unsigned int lastPage =  m_nextPage + numPages - 1;
        const PageMapping mapping{firstPage, lastPage, handler};
        m_mappings.push_back( mapping );

        m_nextPage += numPages;
        return firstPage;
    }

    /// Find the request handler associated with the specified page.  Returns nullptr if not found.
    RequestHandler* getRequestHandler( unsigned int pageId ) const
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        // Pages are allocated in increasing order, so the array of mappings is sorted, allowing us
        // to use binary search to find the the given page id.
        const auto least =
            std::lower_bound( m_mappings.cbegin(), m_mappings.cend(), pageId,
                              []( const PageMapping& entry, unsigned int id ) { return id > entry.lastPage; } );
        return ( least != m_mappings.cend() ) ? least->handler : nullptr;
    }

  private:
    struct PageMapping
    {
        unsigned int    firstPage;
        unsigned int    lastPage;
        RequestHandler* handler;
    };

    unsigned int             m_totalPages;
    unsigned int             m_nextPage{};
    std::vector<PageMapping> m_mappings;
    mutable std::mutex       m_mutex;
};

}  // namespace demandLoading
