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
    explicit PageTableManager( unsigned int totalPages, unsigned int backedPages )
        : m_totalPages( totalPages )
        , m_backedPages( backedPages )
        , m_nextUnbackedPage( backedPages )
    {
    }

    unsigned int getAvailableBackedPages() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_backedPages - m_nextBackedPage;
    }

    unsigned int getAvailableUnbackedPages() const 
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_totalPages - m_nextUnbackedPage;
    }

    /// Return the end page (one past the last used page).
    unsigned int getEndPage() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_nextUnbackedPage > m_backedPages ? m_nextUnbackedPage : m_nextBackedPage;
    }

    /// Reserve the specified number of contiguous page table entries, associating them with the
    /// specified request handler.  Returns the first page reserved.
    unsigned int reserveBackedPages( unsigned int numPages, RequestHandler* handler ) 
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT_MSG( m_nextBackedPage + numPages <= m_backedPages,
                           "Insufficient backed pages in demand loading page table" );

        return insertPageMapping( m_nextBackedPage, numPages, handler );
    }

    /// Reserve unbacked pages (pages with no backing storage on the device).
    unsigned int reserveUnbackedPages( unsigned int numPages, RequestHandler* handler )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        DEMAND_ASSERT_MSG( m_nextUnbackedPage + numPages <= m_totalPages, 
                           "Insufficient unbacked pages in demand loading page table" );

        return insertPageMapping( m_nextUnbackedPage, numPages, handler );
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
        return least != m_mappings.cend() ? least->handler : nullptr;
    }

    void removeRequestHandler( unsigned int pageId ) 
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        const auto least =
            std::lower_bound( m_mappings.cbegin(), m_mappings.cend(), pageId,
                              []( const PageMapping& entry, unsigned int id ) { return id > entry.lastPage; } );

        DEMAND_ASSERT_MSG( least != m_mappings.cend(), 
                           "Trying to replace nonexistent request handler" );
        
        size_t idx = least - m_mappings.begin();
        m_mappings[idx] = PageMapping{ least->firstPage, least->lastPage, &m_nullHandler };
    }

  private:
    struct PageMapping
    {
        unsigned int    firstPage;
        unsigned int    lastPage;
        RequestHandler* handler;
    };

    unsigned int insertPageMapping( unsigned int& nextPage, unsigned int numPages, RequestHandler* handler )
    {
        const unsigned int firstPage = nextPage;
        const unsigned int lastPage = firstPage + numPages - 1;
        if( handler )
            handler->setPageRange( firstPage, numPages );

        const PageMapping mapping{ firstPage, lastPage, handler };

        const auto least =
            std::lower_bound( m_mappings.begin(), m_mappings.end(), firstPage,
                              []( const PageMapping& entry, unsigned int id ) { return id > entry.lastPage; } );
        m_mappings.insert( least, mapping );

        nextPage += numPages;
        return firstPage;
    }

    unsigned int             m_totalPages;
    unsigned int             m_backedPages;

    unsigned int             m_nextBackedPage{};
    unsigned int             m_nextUnbackedPage{};

    std::vector<PageMapping> m_mappings;
    mutable std::mutex       m_mutex;

    RequestHandler           m_nullHandler;
};

}  // namespace demandLoading
