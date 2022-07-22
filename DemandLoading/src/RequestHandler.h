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

#include "Util/MutexArray.h"

#include <cuda.h>

namespace demandLoading {

/// A RequestHandler fills page requests for a particular resource, e.g. a demand-loaded texture.
/// RequestHandlers are associated with a range of pages by the PageTableManager and are invoked by
/// the RequestProcessor.
class RequestHandler
{
  public:
    /// Construct RequestHandler, which shares state with the DemandLoader.
    RequestHandler() { }

    /// The destructor is virtual.
    virtual ~RequestHandler() { }

    /// Set the range of pages assigned to this corresponding resource by the PageTableManager.
    /// These values are invariant once established.
    void setPageRange( unsigned int startPage, unsigned int numPages )
    {
        m_startPage = startPage;
        m_numPages  = numPages;

        // The MutexArray can be used by fillRequest to ensure mutual exclusion on a per-page basis.
        m_mutex.reset( new MutexArray( numPages ) );
    }

    /// Fill a request for the specified page on the specified device using the given stream.
    virtual void fillRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId ) = 0;

  protected:
    unsigned int                m_startPage = 0;
    unsigned int                m_numPages  = 0;
    std::unique_ptr<MutexArray> m_mutex;
};

}  // namespace demandLoading
