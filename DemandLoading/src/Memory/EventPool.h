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

#include <cuda.h>

#include <vector>

namespace demandLoading {

/// EventPool is a pool of CUDA events with unlimited capacity.  Not thread safe.
class EventPool
{
  public:
    /// Construct an event pool, reserving storage for the specified number of events.
    EventPool( unsigned int deviceIndex, size_t count = 0 )
        : m_deviceIndex( deviceIndex )
    {
        m_events.reserve( count );
    }

    /// Destroy the event pool.
    ~EventPool()
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        for( CUevent event : m_events )
        {
            DEMAND_CUDA_CHECK( cuEventDestroy( event ) );
        }
    }

    /// Allocate an event.  Not thread safe.
    CUevent allocate()
    {
        // Return an event from the free list if possible.
        if( !m_freeList.empty() )
        {
            CUevent event = m_freeList.back();
            m_freeList.pop_back();
            return event;
        }

        // Allocate a new event.
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        CUevent event;
        DEMAND_CUDA_CHECK( cuEventCreate( &event, 0U ) );
        m_events.push_back( event );
        return event;
    }

    /// Return the given event back to the pool.  Not thread safe.
    void free( CUevent event ) { m_freeList.push_back( event ); }

    /// Get the size of the event pool.
    size_t size() const { return m_events.size() - m_freeList.size(); }

    /// Get the capacity of the event pool.
    size_t capacity() const { return m_events.capacity(); }

  private:
    unsigned int         m_deviceIndex;
    std::vector<CUevent> m_events;
    std::vector<CUevent> m_freeList;
};

}  // namespace demandLoading
