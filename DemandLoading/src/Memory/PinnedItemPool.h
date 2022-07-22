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

#include "Memory/AsyncItemPool.h"
#include "Util/Exception.h"

#include <cuda.h>

namespace demandLoading {

/// PinnedItemPool is a thread-safe, fixed-capacity pool of items that are allocated in pinned
/// memory.  The base class, AsyncItemPool, provides a free() method takes a stream argument, and
/// the freed item is not reused until all operations on that stream have completed.  This allows
/// items in a PinnedMemoryPool to be use as the source or destination of an asynchronous memcpy.
/// PinnedItemPool is intended for use with "plain old data" (POD) types.  Initializing and
/// destroying elements is the client's responsibility.
template <typename Item>
class PinnedItemPool : public AsyncItemPool<Item>
{
  public:
    /// Construct a PinnedItemPool of the specified capacity.
    PinnedItemPool( size_t capacity )
    {
        capacity = ( capacity > 0 ) ? capacity : 1ull;
        size_t itemSize = align( sizeof( Item ), alignof( Item ) );
        m_capacityInBytes = itemSize * capacity;
        DEMAND_CUDA_CHECK( cudaMallocHost( &m_data, m_capacityInBytes, 0U ) );

        AsyncItemPool<Item>::init( m_data, capacity );
    }

    /// Destroy the PinnedItemPool, reclaiming memory.
    ~PinnedItemPool() { DEMAND_CUDA_CHECK( cudaFreeHost( m_data ) ); }

    /// Get the total amount of pinned memory allocated.
    size_t getTotalPinnedMemory() const { return m_capacityInBytes; }

  private:
    Item*  m_data;
    size_t m_capacityInBytes;
};

}  // namespace demandLoading
