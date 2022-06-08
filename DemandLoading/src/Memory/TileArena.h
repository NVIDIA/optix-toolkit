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

namespace demandLoading {

class TileArena
{
  public:
    /// Get the arena size recommended by CUDA.  Typically this gives 32 tiles per arena.
    static size_t getRecommendedSize( unsigned int deviceIndex )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
        size_t              size;
        CUmemAllocationProp prop( makeAllocationProp( deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemGetAllocationGranularity( &size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED ) );
        return size;
    }

    /// Create tile arena of the specified size for the specified device.
    static TileArena create( unsigned int deviceIndex, size_t capacity ) { return TileArena().init( deviceIndex, capacity ); }

    /// Destroy the arena, reclaiming its memory.
    void destroy()
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemRelease( m_handle ) );
    }

    /// Get the generic allocation handle.
    CUmemGenericAllocationHandle getHandle() const { return m_handle; }

    /// Allocate the specified amount memory from the arena, which must not exceed the remaining
    /// capacity of the arena.  Returns the offset of the allocation.  Individual allocations cannot
    /// be freed.
    size_t allocate( size_t numBytes )
    {
        DEMAND_ASSERT( m_size + numBytes <= m_capacity );
        size_t offset = m_size;
        m_size += numBytes;
        return offset;
    }

    /// Get the current size in bytes.
    size_t size() const { return m_size; }

    /// Get the capacity in bytes.
    size_t capacity() const { return m_capacity; }

  private:
    unsigned int                 m_deviceIndex = 0;
    CUmemGenericAllocationHandle m_handle{};
    size_t                       m_size     = 0;
    size_t                       m_capacity = 0;

    // Construct allocation properties.
    static CUmemAllocationProp makeAllocationProp( unsigned int deviceIndex )
    {
        CUmemAllocationProp prop{};
        prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location         = {CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>( deviceIndex )};
        prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
        return prop;
    }

    TileArena& init( unsigned int deviceIndex, size_t capacity )
    {
        m_deviceIndex = deviceIndex;
        m_capacity = capacity;

        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        CUmemAllocationProp prop( makeAllocationProp( m_deviceIndex ) );

        DEMAND_CUDA_CHECK( cuMemCreate( &m_handle, m_capacity, &prop, 0 ) );
        return *this;
    }
};

}  // namespace demandLoading
