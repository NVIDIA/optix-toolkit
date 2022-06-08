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

#include <cuda_runtime.h>

#include <cstring>

namespace demandLoading {

/// Device memory allocator.
class DeviceAllocator
{
  public:
    DeviceAllocator( unsigned int deviceIndex )
        : m_deviceIndex( deviceIndex )
    {
    }

    void* allocate( size_t numBytes )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

        void* result;
        DEMAND_CUDA_CHECK( cudaMalloc( &result, numBytes ) );
        return result;
    }

    void free( void* data )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaFree( data ) );
    }

    void setToZero( void* data, size_t numBytes )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaMemset( data, 0, numBytes ) );
    }

  private:
    unsigned int m_deviceIndex;
};


/// Pinned host memory allocator.
class PinnedAllocator
{
  public:
    static void* allocate( size_t numBytes )
    {
        void* result;
        DEMAND_CUDA_CHECK( cudaMallocHost( &result, numBytes, 0U ) );
        return result;
    }

    static void free( void* data ) { DEMAND_CUDA_CHECK( cudaFreeHost( data ) ); }

    static void setToZero( void* data, size_t numBytes ) { memset( data, 0, numBytes ); }
};

}  // namespace demandLoading
