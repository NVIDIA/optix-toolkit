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
#include <cuda_runtime.h>

#include <cstring>

namespace demandLoading {

const unsigned int HOST_DEVICE = 0xFFFFFFFF;

/// Host allocator using malloc
class HostAllocator
{
  public:
    void* allocate( size_t numBytes, CUstream dummy = 0 ) { return malloc( numBytes ); }
    void free( void* ptr, CUstream dummy = 0 ) { ::free( ptr ); }
    void set( void* ptr, int val, size_t numBytes, CUstream dummy = 0 ) { memset( ptr, val, numBytes ); }
    unsigned int getDeviceIndex() const { return HOST_DEVICE; }
    bool         allocationIsHandle() const { return false; }
};

/// Pinned host allocator using cudaMallocHost
class PinnedAllocator
{
  public:
    void* allocate( size_t numBytes, CUstream dummy = 0 )
    {
        void* result;
        DEMAND_CUDA_CHECK( cudaMallocHost( &result, numBytes, 0U ) );
        return result;
    }
    void free( void* ptr, CUstream dummy = 0 ) { DEMAND_CUDA_CHECK( cudaFreeHost( ptr ) ); }
    void set( void* ptr, int val, size_t numBytes, CUstream dummy = 0 ) { memset( ptr, val, numBytes ); }
    unsigned int getDeviceIndex() const { return HOST_DEVICE; }
    bool         allocationIsHandle() const { return false; }
};

/// Device allocator using cudaMalloc
class DeviceAllocator
{
  public:
    DeviceAllocator( unsigned int deviceIndex = 0 )
        : m_deviceIndex( deviceIndex )
    {
    }

    void* allocate( size_t numBytes, CUstream dummy = 0 )
    {
        if( numBytes == 0 )
            return nullptr;  // cuMemAlloc does not handle this.
        void* result;
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &result ), numBytes ) );
        return result;
    }

    void free( void* ptr, CUstream dummy = 0 )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( ptr ) ) );
    }

    void set( void* ptr, int val, size_t numBytes, CUstream dummy = 0 )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( ptr ), val, numBytes ) );
    }

    unsigned int getDeviceIndex() const { return m_deviceIndex; }
    bool         allocationIsHandle() const { return false; }

  private:
    unsigned int m_deviceIndex;
};

/// Async device allocator using cudaMallocAsync
class DeviceAsyncAllocator
{
  public:
    DeviceAsyncAllocator( unsigned int deviceIndex = 0 )
        : m_deviceIndex( deviceIndex )
    {
    }

    void* allocate( size_t numBytes, CUstream stream = 0 )
    {
        if( numBytes == 0 )
            return nullptr;  // cuMemAlloc does not handle this.
        void* result;
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemAllocAsync( reinterpret_cast<CUdeviceptr*>( &result ), numBytes, stream ) );
        return result;
    }

    void free( void* ptr, CUstream stream = 0 )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemFreeAsync( reinterpret_cast<CUdeviceptr>( ptr ), stream ) );
    }

    void set( void* ptr, int val, size_t numBytes, CUstream stream = 0 )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemsetD8Async( reinterpret_cast<CUdeviceptr>( ptr ), val, numBytes, stream ) );
    }

    unsigned int getDeviceIndex() const { return m_deviceIndex; }
    bool         allocationIsHandle() const { return false; }

  private:
    unsigned int m_deviceIndex;
};

/// Texture tile allocator using cuMemCreate
class TextureTileAllocator
{
  public:
    TextureTileAllocator( unsigned int deviceIndex = 0 )
        : m_deviceIndex( deviceIndex )
        , m_allocationProp( makeAllocationProp( deviceIndex ) )
    {
    }

    void* allocate( size_t numBytes, CUstream dummy = 0 )
    {
        CUmemGenericAllocationHandle handle;
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemCreate( &handle, numBytes, &m_allocationProp, 0U ) );
        return reinterpret_cast<void*>( handle );
    }

    void free( void* handle, CUstream dummy = 0 )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemRelease( reinterpret_cast<CUmemGenericAllocationHandle>( handle ) ) );
    }

    unsigned int getDeviceIndex() const { return m_deviceIndex; }
    bool         allocationIsHandle() const { return true; }  // The allocation is not a pointer that can be incremented

    static CUmemAllocationProp makeAllocationProp( unsigned int deviceIndex )
    {
        CUmemAllocationProp prop{};
        prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location         = {CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>( deviceIndex )};
        prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
        return prop;
    }

    static size_t getRecommendedAllocationSize( unsigned int deviceIndex )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
        size_t              size;
        CUmemAllocationProp prop( makeAllocationProp( deviceIndex ) );
        DEMAND_CUDA_CHECK( cuMemGetAllocationGranularity( &size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED ) );
        return size ? size : 8 << 20;  // get the recommended size, or 8MB if it returns 0
    }

  private:
    unsigned int        m_deviceIndex;
    CUmemAllocationProp m_allocationProp{};
};

}  // namespace demandLoading
