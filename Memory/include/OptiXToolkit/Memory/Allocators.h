// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

#include <algorithm>
#include <cstring>

namespace otk {

const uint64_t DEFAULT_ALLOC_SIZE = 8 * 1024 * 1024;
const unsigned int HOST_DEVICE = 0xFFFFFFFF;

/// Host allocator using malloc
class HostAllocator
{
  public:
    void* allocate( size_t numBytes, CUstream /*dummy*/ = 0 ) { return malloc( numBytes ); }
    void free( void* ptr, CUstream /*dummy*/ = 0 ) { ::free( ptr ); }
    void set( void* ptr, int val, size_t numBytes, CUstream /*dummy*/ = 0 ) { memset( ptr, val, numBytes ); }
    bool allocationIsHandle() const { return false; }
};

/// Pinned host allocator using cuMallocHost
class PinnedAllocator
{
  public:
    void* allocate( size_t numBytes, CUstream /*dummy*/ = 0 )
    {
        void* result = nullptr;
        OTK_ERROR_CHECK_NOTHROW( cuMemAllocHost( &result, numBytes ) );
        return result;
    }
    void free( void* ptr, CUstream /*dummy*/ = 0 ) { OTK_ERROR_CHECK( cuMemFreeHost( ptr ) ); }
    void set( void* ptr, int val, size_t numBytes, CUstream /*dummy*/ = 0 ) { memset( ptr, val, numBytes ); }
    bool allocationIsHandle() const { return false; }
};

/// Device allocator using cuMemAlloc
class DeviceAllocator
{
  public:
    DeviceAllocator()
    {
        // Record current CUDA context.
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );
    }

    void* allocate( size_t numBytes, CUstream /*dummy*/ = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        if( numBytes == 0 )
            return nullptr;  // cuMemAlloc does not handle this.
        void* result = nullptr;
        OTK_ERROR_CHECK_NOTHROW( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &result ), numBytes ) );
        return result;
    }

    void free( void* ptr, CUstream /*dummy*/ = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( ptr ) ) );
    }

    void set( void* ptr, int val, size_t numBytes, CUstream /*dummy*/ = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        OTK_ERROR_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( ptr ), static_cast<char>( val ), numBytes ) );
    }

    bool allocationIsHandle() const { return false; }

  private:
    CUcontext m_context;
};

/// Async device allocator using cuMemAllocAsync
class DeviceAsyncAllocator
{
  public:
    DeviceAsyncAllocator()
    {
        // Record current CUDA context.
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );

#if OTK_USE_CUDA_MEMORY_POOLS
        CUdevice device;
        OTK_ERROR_CHECK( cuCtxGetDevice( &device ) );
        OTK_ERROR_CHECK( cuDeviceGetAttribute( &m_usePools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device ) );
#endif
    }

    void* allocate( size_t numBytes, CUstream stream = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
        if( numBytes == 0 )
            return nullptr;  // cuMemAlloc does not handle this.
        void* result = nullptr;

#if OTK_USE_CUDA_MEMORY_POOLS
        if( m_usePools )
            OTK_ERROR_CHECK_NOTHROW( cuMemAllocAsync( reinterpret_cast<CUdeviceptr*>( &result ), numBytes, stream ) );
        else
            OTK_ERROR_CHECK_NOTHROW( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &result ), numBytes ) );
#else
        OTK_ERROR_CHECK_NOTHROW( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &result ), numBytes ) );
#endif

        return result;
    }

    void free( void* ptr, CUstream stream = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );

#if OTK_USE_CUDA_MEMORY_POOLS
        if( m_usePools )
            OTK_ERROR_CHECK( cuMemFreeAsync( reinterpret_cast<CUdeviceptr>( ptr ), stream ) );
        else
            OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( ptr ) ) );
#else
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( ptr ) ) );
#endif
    }

    void set( void* ptr, int val, size_t numBytes, CUstream stream = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
        OTK_ERROR_CHECK( cuMemsetD8Async( reinterpret_cast<CUdeviceptr>( ptr ), static_cast<char>( val ), numBytes, stream ) );
    }

    bool allocationIsHandle() const { return false; }

  private:
    CUcontext m_context;
    int m_usePools = 0;
};

/// Texture tile allocator using cuMemCreate
class TextureTileAllocator
{
  public:
    TextureTileAllocator()
        : m_allocationProp( makeAllocationProp() )
    {
        // Record current CUDA context.
        OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );
    }

    void* allocate( size_t numBytes, CUstream /*dummy*/ = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        CUmemGenericAllocationHandle handle = 0U;
        OTK_ERROR_CHECK_NOTHROW( cuMemCreate( &handle, numBytes, &m_allocationProp, 0U ) );
        return reinterpret_cast<void*>( handle );
    }

    void free( void* handle, CUstream /*dummy*/ = 0 )
    {
        OTK_ASSERT_CONTEXT_IS( m_context );
        OTK_ERROR_CHECK( cuMemRelease( reinterpret_cast<CUmemGenericAllocationHandle>( handle ) ) );
    }

    bool allocationIsHandle() const { return true; }  // The allocation is not a pointer that can be incremented

    static CUmemAllocationProp makeAllocationProp()
    {
        CUdevice device;
        OTK_ERROR_CHECK( cuCtxGetDevice( &device ) );

        CUmemAllocationProp prop{};
        prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location         = {CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>( device )};
        prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
        return prop;
    }

    static size_t getRecommendedAllocationSize()
    {
        size_t              size;
        CUmemAllocationProp prop( makeAllocationProp() );
        OTK_ERROR_CHECK( cuMemGetAllocationGranularity( &size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED ) );
        return std::max( size, static_cast<size_t>( DEFAULT_ALLOC_SIZE ) );
    }

  private:
    CUcontext           m_context;
    CUmemAllocationProp m_allocationProp{};
};

}  // namespace otk
