//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

namespace otk {

// DeviceRingBuffer implements a device-side ring buffer.  Allocations
// do not need to be freed, but buffer overflow can be detected by calling
// free after each allocation is done.

const unsigned long long BAD_ALLOC = 0xFFFFFFFFFFFFFFFFULL;
const unsigned int       WARP_SIZE = 32;

#ifdef __CUDACC__
__forceinline__ __device__ unsigned int getLaneId()
{
    unsigned ret;
    asm volatile( "mov.u32 %0, %%laneid;" : "=r"( ret ) );
    return ret;
}
#endif

struct DeviceRingBuffer
{
    char*               buffer;       // Backing store for the ring buffer.
    unsigned long long  buffSize;     // Size of the buffer in bytes. must be power of 2.
    unsigned long long* nextStart;    // Offset of the next allocation (% buffSize).
    bool                allocByWarp;  // Whether to allocate by warp or by individual threads.

#ifndef __CUDACC__
    /// Initialize the ring buffer
    void init( unsigned long long _buffSize, bool _allocByWarp )
    {
        allocByWarp = _allocByWarp;
        buffSize    = _buffSize;
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &buffer ), buffSize ) );
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &nextStart ), sizeof( unsigned long long ) ) );
        clear( 0 );
    }

    /// Tear down, freeing all the memory
    void tearDown()
    {
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( buffer ) ) );
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( nextStart ) ) );
    }

    /// Clear the allocator
    void clear( CUstream stream = 0 )
    {
        OTK_ERROR_CHECK( cuMemsetD8Async( reinterpret_cast<CUdeviceptr>( nextStart ), 0, sizeof( unsigned long long ), stream ) );
    }
#endif

#ifdef __CUDACC__
    /// Allocate memory from the ring buffer, returning a pointer to it, and the memory handle.
    __forceinline__ __device__ char* alloc( unsigned long long allocSize, unsigned long long* handle )
    {
        return ( allocByWarp ) ? allocW( allocSize, handle ) : allocT( allocSize, handle );
    }

    /// Allocate memory from the ring buffer, returning a pointer to it.
    __forceinline__ __device__ char* alloc( unsigned long long allocSize )
    {
        unsigned long long handle;
        return alloc( allocSize, &handle );
    }

    /// Get the pointer for a given handle
    __forceinline__ __device__ char* getPointer( unsigned long long handle )
    {
        return buffer + ( handle & ( buffSize - 1 ) );
    }

    /// Calling free is not required for DeviceRingBuffer, but the call checks to see
    /// if the buffer has overflowed (memory was reallocated before being freed).
    __forceinline__ __device__ bool free( unsigned long long handle )
    {
        if( !allocByWarp )
            return ( ( handle == BAD_ALLOC ) || ( handle + buffSize > atomicAdd( nextStart, 0 ) ) );

        const unsigned int activeMask = __activemask();
        const unsigned int leadLane   = __ffs( activeMask ) - 1;
        const unsigned int laneId     = getLaneId();

        bool rval = 0;
        if( laneId == leadLane )
            rval = ( ( handle == BAD_ALLOC ) || ( handle + buffSize > atomicAdd( nextStart, 0 ) ) );
        rval     = __shfl_sync( activeMask, rval, leadLane );
        return rval;
    }

  protected:
    // Implementation of Thread-based memory allocation
    __forceinline__ __device__ char* allocT( unsigned long long allocSize, unsigned long long* handle )
    {
        *handle = atomicAdd( nextStart, allocSize );
        if( ( *handle & ( buffSize - 1 ) ) + allocSize <= buffSize )
            return getPointer( *handle );

        // Previous allocation straddled end of ring buffer, try again
        *handle = atomicAdd( nextStart, allocSize );
        if( ( *handle & ( buffSize - 1 ) ) + allocSize <= buffSize )
            return getPointer( *handle );

        *handle = BAD_ALLOC;
        return nullptr;
    }

    // Implementation of Warp-based memory allocation
    __forceinline__ __device__ char* allocW( unsigned long long allocSize, unsigned long long* handle )
    {
        const unsigned int activeMask = __activemask();
        const unsigned int leadLane   = __ffs( activeMask ) - 1;
        const unsigned int laneId     = getLaneId();

        unsigned long long ptrBase = 0ULL;
        if( laneId == leadLane )
            ptrBase = reinterpret_cast<unsigned long long>( allocT( allocSize * WARP_SIZE, handle ) );
        ptrBase     = __shfl_sync( activeMask, ptrBase, leadLane );

        return reinterpret_cast<char*>( ptrBase + ( allocSize * laneId ) );
    }
#endif
};

}  // namespace otk
