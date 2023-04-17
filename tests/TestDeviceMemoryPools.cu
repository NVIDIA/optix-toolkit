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

#include "CudaCheck.h"

#include <OptiXToolkit/Memory/DeviceFixedPool.h>
#include <OptiXToolkit/Memory/DeviceRingBuffer.h>
#include "TestDeviceMemoryPools.h"

#include <cuda.h>

using namespace otk;

__global__ static void deviceRingBufferTestKernel( DeviceRingBuffer ringBuffer, char** output, int width )
{
    unsigned long long handle;
    char* p = ringBuffer.alloc(32, &handle);

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x < width )
    {
        output[x] = p;
    }

    ringBuffer.free(handle);
}

__host__ void launchDeviceRingBufferTest( const DeviceRingBuffer& ringBuffer, char** output, int width )
{
    dim3 dimBlock( 512 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x );
    deviceRingBufferTestKernel<<<dimGrid, dimBlock>>>( ringBuffer, output, width );
    OTK_MEMORY_CUDA_CHECK( cudaGetLastError() );
}

__global__ static void deviceFixedPoolTestKernel( DeviceFixedPool fixedPool, char** output, int width )
{
    char* p = fixedPool.alloc();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x < width )
    {
        output[x] = p;
    }

    fixedPool.free(p);
}

__host__ void launchDeviceFixedPoolTest( const DeviceFixedPool& fixedPool, char** output, int width )
{
    dim3 dimBlock( 512 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x );
    deviceFixedPoolTestKernel<<<dimGrid, dimBlock>>>( fixedPool, output, width );
    OTK_MEMORY_CUDA_CHECK( cudaGetLastError() );
}




