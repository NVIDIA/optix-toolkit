// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TestDeviceMemoryPools.h"
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/DeviceFixedPool.h>
#include <OptiXToolkit/Memory/DeviceRingBuffer.h>
#include <OptiXToolkit/Memory/InterleavedAccess.h>
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
    OTK_ERROR_CHECK( cudaGetLastError() );
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
    OTK_ERROR_CHECK( cudaGetLastError() );
}


__global__ static void deviceFixedPoolInterleavedTestKernel( DeviceFixedPool fixedPool, char** output, int width )
{
    int* p = (int*) fixedPool.alloc();
    InterleavedAccess ia( p );

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x < width )
    {
        output[x] = ( char* )p;
        for( int i = 0; i < 8; ++i) // each allocation is 32 bytes (8 ints)
        {
            ia.setInt( &p[i], i );
        }
    }

    fixedPool.free( (char*)p );
}

__host__ void launchDeviceFixedPoolInterleavedTest( const DeviceFixedPool& fixedPool, char** output, int width )
{
    dim3 dimBlock( 512 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x );
    deviceFixedPoolInterleavedTestKernel<<<dimGrid, dimBlock>>>( fixedPool, output, width );
    OTK_ERROR_CHECK( cudaGetLastError() );
}

