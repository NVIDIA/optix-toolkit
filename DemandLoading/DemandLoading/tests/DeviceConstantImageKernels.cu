// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cuda.h>

#include "DeviceConstantImageParams.h"

extern "C" __global__ void deviceReadConstantImage(const DeviceConstantImageParams params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= params.num_pixels )
        return;
    
    if( idx < 64 || (idx & 63) == 0 )
        params.output_buffer[idx] = float4{0.0f, 0.0f, 0.0f, 0.0f};
    else
        params.output_buffer[idx] = params.color;
}

__host__ void launchReadConstantImage( const DeviceConstantImageParams& params, CUstream stream )
{
    unsigned int       totalThreads    = params.num_pixels;
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBLocks       = ( totalThreads + threadsPerBlock - 1 ) / threadsPerBlock;

    dim3 grid( numBLocks, 1, 1 );
    dim3 block( threadsPerBlock, 1, 1 );

    deviceReadConstantImage<<<grid, block, 0U, stream>>>( params );
}

