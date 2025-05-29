// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TestDrawTexture.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>

__device__ __forceinline__ float mix( float a, float b, float x )
{
    return (1.0f-x)*a + x*b;
}

__global__ static void drawTextureKernel( float4* image, int width, int height, unsigned long long texObj, 
                                          float2 uv00, float2 uv11, float2 ddx, float2 ddy )
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if( i >= width || j >= height )
        return;

    float x = ( i + 0.5f ) / width;
    float y = ( j + 0.5f ) / height;
    float s = mix( uv00.x, uv11.x, x );
    float t = mix( uv00.y, uv11.y, y );

    int pixelId = j * width + i;
    image[pixelId] = tex2DGrad<float4>( texObj, s, t, ddx, ddy );
}

__host__ void launchDrawTextureKernel( CUstream stream, float4* image, int width, int height, unsigned long long texObj,
                                       float2 uv00, float2 uv11, float2 ddx, float2 ddy )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    drawTextureKernel<<<dimGrid, dimBlock, 0U, stream>>>( image, width, height, texObj, uv00, uv11, ddx, ddy );
    OTK_ERROR_CHECK( cudaGetLastError() );
}
