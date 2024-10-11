// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/DemandLoading/Texture2DCubic.h>

#include "TestSparseTexture.h"

using namespace demandLoading;

__global__ static void sparseTextureKernel( cudaTextureObject_t texture, float4* output, int width, int height, float lod )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= width || y >= height )
        return;

    float s = x / (float)width;
    float t = y / (float)height;

    bool   isResident = true;
#ifdef SPARSE_TEX_SUPPORT
    float4 pixel = tex2DLod<float4>( texture, s, t, lod, &isResident );
#else
    float4 pixel = tex2DLod<float4>( texture, s, t, lod );
#endif

    output[y * width + x] = isResident ? pixel : make_float4( -1.f, -1.f, -1.f, -1.f );
}

__host__ void launchSparseTextureKernel( cudaTextureObject_t texture, float4* output, int width, int height, float lod )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    sparseTextureKernel<<<dimGrid, dimBlock>>>( texture, output, width, height, lod );
    OTK_ERROR_CHECK( cudaGetLastError() );
}

__global__ static void wrapTestKernel( cudaTextureObject_t texture, float4* output, int width, int height, float lod )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= width || y >= height )
        return;

    // Test with s/t from [-1:2]
    float s = 3.f * x / (float)width - 1;
    float t = 3.f * y / (float)height - 1;

    bool isResident = true;
#ifdef SPARSE_TEX_SUPPORT
    float4 pixel = tex2DLod<float4>( texture, s, t, lod, &isResident );
#else
    float4 pixel = tex2DLod<float4>( texture, s, t, lod );
#endif

    output[y * width + x] = isResident ? pixel : make_float4( 1.f, 0.f, 1.f, 0.f );
}

__host__ void launchWrapTestKernel( cudaTextureObject_t texture, float4* output, int width, int height, float lod )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    wrapTestKernel<<<dimGrid, dimBlock>>>( texture, output, width, height, lod );
    OTK_ERROR_CHECK( cudaGetLastError() );
}


__global__ static void textureDrawKernel( demandLoading::DeviceContext context, unsigned int textureId,
                                          float4* output, int width, int height )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= width || y >= height )
        return;

    bool resident = true;
    float s = (x + 0.5f) / width;
    float t = (y + 0.5f) / height;
    float2 ddx = float2{ 1.0f / width, 0.0f };
    float2 ddy = float2{ 0.0f, 1.0f / height };
    float4 color = tex2DGrad<float4>( context, textureId, s, t, ddx, ddy, &resident );

    output[y * width + x] = resident ? color : make_float4( 1.f, 0.f, 1.f, 0.f );
}

__host__ void launchTextureDrawKernel( CUstream stream, demandLoading::DeviceContext& context, unsigned int textureId,
                                       float4* output, int width, int height )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    textureDrawKernel<<<dimGrid, dimBlock, 0U, stream>>>( context, textureId, output, width, height );
    OTK_ERROR_CHECK( cudaGetLastError() );
}
