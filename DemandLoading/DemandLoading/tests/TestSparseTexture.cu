//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


__global__ static void cubicTextureDrawKernel( demandLoading::DeviceContext context, unsigned int textureId,
                                          float4* output, int width, int height, float2 ddx, float2 ddy )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= width || y >= height )
        return;

    bool resident = true;
    float s = (x + 0.5f) / width;
    float t = (y + 0.5f) / height;
    float4 color;
    resident = textureUdim<float4>( context, textureId, s, t, ddx, ddy, &color );

    output[y * width + x] = resident ? color : make_float4( 1.f, 0.f, 1.f, 0.f );
}

__host__ void launchCubicTextureDrawKernel( CUstream stream, demandLoading::DeviceContext& context, unsigned int textureId,
                                            float4* output, int width, int height, float2 ddx, float2 ddy )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    cubicTextureDrawKernel<<<dimGrid, dimBlock, 0U, stream>>>( context, textureId, output, width, height, ddx, ddy );
    OTK_ERROR_CHECK( cudaGetLastError() );
}
