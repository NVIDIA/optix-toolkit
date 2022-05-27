//
//  Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include "TestSparseTexture.h"

#include "Util/Exception.h"

__global__ static void sparseTextureKernel( cudaTextureObject_t texture, float4* output, int width, int height, float lod )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= width || y >= height )
        return;

    float s = x / (float)width;
    float t = y / (float)height;

    bool   isResident = false;
    float4 pixel      = tex2DLod<float4>( texture, s, t, lod, &isResident );

    output[y * width + x] = isResident ? pixel : make_float4( -1.f, -1.f, -1.f, -1.f );
}

__host__ void launchSparseTextureKernel( cudaTextureObject_t texture, float4* output, int width, int height, float lod )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    sparseTextureKernel<<<dimGrid, dimBlock>>>( texture, output, width, height, lod );
    DEMAND_CUDA_CHECK( cudaGetLastError() );
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

    bool   isResident = false;
    float4 pixel      = tex2DLod<float4>( texture, s, t, lod, &isResident );

    output[y * width + x] = isResident ? pixel : make_float4( 1.f, 0.f, 1.f, 0.f );
}

__host__ void launchWrapTestKernel( cudaTextureObject_t texture, float4* output, int width, int height, float lod )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    wrapTestKernel<<<dimGrid, dimBlock>>>( texture, output, width, height, lod );
    DEMAND_CUDA_CHECK( cudaGetLastError() );
}
