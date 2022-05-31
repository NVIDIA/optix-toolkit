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

#include <optix.h>

#include <DemandLoading/DeviceContext.h>

#include <cuda_runtime.h>

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};

struct Params
{
    // Render buffer
    uchar4*      result_buffer;
    unsigned int image_width;
    unsigned int image_height;

    // Device-related data for multi-gpu rendering
    unsigned int device_idx;
    unsigned int num_devices;
    int          origin_x;
    int          origin_y;

    // Handle to scene description for ray traversal
    OptixTraversableHandle handle;

    // Camera parameters
    float3 eye;
    float3 U;
    float3 V;
    float3 W;

    // Bucket parameters (for tiled rendering)
    unsigned int bucket_index;
    unsigned int bucket_width;
    unsigned int bucket_height;

    // Texture data
    float                        mipLevelBias;
    demandLoading::DeviceContext demandTextureContext;
    cudaMipmappedArray_t         nonDemandTextureArray;
    cudaTextureObject_t          nonDemandTexture;

    // Render mode
    float diffScale;
    int   numTextureTaps;
};


struct RayGenData
{
    // Empty
};


struct MissData
{
    // Background color
    float r, g, b;
};


struct HitGroupData
{
    float        radius;
    unsigned int demand_texture_id;
    float        texture_scale;
    float        texture_lod;
};
