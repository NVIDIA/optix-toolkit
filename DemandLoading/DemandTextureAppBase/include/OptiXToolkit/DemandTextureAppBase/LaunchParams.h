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

#include <cuda_runtime.h>
#include <optix.h>
#include <OptiXToolkit/DemandLoading/DeviceContext.h>

namespace demandTextureApp
{

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};

// OptiX launch params
struct Params
{
    // Device-related data for multi-gpu rendering
    unsigned int device_idx;
    unsigned int num_devices;

    // Render buffer
    uchar4*      result_buffer;
    unsigned int image_width;
    unsigned int image_height;

    // Orthographic view 
    float3 eye;
    float2 view_dims;

    // Handle to scene bvh for ray traversal
    OptixTraversableHandle traversable_handle;

    // Demand texture context
    demandLoading::DeviceContext demand_texture_context;

    int  display_texture_id;  // which texture to show resident tiles
    bool interactive_mode;    // whether the application is running in interactive mode
};


struct RayGenData
{
    // Handled by params
};

struct MissData
{
    float4 background_color;
};

struct HitGroupData
{
    unsigned int texture_id;
};

} // namespace demandTextureApp
