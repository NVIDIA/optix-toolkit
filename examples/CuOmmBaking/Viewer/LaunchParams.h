// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace ommBakingApp
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

    // color the unknown area
    bool visualize_omm;
};


struct RayGenData
{
    // Handled by params
};

struct MissData
{
    float3 background_color;
};

struct HitGroupData
{
    cudaTextureObject_t texture_id;
    const uint3*        indices;
    const float2*       texCoords;
};

} // namespace ommBakingApp
