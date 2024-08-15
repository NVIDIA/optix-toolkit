// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>

#include <OptiXToolkit/DemandLoading/DeviceContext.h>

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
