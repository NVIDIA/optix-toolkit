// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <OptiXToolkit/DemandLoading/DeviceContext.h>

namespace otkApp
{

const unsigned int EXTRA_DATA_SIZE = 8;

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};

enum Projection
{
    ORTHOGRAPHIC = 0,
    PINHOLE,
    THINLENS,
    NUM_PROJECTIONS
};

struct CameraFrame
{
    float3 eye;
    float3 U;
    float3 V;
    float3 W;
};

// OptiX launch params
struct OTKAppLaunchParams
{
    // Device-related data for multi-gpu rendering
    unsigned int device_idx;
    unsigned int num_devices;

    // Render buffer
    float4* accum_buffer;
    uchar4* result_buffer;
    uint2   image_dim;

    // Camera params
    CameraFrame camera;
    float lens_width;
    Projection projection;

    // Handle to scene bvh for ray traversal
    OptixTraversableHandle traversable_handle;

    // Demand texture context
    demandLoading::DeviceContext demand_texture_context;
    int  display_texture_id;  // which texture to show resident tiles

    unsigned int render_mode; // how to render the scene
    bool interactive_mode;    // whether the application is running in interactive mode

    // Extra params that can be used by the application
    float4 c[EXTRA_DATA_SIZE];
    float  f[EXTRA_DATA_SIZE];
    int    i[EXTRA_DATA_SIZE];
};


struct OTKAppRayGenData
{
    // Handled by params
};

struct OTKAppMissData
{
    // Handled by raygen
};

struct OTKAppHitGroupData
{
    unsigned int texture_id;
};

struct ColorTex
{
    float3 color;
    int texid;
};

struct SurfaceTexture
{
    ColorTex diffuse;
    ColorTex specular;
    ColorTex transmission;
    ColorTex emission;
    float roughness;
    float ior;
};

struct SurfaceGeometry
{
    float3 P, Ng;        // intersection point and geometric normal
    float3 N, S, T;      // shading normal and basis
    float2 uv, ddx, ddy; // texture coordinates and derivatives
    float  curvature;    // Surface curvature
    bool flipped;        // Whether the normal was flipped 
};

struct OTKAppTriangleHitGroupData
{
    SurfaceTexture tex;
    float4* vertices;
    float3* normals;
    float2* tex_coords;
};

} // namespace demandTextureApp
