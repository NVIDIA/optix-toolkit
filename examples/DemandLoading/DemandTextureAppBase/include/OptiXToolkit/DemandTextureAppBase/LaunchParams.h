//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
struct Params
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

struct TriangleHitGroupData
{
    SurfaceTexture tex;
    float4* vertices;
    float3* normals;
    float2* tex_coords;
};

} // namespace demandTextureApp
