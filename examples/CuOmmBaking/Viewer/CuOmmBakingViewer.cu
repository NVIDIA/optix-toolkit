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

#include "Procedural.h"
#include "LaunchParams.h"
#include "CuOmmBakingAppDeviceUtil.h"

#include <OptiXToolkit/ShaderUtil/vec_math.h>

using namespace ommBakingApp;
using namespace otk;  // for vec_math operators

//------------------------------------------------------------------------------
// Params - globally visible struct
//------------------------------------------------------------------------------

extern "C" {
__constant__ Params params;
}

//------------------------------------------------------------------------------
// Ray Payload - per ray data for OptiX programs
//------------------------------------------------------------------------------

struct RayPayload
{
    float3 color; // return color
    float  alpha; // return alpha
};

//------------------------------------------------------------------------------
// OptiX programs
//------------------------------------------------------------------------------
 
extern "C" __global__ void __raygen__rg()
{
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_width, params.image_height ) )
        return;

    // Ray for an orthographic view facing in the -z direction
    float3 origin;
    origin.x = params.eye.x + params.view_dims.x * (0.5f + px.x - 0.5f * params.image_width) / (float)params.image_width;
    origin.y = params.eye.y + params.view_dims.y * (0.5f + px.y - 0.5f * params.image_height) / ( float )params.image_height;
    origin.z = params.eye.z;
    float3 direction = make_float3( 0.0f, 0.0f, -1.0f );

    // Ray payload
    RayPayload payload;
    payload.color = make_float3( 0.f, 0.f, 0.f );
    payload.alpha = 1.f;
    
    // Trace the ray
    float tmin = 0.0f;
    float tmax = 1e16f;
    float time = 0.0f;
    traceRay( params.traversable_handle, RAY_TYPE_RADIANCE, origin, direction, tmin, tmax, time, &payload );

    float4 color = make_float4( payload.color.x, payload.color.y, payload.color.z, payload.alpha );

    // Put the final color in the result buffer
    params.result_buffer[px.y * params.image_width + px.x] = make_color( color );
}

extern "C" __global__ void __miss__ms()
{
    MissData* missData = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );

    RayPayload* payload = getRayPayload();
    float3 color = payload->color;
    float alpha  = payload->alpha;

    // Mix color with background
    payload->color = color + alpha * missData->background_color;
}

__device__ float4 eval()
{
    // Evaluate the texture at the current intersection

    HitGroupData* hitData = reinterpret_cast< HitGroupData* >( optixGetSbtDataPointer() );

    float2 uv = optixGetTriangleBarycentrics();

    uint3 idx3 = hitData->indices[optixGetPrimitiveIndex()];
    float2 uv0 = hitData->texCoords[idx3.x];
    float2 uv1 = hitData->texCoords[idx3.y];
    float2 uv2 = hitData->texCoords[idx3.z];

    uv = uv0 + uv.x * ( uv1 - uv0 ) + uv.y * ( uv2 - uv0 );

    float4 color;
    if( hitData->texture_id )
    {
        color = tex2D<float4>( hitData->texture_id, uv.x, uv.y );
    }
    else
    {
        float alpha = eval_procedural<float>( { uv.x, uv.y } );
        color = { alpha, alpha, alpha, alpha };
    }

    return color;
}

extern "C" __global__ void __anyhit__ah()
{
    HitGroupData* hitData = reinterpret_cast< HitGroupData* >( optixGetSbtDataPointer() );

    float4 color = eval();

    RayPayload* payload = getRayPayload();
    if( params.visualize_omm )
    {
        payload->color = make_float3( 1.0f, 0.f, 0.4f );
        payload->alpha = 0.f;
    }
    else
    {
        payload->color += payload->alpha * color.w * make_float3( color.x, color.y, color.z );
        payload->alpha *= ( 1 - color.w );
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__ch()
{
    float4 color = eval();

    RayPayload* payload = getRayPayload();
    payload->color = make_float3( color.x, color.y, color.z );
    payload->alpha = 0.f;
}
