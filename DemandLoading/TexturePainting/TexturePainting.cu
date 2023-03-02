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

#include <OptiXToolkit/DemandTextureAppBase/LaunchParams.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureAppDeviceUtil.h>

#include "TexturePaintingParams.h"

using namespace demandLoading;
using namespace demandTextureApp;

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
    float4 color; // return color
    float  cone_width; 
    float  cone_angle;
};

//------------------------------------------------------------------------------
// Drawing overlay
//------------------------------------------------------------------------------

__device__ __forceinline__ float4 drawSuperellipse( uint2 px, int cx, int cy, int width, int height, float4 color, int exp )
{
    float dx = fabs( cx - static_cast<float>(px.x) ) / (width * 0.5f);
    float dy = fabs( cy - static_cast<float>(px.y) ) / (height * 0.5f);
    if( dx > 1.0f || dy > 1.0f )
        return float4{0.0f, 0.0f, 0.0f, 0.0f};
    float d2 = pow(dx, exp) + pow(dy, exp);
    if( d2 > 1.0f )
        return float4{0.0f, 0.0f, 0.0f, 0.0f};
    float4 rval = color;
    rval.w *= ( 1.0f - d2 * d2 );
    return rval;
}

__device__ __forceinline__ float4 overlayColor( uint2 px )
{
    // Draw buttons
    float4 rval = float4{0.0f, 0.0f, 0.0f, 0.0f};
    for( int i = 0; i < params.i[NUM_CANVASES_ID]; ++i )
    {
        const int cx = BUTTON_SPACING * (i+1) + BUTTON_SIZE * i + BUTTON_SIZE / 2;
        const int cy = BUTTON_SPACING + BUTTON_SIZE / 2;
        const int bsize = ( i == params.i[ACTIVE_CANVAS_ID] ) ? BUTTON_SIZE + 4 : BUTTON_SIZE;
        const float4 bcolor = ( i == params.i[ACTIVE_CANVAS_ID] ) ? float4{ 0.3f, 0.0f, 0.0f, 1.0f } : float4{ 0.0f, 0.0f, 0.3f, 1.0f };
        rval += drawSuperellipse( px, cx, cy, bsize, bsize, bcolor, 6 );
    }

    // Draw brush
    const int cx = params.image_width - BUTTON_SPACING / 2 - params.i[BRUSH_WIDTH_ID] / 2;
    const int cy = BUTTON_SPACING / 2 + params.i[BRUSH_HEIGHT_ID] / 2; 
    rval += drawSuperellipse( px, cx, cy, params.i[BRUSH_WIDTH_ID], params.i[BRUSH_HEIGHT_ID], params.c[BRUSH_COLOR_ID], 2 );
    
    return rval;
}

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
    origin.x = params.eye.x + params.U.x * (0.5f + px.x - 0.5f * params.image_width) / params.image_width;
    origin.y = params.eye.y + params.V.y * (0.5f + px.y - 0.5f * params.image_height) / params.image_height;
    origin.z = params.eye.z;
    float3 direction = make_float3( 0.0f, 0.0f, -1.0f );

    // Ray payload with ray cone for orthographic view
    RayPayload payload;
    payload.color = make_float4( 0.0f );
    payload.cone_width = minf( params.U.x / params.image_width, params.V.y / params.image_height );
    payload.cone_angle = 0.0f;
    
    // Trace the ray
    float tmin = 0.0f;
    float tmax = 1e16f;
    float time = 0.0f;
    traceRay( params.traversable_handle, RAY_TYPE_RADIANCE, origin, direction, tmin, tmax, time, &payload );

    // Mix the texture color with the overlay color
    float4 ocolor = overlayColor( px );
    float4 color  = ( 1.0f - ocolor.w ) * payload.color + ocolor.w * ocolor;
    params.result_buffer[px.y * params.image_width + px.x] = make_color( color );
}

extern "C" __global__ void __miss__ms()
{
    // Copy miss color to ray payload
    MissData* missData = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    getRayPayload()->color = missData->background_color;
}

extern "C" __global__ void __intersection__is()
{
    const float3 origin    = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();

    // Intersect ray with unit square on xy plane
    float  t = -origin.z / direction.z;
    float3 p = origin + t * direction;
    float3 n = make_float3( 0.0f, 0.0f, 1.0f );

    if( t > optixGetRayTmin() && t < optixGetRayTmax() && p.x >= 0.0f && p.x <= 1.0f && p.y >= 0.0f && p.y <= 1.0f )
        optixReportIntersection( t, 0, float3_as_uints( p ), float3_as_uints( n ) );
}

extern "C" __global__ void __closesthit__ch()
{
    // The hit group data has the demand texture id.
    HitGroupData* hitData   = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    unsigned int  textureId = hitData->texture_id;

    // The hit object is a unit square, so the texture coord is the same as the hit point.
    float2 uv = make_float2( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ) );
    
    // The world space texture derivatives for a unit square are just dPds=(1,0,0) and dPdt=(0,1,0). 
    float dPds_len = 1.0f;
    float dPdt_len = 1.0f;

    // Get the world space ray cone width at the intersection point
    RayPayload* payload = getRayPayload();
    float rayDistance = optixGetRayTmax();
    float coneWidth = propagateRayCone( payload->cone_width, payload->cone_angle, rayDistance );
    
    // Get the texture footprint to sample from the cone width
    float footprintWidth = computeTextureFootprintMinWidth( dPds_len, dPdt_len, coneWidth );
    float2 ddx = make_float2( footprintWidth, 0.0f );
    float2 ddy = make_float2( 0.0f, footprintWidth );

    // Sample the texture, and put the value in the ray payload.
    bool resident  = false;
    payload->color = tex2DGrad<float4>( params.demand_texture_context, textureId, uv.x, uv.y, ddx, ddy, &resident );
    if( !resident )
        payload->color = make_float4( 1.0f, 1.0f, 1.0f, 0.0f );
}
