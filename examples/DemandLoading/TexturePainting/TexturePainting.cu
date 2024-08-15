// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/DemandTextureAppBase/LaunchParams.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureAppDeviceUtil.h>

#include "TexturePaintingParams.h"

using namespace demandLoading;
using namespace demandTextureApp;
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
    float4 color; // return color
    RayCone rayCone;
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
    float d2 = powf(dx, exp) + powf(dy, exp);
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
    const int cx = params.image_dim.x - BUTTON_SPACING / 2 - params.i[BRUSH_WIDTH_ID] / 2;
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
    if( !pixelInBounds( px, params.image_dim ) )
        return;

    // Eye ray
    float3 origin, direction;
    makeEyeRayOrthographic( params.camera, params.image_dim, float2{px.x+0.5f, px.y+0.5f}, origin, direction );

    // Ray payload with ray cone for orthographic view
    RayPayload payload;
    payload.color = make_float4( 0.0f );
    payload.rayCone = initRayConeOrthoCamera( params.camera.U, params.camera.V, params.image_dim );

    // Trace the ray
    float tmin = 0.0f;
    float tmax = 1e16f;
    traceRay( params.traversable_handle, origin, direction, tmin, tmax, OPTIX_RAY_FLAG_NONE, &payload );

    // Mix the texture color with the overlay color
    float4 ocolor = overlayColor( px );
    float4 color  = ( 1.0f - ocolor.w ) * payload.color + ocolor.w * ocolor;
    params.result_buffer[px.y * params.image_dim.x + px.x] = make_color( color );
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
    payload->rayCone = propagate( payload->rayCone, rayDistance );
    
    // Get the texture footprint to sample from the cone width
    float footprintWidth = texFootprintWidth( payload->rayCone.width, dPds_len, dPdt_len );
    float2 ddx = make_float2( footprintWidth, 0.0f );
    float2 ddy = make_float2( 0.0f, footprintWidth );

    // Sample the texture, and put the value in the ray payload.
    bool resident  = false;
    payload->color = tex2DGrad<float4>( params.demand_texture_context, textureId, uv.x, uv.y, ddx, ddy, &resident );
    if( !resident )
        payload->color = make_float4( 1.0f, 1.0f, 1.0f, 0.0f );
}
