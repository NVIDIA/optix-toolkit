// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>

using namespace demandLoading;
using namespace otkApp;
using namespace otk;

//------------------------------------------------------------------------------
// Params - globally visible struct
//------------------------------------------------------------------------------

extern "C" {
__constant__ OTKAppLaunchParams params;
}

//------------------------------------------------------------------------------
// Ray Payload - per ray data for OptiX programs
//------------------------------------------------------------------------------

struct RayPayload
{
    float4 color;     // return color
    RayCone rayCone;
};

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

    // Blend result of ray trace with tile display color
    float4 tcolor = tileDisplayColor( params.demand_texture_context, params.display_texture_id, 10, 10, px );
    float4 color  = ( 1.0f - tcolor.w ) * payload.color + tcolor.w * tcolor;

    // Put the final color in the result buffer
    params.result_buffer[px.y * params.image_dim.x + px.x] = make_color( color );
}

extern "C" __global__ void __miss__ms()
{
    // Copy miss color to ray payload
    //OTKAppMissData* missData = reinterpret_cast<OTKAppMissData*>( optixGetSbtDataPointer() );
    getRayPayload()->color = params.background_color;
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
    OTKAppHitGroupData* hitData   = reinterpret_cast<OTKAppHitGroupData*>( optixGetSbtDataPointer() );
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
    if( params.interactive_mode )
        payload->color = tex2DGradWalkup<float4>( params.demand_texture_context, textureId, uv.x, uv.y, ddx, ddy, &resident );
    else 
        payload->color = tex2DGradUdimBlend<float4>( params.demand_texture_context, textureId, uv.x, uv.y, ddx, ddy, &resident );

}
