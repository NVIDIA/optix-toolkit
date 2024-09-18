// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>
#include <OptiXToolkit/OTKAppBase/OTKAppSimpleBsdf.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>

#include "RayConesParams.h"

using namespace demandLoading;
using namespace otkApp;
using namespace otk;

OTK_DEVICE const float EPS = 0.0001f;
OTK_DEVICE const float INF = 1e16f;

//------------------------------------------------------------------------------
// Render modes
//------------------------------------------------------------------------------

enum RenderModes
{
    FULL_RENDER = 1,
    NORMALS,
    GEOMETRIC_NORMALS,
    TEX_COORDS,
    DDX_LENGTH, 
    CURVATURE,
    DIFFUSE_TEXTURE
};

//------------------------------------------------------------------------------
// Params - globally visible struct
//------------------------------------------------------------------------------

extern "C" {
__constant__ OTKAppLaunchParams params;
}

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------
 
static __forceinline__ __device__
bool getSurfaceValues( SurfaceGeometry g, const SurfaceTexture* tex, float3& Ke, float3& Kd, float3& Ks, float3& Kt, float& roughness, float& ior )
{
    if( tex == nullptr )
    {
        Ke = Ks = Kt = float3{0.0f, 0.0f, 0.0f};
        Kd = float3{1.0f, 0.0f, 1.0f};
        roughness = 0.0f;
        ior = 0.0f;
        return false;
    }
    else
    {
        bool resident = false;
        Ke = sampleTexture( params.demand_texture_context, g, tex->emission, &resident );
        Kd = sampleTexture( params.demand_texture_context, g, tex->diffuse, &resident ); 
        Ks = sampleTexture( params.demand_texture_context, g, tex->specular, &resident ); 
        Kt = sampleTexture( params.demand_texture_context, g, tex->transmission, &resident ); 
        roughness = tex->roughness + MIN_ROUGHNESS;
        ior = tex->ior;
        return resident;
    }
}

static __forceinline__ __device__ void makeEyeRay( uint2 px, float2 xi, float2 lx, float3& origin, float3& direction )
{
    xi = tentFilter( xi ) + float2{0.5f, 0.5f};
    if( params.projection == ORTHOGRAPHIC )
        makeEyeRayOrthographic( params.camera, params.image_dim, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
    else if( params.projection == PINHOLE )
        makeEyeRayPinhole( params.camera, params.image_dim, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
    else // THINLENS
        makeCameraRayThinLens( params.camera, params.lens_width, params.image_dim, lx, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
}

static __forceinline__ __device__ float4 alternateOutputColor( OTKAppRayPayload& payload )
{
    bool resident = false;
    SurfaceGeometry& g = payload.geometry;
    const SurfaceTexture* tex = (SurfaceTexture*)payload.material;

    if( params.render_mode == NORMALS )
        return 0.5f * float4{g.N.x, g.N.y, g.N.z, 0.0f} + 0.5f * float4{1.0f, 1.0f, 1.0f, 0.0f};
    else if( params.render_mode == GEOMETRIC_NORMALS )
        return 0.5f * float4{g.Ng.x, g.Ng.y, g.Ng.z, 0.0f} + 0.5f * float4{1.0f, 1.0f, 1.0f, 0.0f};
    else if( params.render_mode == TEX_COORDS )
        return float4{g.uv.x, g.uv.y, 0.0f, 0.0f};
    else if( params.render_mode == DDX_LENGTH )
        return float4{ 10.0f * length(g.ddx), 10.0f * length(g.ddy), 0.0f, 0.0f };
    else if( params.render_mode == CURVATURE )
        return float4{maxf(g.curvature, 0.0f), maxf(-g.curvature, 0.0f), 0.0f, 0.0f};
    else if( tex != nullptr ) // diffuse texture
        return make_float4( sampleTexture( params.demand_texture_context, g, tex->diffuse, &resident ) );
    return float4{0.0f, 0.0f, 0.0f, 0.0f};
}

//------------------------------------------------------------------------------
// Ray cone functions
//------------------------------------------------------------------------------

static __forceinline__ __device__ 
void initRayCones( CameraFrame camera, uint2 image_dim, unsigned int projection, float lens_width, float3 direction, 
                   RayCone& rayCone1, RayCone& rayCone2 )
{
    if( projection == ORTHOGRAPHIC )
    {
        rayCone1 = initRayConeOrthoCamera( camera.U, camera.V, image_dim );
        rayCone2 = initRayConeThinLensCamera( camera.W, length(camera.U) / image_dim.x, direction );
    }
    else if( projection == PINHOLE )
    {
        rayCone1 = initRayConePinholeCamera( camera.U, camera.V, camera.W, image_dim, direction );
        rayCone2 = initRayConeThinLensCamera( camera.W, length(camera.U) / image_dim.x, direction );
    } 
    else // THINLENS
    {
        rayCone1 = initRayConeThinLensCamera( camera.W, lens_width, direction );
        rayCone2 = initRayConePinholeCamera( camera.U, camera.V, camera.W, image_dim, direction );
    }
}

static __forceinline__ __device__ 
void rayConeScatter( float curvature, float ior, float roughness, bool isTransmission, float fs, RayCone& rayCone )
{
    if( isTransmission )
        rayCone = refract( rayCone, curvature, 1.0f, ior );
    else
        rayCone = reflect( rayCone, curvature );
    rayCone = scatterBsdf( rayCone, fs );
}

//------------------------------------------------------------------------------
// OptiX programs
//------------------------------------------------------------------------------
 
extern "C" __global__ void __raygen__rg()
{
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) )
        return;

    const RayConesParams* rcParams = reinterpret_cast<RayConesParams*>( params.extraData );

    // Ray and payload definition
    float3 origin, direction;
    OTKAppRayPayload payload{};
   
    // Handle alternate output values
    if( params.render_mode != FULL_RENDER )
    {
        makeEyeRay( px, float2{0.5f, 0.5f}, float2{0.5f, 0.5f}, origin, direction );
        initRayCones( params.camera, params.image_dim, params.projection, params.lens_width, direction, 
                      payload.rayCone1, payload.rayCone2 );
        traceRay( params.traversable_handle, origin, direction, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
        float4 color = alternateOutputColor( payload );
        params.result_buffer[px.y * params.image_dim.x + px.x] = make_color( color );
        return;
    }

    unsigned int rseed = srand( px.x, px.y, params.subframe );
    float3 radiance = float3{0.0f, 0.0f, 0.0f};

    makeEyeRay( px, float2{rnd(rseed), rnd(rseed)}, float2{rnd(rseed), rnd(rseed)}, origin, direction );
    initRayCones( params.camera, params.image_dim, params.projection, params.lens_width, direction, 
                  payload.rayCone1, payload.rayCone2 );
    
    float3 weight = float3{1.0f, 1.0f, 1.0f};
    int rayDepth = 0;

    while( true )
    {
        payload.rayDepth = rayDepth;
        payload.material = nullptr;
        traceRay( params.traversable_handle, origin, direction, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
        payload.geometry.ddx *= rcParams->mipScale;
        payload.geometry.ddy *= rcParams->mipScale;
        if( payload.occluded == false )
        {
            const float4& bg = params.background_color;
            if( rayDepth >= rcParams->minRayDepth )
                radiance += weight * float3{bg.x, bg.y, bg.z};
            break;
        }
        if( rayDepth >= rcParams->maxRayDepth )
            break;

        float3 Ke, Kd, Ks, Kt;
        float roughness, ior;
        bool success = getSurfaceValues( payload.geometry, (SurfaceTexture*)payload.material, Ke, Kd, Ks, Kt, roughness, ior );
        //if( rayDepth > 3 ) payload.rayCone1.setDiffuse(); // clamp cone angle for deeper paths
        //if( rayDepth > 3 ) payload.rayCone2.setDiffuse(); // clamp cone angle for deeper paths
        radiance += weight * Ke;
        if( payload.geometry.flipped && ior != 0.0f )
            ior = 1.0f / ior;

        float3 R, bsdf;
        float prob;

        // Sample the BSDF
        float2 xi = float2{ rnd(rseed), rnd(rseed) };
        if( !sampleBsdf( xi, Kd, Ks, Kt, ior, roughness, payload.geometry, direction, R ) )
            break;
        bsdf = evalBsdf( Kd, Ks, Kt, ior, roughness, payload.geometry, direction, R, prob, payload.geometry.curvature );
        
        // Sanity check.  Discard 0 probability and nan samples
        if( !( prob > 0.0f ) || !( dot( bsdf, bsdf ) >= 0.0f ) )
            break;

        // Update the ray cone
        if( rcParams->updateRayCones )
        {
            bool isTransmission = dot( R, payload.geometry.N ) < 0.0f;
            float fs = bsdf.x + bsdf.y + bsdf.z;
            rayConeScatter( payload.geometry.curvature, ior, roughness, isTransmission, fs, payload.rayCone1 );
            rayConeScatter( payload.geometry.curvature, ior, roughness, isTransmission, fs, payload.rayCone2 );
        }

        weight = weight * bsdf * (1.0f / prob);
        rayDepth++;
        origin = payload.geometry.P;
        direction = R;
    }

    // Accumulate sample in the accumulation buffer
    unsigned int image_idx = px.y * params.image_dim.x + px.x;
    float4 accum_color = float4{radiance.x, radiance.y, radiance.z, 1.0f};
    if( params.subframe != 0 )
        accum_color += params.accum_buffer[image_idx];
    params.accum_buffer[image_idx] = accum_color;

    // Blend result of ray trace with tile display and put in result buffer
    accum_color *= ( 1.0f / accum_color.w );
    float4 tcolor = tileDisplayColor( params.demand_texture_context, params.display_texture_id, 10, 10, px );
    float4 color  = ( 1.0f - tcolor.w ) * accum_color + tcolor.w * tcolor;
    params.result_buffer[image_idx] = make_color( color );
}

