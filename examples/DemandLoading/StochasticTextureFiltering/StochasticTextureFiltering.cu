// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>
#include <OptiXToolkit/ShaderUtil/ray_cone.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>
#include <OptiXToolkit/OTKAppBase/OTKAppSimpleBsdf.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>

#include "Texture2DFilters.h"
#include "StochasticTextureFilteringParams.h"

using namespace demandLoading;
using namespace otkApp;
using namespace otk;

OTK_DEVICE const float EPS = 0.0001f;
OTK_DEVICE const float INF = 1e16f;

extern "C" {
__constant__ OTKAppLaunchParams params;
}

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
// Utility functions
//------------------------------------------------------------------------------

// Sample a ColorTex
static __forceinline__ __device__ 
float3 sampleSurfaceValue( const DeviceContext context, SurfaceGeometry g, ColorTex tex, bool* is_resident, float2 xi )
{
    if( tex.texid < 0 )
        return tex.color;

    const StochasticTextureFilteringParams* stp = reinterpret_cast<StochasticTextureFilteringParams*>( params.extraData );

    // Choose between point and linear base filter
    int variantOffset = ( stp->textureFilterMode == fmPOINT ) ? 0 : 1;

    // Copy uv, ddx, ddy
    float2 uv = g.uv;
    float2 ddx = g.ddx;
    float2 ddy = g.ddy;

    // Exact multi-tap filters (cubic, lanczos, mitchell)
    if( stp->textureFilterMode == fmCUBIC )
    {
        // Mitchell(1,0,...) is bicubic upsampling
        float4 t = tex2DMitchell<float4>( 1.0f, 0.0f, context, tex.texid + variantOffset, uv.x, uv.y, is_resident );
        return tex.color * float3{t.x, t.y, t.z};
    }
    else if( stp->textureFilterMode == fmLANCZOS )
    {
        float4 t = tex2DLanczos<float4>( context, tex.texid + variantOffset, uv.x, uv.y, is_resident );
        return tex.color * float3{t.x, t.y, t.z};
    }
    else if( stp->textureFilterMode == fmMITCHELL )
    {
        float4 t = tex2DMitchell<float4>( 0.33333f, 0.33333f, context, tex.texid + variantOffset, uv.x, uv.y, is_resident );
        return tex.color * float3{t.x, t.y, t.z};
    }

    // Single tap jitter kernels (box, tent, gaussian, ewa0, extendAnisotropy)
    if( stp->textureJitterMode <= jmEXT_ANISOTROPY )
    {
        float2 texelJitter = float2{0.0f, 0.0f};

        if( stp->textureJitterMode == jmBOX )
        {
            texelJitter = stp->textureFilterWidth * boxFilter( xi );
        }
        else if( stp->textureJitterMode == jmTENT )
        {
            texelJitter = stp->textureFilterWidth * tentFilter( xi );
        }
        else if( stp->textureJitterMode == jmGAUSSIAN )
        {
            texelJitter = stp->textureFilterWidth * GAUSSIAN_STANDARD_WIDTH * boxMuller( xi );
        }
        else if( stp->textureJitterMode == jmEWA0 )
        {
            texelJitter = stp->textureFilterWidth * GAUSSIAN_STANDARD_WIDTH * boxMuller( xi );
            uv += jitterEWA( ddx, ddy, xi );
            ddx = float2{0.0f, 0.0f}; // force sampling mip level 0
            ddy = float2{0.0f, 0.0f};
        }
        else if( stp->textureJitterMode == jmEXT_ANISOTROPY )
        {
            texelJitter = stp->textureFilterWidth * GAUSSIAN_STANDARD_WIDTH * boxMuller( xi );
            uv += extendAnisotropy( ddx, ddy, xi );
        }

        float4 t = tex2DGradUdim<float4>( context, tex.texid + variantOffset, uv.x, uv.y, ddx, ddy, is_resident, texelJitter );
        return tex.color * float3{t.x, t.y, t.z};
    }

    // Two tap jitter kernels (unsharp mask, lanczos, mitchell, cylindrical mitchell)
    float2 posJitter = float2{0.0f, 0.0f};
    float2 negJitter = float2{0.0f, 0.0f};
    float blendWeight = 0.0f;

    if( stp->textureJitterMode == jmUNSHARPMASK )
    {
        const float posRadius = 0.5f;
        const float negRadius = 0.7f;
        const float unsharpStrength = 1.0f;

        float2 gjitter = stp->textureFilterWidth * boxMuller( xi );
        posJitter = gjitter * posRadius;
        negJitter = gjitter * negRadius;
        blendWeight = stp->textureFilterStrength * unsharpStrength;
    }
    else if( stp->textureJitterMode == jmLANCZOS )
    {
        if( variantOffset == 0 )
        {
            posJitter = stp->textureFilterWidth * sampleSharpenPos( LANCZOS_BOX, xi );
            negJitter = stp->textureFilterWidth * sampleSharpenNeg( LANCZOS_BOX, xi );
            blendWeight = stp->textureFilterStrength * LANCZOS_BOX_NWEIGHT;
        }
        else
        {
            posJitter = stp->textureFilterWidth * sampleSharpenPos( LANCZOS_TENT, xi );
            negJitter = stp->textureFilterWidth * sampleSharpenNeg( LANCZOS_TENT, xi );
            blendWeight = stp->textureFilterStrength * LANCZOS_TENT_NWEIGHT;
        }
    }
    else if( stp->textureJitterMode == jmMITCHELL )
    {
        if( variantOffset == 0 )
        {
            posJitter = stp->textureFilterWidth * sampleSharpenPos( MITCHELL_BOX, xi );
            negJitter = stp->textureFilterWidth * sampleSharpenNeg( MITCHELL_BOX, xi );
            blendWeight = stp->textureFilterStrength * MITCHELL_BOX_NWEIGHT;
        }
        else
        {
            posJitter = stp->textureFilterWidth * sampleSharpenPos( MITCHELL_TENT, xi );
            negJitter = stp->textureFilterWidth * sampleSharpenNeg( MITCHELL_TENT, xi );
            blendWeight = stp->textureFilterStrength * MITCHELL_TENT_NWEIGHT;
        }
    }
    else if( stp->textureJitterMode == jmCLANCZOS )
    {
        float2 cjitter = stp->textureFilterWidth * sampleCircle( xi.x );
        if( variantOffset == 0 )
        {
            posJitter = cjitter * stretchedCubic01( CLANCZOS_BOX_POS, xi.y );
            negJitter = cjitter * stretchedCubic01( CLANCZOS_BOX_NEG, xi.y );
            blendWeight = stp->textureFilterStrength * CLANCZOS_BOX_WEIGHT;
        }
        else
        {
            posJitter = cjitter * stretchedCubic01( CLANCZOS_TENT_POS, xi.y );
            negJitter = cjitter * stretchedCubic01( CLANCZOS_TENT_NEG, xi.y );
            blendWeight = stp->textureFilterStrength * CLANCZOS_TENT_WEIGHT;
        }
    }

    // Take positive and negative taps and combine them
    float4 gp = tex2DGradUdim<float4>( context, tex.texid + variantOffset, uv.x, uv.y, ddx, ddy, is_resident, posJitter );
    float4 gn = tex2DGradUdim<float4>( context, tex.texid + variantOffset, uv.x, uv.y, ddx, ddy, is_resident, negJitter );
    float4 t  = ( 1.0f + blendWeight ) * gp - ( blendWeight ) * gn;
    return tex.color * float3{t.x, t.y, t.z};
}

static __forceinline__ __device__
bool getSurfaceValues( SurfaceGeometry g, const SurfaceTexture* tex, float3& Ke, float3& Kd, float3& Ks, float3& Kt, float& roughness, float& ior, float2 xi )
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
        Ke = sampleSurfaceValue( params.demand_texture_context, g, tex->emission, &resident, xi );
        Kd = sampleSurfaceValue( params.demand_texture_context, g, tex->diffuse, &resident, xi ); 
        Ks = sampleSurfaceValue( params.demand_texture_context, g, tex->specular, &resident, xi ); 
        Kt = sampleSurfaceValue( params.demand_texture_context, g, tex->transmission, &resident, xi ); 
        roughness = tex->roughness + MIN_ROUGHNESS;
        ior = tex->ior;
        return resident;
    }
}

static __forceinline__ __device__ void makeEyeRay( uint2 px, float2 xi, float2 lx, float3& origin, float3& direction )
{
    // Jitter the pixel position based on the pixel filter mode
    const StochasticTextureFilteringParams* stp = reinterpret_cast<StochasticTextureFilteringParams*>( params.extraData );

    if( stp->pixelFilterMode == pfPIXELCENTER )
        xi = float2{0.5f, 0.5f};
    else if( stp->pixelFilterMode == pfBOX )
        ; //xi = xi;
    else if( stp->pixelFilterMode == pfGAUSSIAN )
        xi =  GAUSSIAN_STANDARD_WIDTH * boxMuller( xi ) + float2{0.5f, 0.5f};
    else // pixel center
        xi = tentFilter( xi ) + float2{0.5f, 0.5f};

    if( params.projection == ORTHOGRAPHIC )
        makeEyeRayOrthographic( params.camera, params.image_dim, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
    else if( params.projection == PINHOLE )
        makeEyeRayPinhole( params.camera, params.image_dim, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
    else // THINLENS
        makeCameraRayThinLens( params.camera, params.lens_width, params.image_dim, lx, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
}

static __forceinline__ __device__ float4 alternateOutputColor( OTKAppRayPayload payload )
{
    bool resident = false;
    SurfaceGeometry& geom = payload.geometry;
    const SurfaceTexture* tex = (const SurfaceTexture*)payload.material;

    if( params.render_mode == NORMALS )
        return 0.5f * float4{geom.N.x, geom.N.y, geom.N.z, 0.0f} + 0.5f * float4{1.0f, 1.0f, 1.0f, 0.0f};
    else if( params.render_mode == GEOMETRIC_NORMALS )
        return 0.5f * float4{geom.Ng.x, geom.Ng.y, geom.Ng.z, 0.0f} + 0.5f * float4{1.0f, 1.0f, 1.0f, 0.0f};
    else if( params.render_mode == TEX_COORDS )
        return float4{geom.uv.x, geom.uv.y, 0.0f, 0.0f};
    else if( params.render_mode == DDX_LENGTH )
        return float4{ 10.0f * length(geom.ddx), 10.0f * length(geom.ddy), 0.0f, 0.0f };
    else if( params.render_mode == CURVATURE )
        return float4{maxf(geom.curvature, 0.0f), maxf(-geom.curvature, 0.0f), 0.0f, 0.0f};
    else if( tex != nullptr ) // diffuse texture
        return make_float4( sampleTexture( params.demand_texture_context, geom, tex->diffuse, &resident ) );
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

    const int minDepth   = 0;
    const int maxDepth   = 1;

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
        if( payload.occluded == false )
        {
            const float4 bg = params.background_color;
            if( rayDepth >= minDepth )
                radiance += weight * float3{bg.x, bg.y, bg.z};
            break;
        }
        if( rayDepth >= maxDepth )
            break;

        float3 Ke, Kd, Ks, Kt;
        float roughness, ior;
        bool success = getSurfaceValues( payload.geometry, (SurfaceTexture*)payload.material, Ke, Kd, Ks, Kt, roughness, ior, float2{rnd(rseed), rnd(rseed)} );

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
        bool isTransmission = dot( R, payload.geometry.N ) < 0.0f;
        float fs = fabsf( bsdf.x + bsdf.y + bsdf.z );
        rayConeScatter( payload.geometry.curvature, ior, roughness, isTransmission, fs, payload.rayCone1 );
        rayConeScatter( payload.geometry.curvature, ior, roughness, isTransmission, fs, payload.rayCone2 );

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

    // Blend result of ray trace with inset display and put in result buffer
    accum_color *= ( 1.0f / accum_color.w );
    accum_color = float4{ maxf(accum_color.x, 0.0f), maxf(accum_color.y, 0.0f), maxf(accum_color.z, 0.0f), 1.0f};

    params.result_buffer[image_idx] = make_color( accum_color );
}
