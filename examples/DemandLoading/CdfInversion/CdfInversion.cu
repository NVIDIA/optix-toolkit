// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/AliasTable.h>
#include <OptiXToolkit/ShaderUtil/CdfInversionTable.h>
#include <OptiXToolkit/ShaderUtil/PdfTable.h>
#include <OptiXToolkit/ShaderUtil/ISummedAreaTable.h>
#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/Reservoir.h>
#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>
#include <OptiXToolkit/OTKAppBase/OTKAppSimpleBsdf.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>

#include "CdfInversionParams.h"

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
    EMAP_RENDER = 1,
    BSDF_RENDER,
    MIS_RENDER,
    RIS_RENDER
};

//------------------------------------------------------------------------------
// Params - globally visible struct
//------------------------------------------------------------------------------

extern "C" {
__constant__ OTKAppLaunchParams params;
}

struct Ray
{
    float3 o, d;
};

//------------------------------------------------------------------------------
// Surface values
//------------------------------------------------------------------------------

struct SurfaceValues
{
    float3 Kd;
    float3 Ks;
    float3 Kt;
    float3 Ke;
    float roughness;
    float ior;
};

static __forceinline__ __device__
bool getSurfaceValues( SurfaceGeometry g, const SurfaceTexture* tex, SurfaceValues& sv )
{
    if( tex == nullptr )
    {
        sv = SurfaceValues{};
        sv.Kd = float3{1.0f, 0.0f, 1.0f};
        return false;
    }
    
    bool resident = false;
    sv.Ke = sampleTexture( params.demand_texture_context, g, tex->emission, &resident );
    sv.Kd = sampleTexture( params.demand_texture_context, g, tex->diffuse, &resident ); 
    sv.Ks = sampleTexture( params.demand_texture_context, g, tex->specular, &resident ); 
    sv.Kt = sampleTexture( params.demand_texture_context, g, tex->transmission, &resident ); 
    sv.roughness = tex->roughness + MIN_ROUGHNESS;
    sv.ior = tex->ior;
    return resident;
}

//------------------------------------------------------------------------------
// Rays and ray cones
//------------------------------------------------------------------------------

static __forceinline__ __device__
void makeEyeRay( uint2 px, float3& origin, float3& direction, RayCone& rayCone, float2 xi )
{
    xi = tentFilter( xi ) + float2{0.5f, 0.5f};
    makeEyeRayPinhole( params.camera, params.image_dim, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
    rayCone = initRayConePinholeCamera( params.camera.U, params.camera.V, params.camera.W, params.image_dim, direction );
}

static __forceinline__ __device__ 
void rayConeReflect( float curvature, float fs, RayCone& rayCone )
{
    rayCone = reflect( rayCone, curvature );
    rayCone = scatterBsdf( rayCone, fs );
}

//------------------------------------------------------------------------------
// Environment map functions
//------------------------------------------------------------------------------

static __forceinline__ __device__ float2 vecToUv( float3 v ) 
{
	return float2{ 0.5f + ( 0.5f * M_1_PIf ) * ( atan2f( v.y, v.x ) ), M_1_PIf * acosf( v.z )};
}

static __forceinline__ __device__ float3 uvToVec( float2 uv ) 
{
	float phi = M_PIf + 2.0f * M_PIf * uv.x;
    float theta = M_PIf * uv.y;
	return float3{cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta)};
}

//------------------------------------------------------------------------------
// Sampling
//------------------------------------------------------------------------------

static __forceinline__ __device__
bool sampleBsdf( SurfaceGeometry& geom, SurfaceValues& sv, float3 inDirection, float2 xi, float3& outDirection )
{
    return sampleBsdf( xi, sv.Kd, sv.Ks, sv.Kt, sv.ior, sv.roughness, geom, inDirection, outDirection );
}

static __forceinline__ __device__
void evalBsdf( SurfaceGeometry& geom, SurfaceValues& sv, float3 inDirection, float3 outDirection, float3& bsdfVal, float& prob)
{
    bsdfVal = evalBsdf( sv.Kd, sv.Ks, sv.Kt, sv.ior, sv.roughness, geom, inDirection, outDirection, prob, geom.curvature );
    if( !( prob > 0.0f ) )
        prob = 0.0f;
}

static __forceinline__ __device__
float2 sampleEmap( float2 xi )
{
    CdfInversionParams* cdfParams = reinterpret_cast<CdfInversionParams*>( params.extraData );

    #ifdef emBIN_SEARCH
        return sampleCdfBinSearch( cdfParams->emapCdfInversionTable, xi );
    #elif emLIN_SEARCH
        return sampleCdfLinSearch( cdfParams->emapCdfInversionTable, xi );
    #elif emDIRECT_LOOKUP
        return sampleCdfDirectLookup( cdfParams->emapCdfInversionTable, xi );
    #elif emALIAS_TABLE
        CdfInversionTable* it = cdfParams->emapCdfInversionTable;
        return alias2D( cdfParams->emapAliasTable, it->width, it->height, xi );
    #elif emSUMMED_AREA_TABLE
        return sample( cdfParams->emapSummedAraeTable, xi );
    #endif
}

static __forceinline__ __device__
float3 evalEmap( float2 uv, float rayConeAngle, bool* isResident, float2 xi )
{
    // Switch using ray cone and mip level 0
    const CdfInversionParams* cdfParams = reinterpret_cast<CdfInversionParams*>( params.extraData );
    if( cdfParams->useMipLevelZero )
    {
        xi = float2{0.5f, 0.5f};
        rayConeAngle = 0.0f;
    }

    unsigned int texId = cdfParams->emapTextureId;
    float2 ddx = float2{rayConeAngle * 0.5f * M_1_PIf, 0.0f};
    float2 ddy = float2{0.0f, rayConeAngle * M_1_PIf};
    float2 texelJitter = tentFilter( xi ); // stochastic cubic filter
    float4 f = tex2DGrad<float4>( params.demand_texture_context, texId, uv.x, uv.y, ddx, ddy, isResident, texelJitter );
    return float3{f.x, f.y, f.z};
}

//------------------------------------------------------------------------------
// Sampling
//------------------------------------------------------------------------------

struct DirectionSample
{
    float3 direction;
    float3 emap_bsdf_cos;
};

static __forceinline__ __device__
float3 renderRis( Ray& ray, OTKAppRayPayload& payload, SurfaceValues& sv, unsigned int rseed )
{
    const CdfInversionParams* cdfParams = reinterpret_cast<CdfInversionParams*>( params.extraData );
    const float emapAverage = cdfParams->emapCdfInversionTable.aveValue;

    bool resident;
    float3 scatterDirection, bsdfVal, emapVal;
    float bsdfProb, emapProb, psampled, pdesired, cosTerm, sampleWeight;
    
    // Set up random direction update
    unsigned int numSamples = cdfParams->numRisSamples;
    float2 dxi = float2{ 1.0f / sqrtf(numSamples) + 1.0f / numSamples, 1.0f / numSamples };
    float2 xi = rnd2(rseed);

    // Use a reservoir to figure out final sample
    Reservoir<DirectionSample> reservoir;
    DirectionSample sample;
    for( int i = 0; i < numSamples; ++i )
    {
        // Alternately sample BSDF or EMAP
        if( i%2 == 0 )
            sampleBsdf( payload.geometry, sv, ray.d, xi, scatterDirection );
        else
            scatterDirection = uvToVec( sampleEmap( xi ) );

        // Compute BSDF and EMAP values
        cosTerm = maxf( 0.0f, dot(payload.geometry.N, scatterDirection) );
        evalBsdf( payload.geometry, sv, ray.d, scatterDirection, bsdfVal, bsdfProb );
        RayCone rayCone = payload.rayCone1;
        float fs = 0.33333f * (bsdfVal.x + bsdfVal.y + bsdfVal.z);
        rayConeReflect( payload.geometry.curvature, fs, rayCone );
        emapVal = evalEmap( vecToUv(scatterDirection), rayCone.angle, &resident, xi );
        emapProb = EPS + LUMINANCE( emapVal ) / ( emapAverage * 4.0f * M_PIf );

        // Calculate probabilities and update reservoir
        sample = DirectionSample{scatterDirection, bsdfVal * emapVal * cosTerm};
        psampled = 0.5f * ( bsdfProb + emapProb );
        pdesired = LUMINANCE( sample.emap_bsdf_cos );
        if( pdesired > 0.0f )
            sample.emap_bsdf_cos /= pdesired;
        sampleWeight = ( psampled > 0.0f ) ? pdesired / psampled : 0.0f;
        reservoir.update( sample, sampleWeight, rnd(rseed) );

        // Update random number
        xi = xi + dxi;
        xi -= float2{ floorf(xi.x), floorf(xi.y) };
    }

    // No viable samples found.
    if( reservoir.wsum == 0.0f )
        return float3{0.0f};

    // Determine if the direction is occluded
    traceRay( params.traversable_handle, payload.geometry.P, reservoir.y.direction, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
    if( payload.occluded )
        return float3{0.0f};

    return reservoir.y.emap_bsdf_cos * (reservoir.wsum / reservoir.m);
}

static __forceinline__ __device__
float3 renderMis( Ray& ray, OTKAppRayPayload& payload, SurfaceValues& sv, float2 xi )
{
    const CdfInversionParams* cdfParams = reinterpret_cast<CdfInversionParams*>( params.extraData );
    const float emapAverage = cdfParams->emapCdfInversionTable.aveValue;

    bool resident;
    float3 scatterDirection, bsdfVal, emapVal;
    float bsdfProb, emapProb, prob;

    // Switch between BSDF, EMAP, and MIS render
    float bsdfWeight = 0.5f;
    if( params.render_mode == BSDF_RENDER )
        bsdfWeight = 1.0f;
    else if( params.render_mode == EMAP_RENDER )
        bsdfWeight = 0.0f;

    // Choose bsdf or environment map sample
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( bsdfWeight == 1.0f || bsdfWeight > 0.0f &&
        ((params.subframe + px.x + px.y) % 2 == 0) )
    {
        sampleBsdf( payload.geometry, sv, ray.d, xi, scatterDirection );
    }
    else 
    {
        float2 emapUv = sampleEmap( xi );
        scatterDirection = uvToVec( emapUv );
    }

    // Cosine term and backface test
    float cosTerm = dot(payload.geometry.N, scatterDirection);
    if( cosTerm < 0.0f )
        return float3{0.0f};

    // Evaluate the bsdf
    evalBsdf( payload.geometry, sv, ray.d, scatterDirection, bsdfVal, bsdfProb );
    if( !(bsdfProb > 0.0f) )
        return float3{0.0f};

    // Determine if the direction is occluded
    traceRay( params.traversable_handle, payload.geometry.P, scatterDirection, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
    if( payload.occluded )
        return float3{0.0f};

    // Update the ray cone and evaluate environment map
    float fs = 0.33333f * (bsdfVal.x + bsdfVal.y + bsdfVal.z);
    rayConeReflect( payload.geometry.curvature, fs, payload.rayCone1 );

    emapVal = evalEmap( vecToUv(scatterDirection), payload.rayCone1.angle, &resident, xi );
    emapProb = EPS + LUMINANCE( emapVal ) / ( emapAverage * 4.0f * M_PIf );
    prob = (1.0f - bsdfWeight) * emapProb + bsdfWeight * bsdfProb;

    return emapVal * bsdfVal * cosTerm / prob;
}

//------------------------------------------------------------------------------
// OptiX programs
//------------------------------------------------------------------------------
 
extern "C" __global__ void __raygen__rg()
{
    const CdfInversionParams* cdfParams = reinterpret_cast<CdfInversionParams*>( params.extraData );

    bool resident;
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) )
        return;

    // Initialize random number seed and radiance
    const int subframeId = params.subframe;
    unsigned int rseed = srand( px.x, px.y, subframeId );

    // Ray and payload definition
    Ray ray;
    OTKAppRayPayload payload{};
    makeEyeRay( px, ray.o, ray.d, payload.rayCone1, rnd2( rseed ) );
    payload.rayCone2 = payload.rayCone1;

    // Trace ray. If it misses, sample the environment map, otherwise render
    float3 radiance = float3{0.0f, 0.0f, 0.0f};
    traceRay( params.traversable_handle, ray.o, ray.d, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
    payload.geometry.ddx *= cdfParams->mipScale;
    payload.geometry.ddy *= cdfParams->mipScale;

    if( !payload.occluded )
    {
        radiance = evalEmap( vecToUv( ray.d ), payload.rayCone1.angle, &resident, rnd2(rseed) );
    }
    else
    {
        SurfaceValues sv;
        getSurfaceValues( payload.geometry, (SurfaceTexture*)payload.material, sv );
        float3 colorScale = float3{1.0f, 1.0f, 1.0f};
        float3 scatterDirection;

        // Special case for perfect mirror: allow an extra bounce
        if( sv.roughness <= MIN_ROUGHNESS )
        {
            colorScale = sv.Ks;
            scatterDirection = reflect( ray.d, payload.geometry.N );
            payload.rayCone1 = reflect( payload.rayCone1, payload.geometry.curvature );
            traceRay( params.traversable_handle, payload.geometry.P, scatterDirection, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
            ray.d = scatterDirection;
            getSurfaceValues( payload.geometry, (SurfaceTexture*)payload.material, sv );
        }

        if( !payload.occluded )
        {
            radiance = colorScale * evalEmap( vecToUv(scatterDirection), payload.rayCone1.angle, &resident, rnd2(rseed) );
        }
        else
        {
            if( params.render_mode == RIS_RENDER )
                radiance = colorScale * renderRis( ray, payload, sv, rseed );
            else
                radiance = colorScale * renderMis( ray, payload, sv, rnd2(rseed) );
        }
    }

    // Accumulate sample in the accumulation buffer
    unsigned int image_idx = px.y * params.image_dim.x + px.x;
    float4 accum_color = float4{radiance.x, radiance.y, radiance.z, 1.0f};
    if( subframeId != 0 )
        accum_color += params.accum_buffer[image_idx];
    params.accum_buffer[image_idx] = accum_color;

    accum_color *= ( 1.0f / accum_color.w );
    params.result_buffer[image_idx] = make_color( accum_color );
}
