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

#include <OptiXToolkit/ShaderUtil/AliasTable.h>
#include <OptiXToolkit/ShaderUtil/CdfInversionTable.h>
#include <OptiXToolkit/ShaderUtil/PdfTable.h>
#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/Reservoir.h>
#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/DemandTextureAppBase/LaunchParams.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureAppDeviceUtil.h>
#include <OptiXToolkit/DemandTextureAppBase/SimpleBsdf.h>

#include "CdfInversionParams.h"

using namespace demandLoading;
using namespace demandTextureApp;
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
__constant__ Params params;
}

//------------------------------------------------------------------------------
// Ray Payload - per ray data for OptiX programs
//------------------------------------------------------------------------------

struct Ray
{
    float3 o, d;
};

struct RayPayload
{
    RayCone rayCone;
    SurfaceGeometry geom;
    const SurfaceTexture* tex;
    bool occluded;
    int rayDepth;
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
CdfInversionTable* getInversionTable() 
{
    return reinterpret_cast<CdfInversionTable*>( &params.c[EMAP_INVERSION_TABLE_ID] );
}

static __forceinline__ __device__
float2 sampleEmap( float2 xi )
{
    #ifdef emBIN_SEARCH
        return sampleCdfBinSearch( *getInversionTable(), xi );
    #elif emLIN_SEARCH
        return sampleCdfLinSearch( *getInversionTable(), xi );
    #elif emDIRECT_LOOKUP
        return sampleCdfDirectLookup( *getInversionTable(), xi );
    #elif emALIAS_TABLE
        CdfInversionTable* it = getInversionTable();
        AliasTable* at = reinterpret_cast<AliasTable*>( &params.c[EMAP_ALIAS_TABLE_ID] );
        return alias2D( *at, it->width, it->height, xi );
    #endif
}

static __forceinline__ __device__
float3 evalEmap( float2 uv, float rayConeAngle, bool* isResident, float2 xi )
{
    // Switch using ray cone and mip level 0
    if( params.i[MIP_LEVEL_0_ID] )
    {
        xi = float2{0.5f, 0.5f};
        rayConeAngle = 0.0f;
    }

    unsigned int texId = params.i[EMAP_ID];
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
float3 renderRis( Ray& ray, RayPayload& payload, SurfaceValues& sv, unsigned int rseed )
{
    bool resident;
    float3 scatterDirection, bsdfVal, emapVal;
    float bsdfProb, emapProb, psampled, pdesired, cosTerm, sampleWeight;
    
    // Set up random direction update
    unsigned int numSamples = params.i[NUM_RIS_SAMPLES];
    float2 dxi = float2{ 1.0f / sqrtf(numSamples) + 1.0f / numSamples, 1.0f / numSamples };
    float2 xi = rnd2(rseed);

    // Use a reservoir to figure out final sample
    Reservoir<DirectionSample> reservoir;
    DirectionSample sample;
    for( int i = 0; i < numSamples; ++i )
    {
        // Alternately sample BSDF or EMAP
        if( i%2 == 0 )
            sampleBsdf( payload.geom, sv, ray.d, xi, scatterDirection );
        else
            scatterDirection = uvToVec( sampleEmap( xi ) );

        // Compute BSDF and EMAP values
        cosTerm = maxf( 0.0f, dot(payload.geom.N, scatterDirection) );
        evalBsdf( payload.geom, sv, ray.d, scatterDirection, bsdfVal, bsdfProb );
        RayCone rayCone = payload.rayCone;
        float fs = 0.33333f * (bsdfVal.x + bsdfVal.y + bsdfVal.z);
        rayConeReflect( payload.geom.curvature, fs, rayCone );
        emapVal = evalEmap( vecToUv(scatterDirection), rayCone.angle, &resident, xi );
        emapProb = EPS + LUMINANCE( emapVal ) / ( getInversionTable()->aveValue * 4.0f * M_PIf );

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
    traceRay( params.traversable_handle, payload.geom.P, reservoir.y.direction, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
    if( payload.occluded )
        return float3{0.0f};

    return reservoir.y.emap_bsdf_cos * (reservoir.wsum / reservoir.m);
}

static __forceinline__ __device__
float3 renderMis( Ray& ray, RayPayload& payload, SurfaceValues& sv, float2 xi )
{
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
        ((params.i[SUBFRAME_ID] + px.x + px.y) % 2 == 0) )
    {
        sampleBsdf( payload.geom, sv, ray.d, xi, scatterDirection );
    }
    else 
    {
        float2 emapUv = sampleEmap( xi );
        scatterDirection = uvToVec( emapUv );
    }

    // Cosine term and backface test
    float cosTerm = dot(payload.geom.N, scatterDirection);
    if( cosTerm < 0.0f )
        return float3{0.0f};

    // Evaluate the bsdf
    evalBsdf( payload.geom, sv, ray.d, scatterDirection, bsdfVal, bsdfProb );
    if( !(bsdfProb > 0.0f) )
        return float3{0.0f};

    // Determine if the direction is occluded
    traceRay( params.traversable_handle, payload.geom.P, scatterDirection, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
    if( payload.occluded )
        return float3{0.0f};

    // Update the ray cone and evaluate environment map
    float fs = 0.33333f * (bsdfVal.x + bsdfVal.y + bsdfVal.z);
    rayConeReflect( payload.geom.curvature, fs, payload.rayCone );

    emapVal = evalEmap( vecToUv(scatterDirection), payload.rayCone.angle, &resident, xi );
    emapProb = EPS + LUMINANCE( emapVal ) / ( getInversionTable()->aveValue * 4.0f * M_PIf );
    prob = (1.0f - bsdfWeight) * emapProb + bsdfWeight * bsdfProb;

    return emapVal * bsdfVal * cosTerm / prob;
}

//------------------------------------------------------------------------------
// OptiX programs
//------------------------------------------------------------------------------
 
extern "C" __global__ void __raygen__rg()
{
    bool resident;
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) )
        return;

    // Initialize random number seed and radiance
    const int subframeId = params.i[SUBFRAME_ID];
    unsigned int rseed = srand( px.x, px.y, subframeId );

    // Ray and payload definition
    Ray ray;
    RayPayload payload{};
    makeEyeRay( px, ray.o, ray.d, payload.rayCone, rnd2( rseed ) );

    // Trace ray. If it misses, sample the environment map, otherwise render
    float3 radiance = float3{0.0f, 0.0f, 0.0f};
    traceRay( params.traversable_handle, ray.o, ray.d, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
    if( !payload.occluded )
    {
        radiance = evalEmap( vecToUv( ray.d ), payload.rayCone.angle, &resident, rnd2(rseed) );
    }
    else
    {
        SurfaceValues sv;
        getSurfaceValues( payload.geom, payload.tex, sv );
        float3 colorScale = float3{1.0f, 1.0f, 1.0f};
        float3 scatterDirection;

        // Special case for perfect mirror: allow an extra bounce
        if( sv.roughness <= MIN_ROUGHNESS )
        {
            colorScale = sv.Ks;
            scatterDirection = reflect( ray.d, payload.geom.N );
            payload.rayCone = reflect( payload.rayCone, payload.geom.curvature );
            traceRay( params.traversable_handle, payload.geom.P, scatterDirection, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
            ray.d = scatterDirection;
            getSurfaceValues( payload.geom, payload.tex, sv );
        }

        if( !payload.occluded )
        {
            radiance = colorScale * evalEmap( vecToUv(scatterDirection), payload.rayCone.angle, &resident, rnd2(rseed) );
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

extern "C" __global__ void __miss__ms()
{
    const float4 bg = reinterpret_cast<MissData*>( optixGetSbtDataPointer() )->background_color;
    getRayPayload()->occluded = false;
}

extern "C" __global__ void __closesthit__ch()
{
    RayPayload* payload = getRayPayload();
    payload->occluded = true;
    if( isOcclusionRay() ) // for occlusion query, just return
        return;

    // Get hit info
    const TriangleHitGroupData* hg_data = reinterpret_cast<TriangleHitGroupData*>( optixGetSbtDataPointer() );
    const int vidx = optixGetPrimitiveIndex() * 3;
    const float3 bary = getTriangleBarycentrics();
    const float3 D = optixGetWorldRayDirection();
    const float rayDistance = optixGetRayTmax();

    // Get triangle geometry
    float4* vertices = &hg_data->vertices[vidx];
    float3& Va = *reinterpret_cast<float3*>( &vertices[0] );
    float3& Vb = *reinterpret_cast<float3*>( &vertices[1] );
    float3& Vc = *reinterpret_cast<float3*>( &vertices[2] );
    const float3* normals = &hg_data->normals[vidx];
    const float2* tex_coords = &hg_data->tex_coords[vidx];

    // Compute Surface geometry for hit point
    SurfaceGeometry& geom = payload->geom;
    geom.P = bary.x * Va + bary.y * Vb + bary.z * Vc; 
    geom.Ng = normalize( cross( Vb-Va, Vc-Va ) ); //geometric normal
    geom.N = normalize( bary.x * normals[0] + bary.y * normals[1] + bary.z * normals[2] ); // shading normal
    makeOrthoBasis( geom.N, geom.S, geom.T );
    geom.uv = bary.x * tex_coords[0] + bary.y * tex_coords[1] + bary.z * tex_coords[2];
    geom.curvature = minTriangleCurvature( Va, Vb, Vc, normals[0], normals[1], normals[2] );

    // Flip normal for local geometry if needed
    geom.flipped = false;
    if( dot( D, geom.Ng ) > 0.0f )
    {
        geom.Ng *= -1.0f;
        geom.N *= -1.0f;
        geom.curvature *= -1.0f;
        geom.flipped = true;
    }

    // Get the surface texture
    payload->tex = &hg_data->tex;

    // Propagate the ray cone and construct ray differentials on surface
    float3 dPdx, dPdy;
    payload->rayCone = propagate( payload->rayCone, rayDistance );
    projectToRayDifferentialsOnSurface( payload->rayCone.width, D, geom.N, dPdx, dPdy );
    computeTexGradientsForTriangle( Va, Vb, Vc, tex_coords[0], tex_coords[1], tex_coords[2], dPdx, dPdy, geom.ddx, geom.ddy );
    geom.ddx *= params.f[MIP_SCALE_ID];
    geom.ddy *= params.f[MIP_SCALE_ID];
}
