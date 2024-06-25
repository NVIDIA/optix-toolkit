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

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/DemandTextureAppBase/LaunchParams.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureAppDeviceUtil.h>
#include <OptiXToolkit/DemandTextureAppBase/SimpleBsdf.h>

#include "RayConesParams.h"

using namespace demandLoading;
using namespace demandTextureApp;
using namespace otk;  // for vec_math operators

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
    CURVATURE
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

struct RayPayload
{
    RayCone rayCone1;
    RayCone rayCone2;
    SurfaceGeometry geom;
    const SurfaceTexture* tex;
    bool occluded;
    float3 background;
    int rayDepth;
};

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

static __forceinline__ __device__ float4 alternateOutputColor( RayPayload payload )
{
    bool resident = false;
    SurfaceGeometry& geom = payload.geom;
    const SurfaceTexture* tex = payload.tex;

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
    RayPayload payload{};
   
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

    const int minDepth   = params.i[MIN_RAY_DEPTH_ID];
    const int maxDepth   = params.i[MAX_RAY_DEPTH_ID];
    const int subframeId = params.i[SUBFRAME_ID];

    unsigned int rseed = srand( px.x, px.y, subframeId );
    float3 radiance = float3{0.0f, 0.0f, 0.0f};

    makeEyeRay( px, float2{rnd(rseed), rnd(rseed)}, float2{rnd(rseed), rnd(rseed)}, origin, direction );
    initRayCones( params.camera, params.image_dim, params.projection, params.lens_width, direction, 
                  payload.rayCone1, payload.rayCone2 );
    
    float3 weight = float3{1.0f, 1.0f, 1.0f};
    int rayDepth = 0;

    while( true )
    {
        payload.rayDepth = rayDepth;
        payload.tex = nullptr;
        traceRay( params.traversable_handle, origin, direction, EPS, INF, OPTIX_RAY_FLAG_NONE, &payload );
        if( payload.occluded == false )
        {
            if( rayDepth >= minDepth )
                radiance += weight * payload.background;
            break;
        }
        if( rayDepth >= maxDepth )
            break;

        float3 Ke, Kd, Ks, Kt;
        float roughness, ior;
        bool success = getSurfaceValues( payload.geom, payload.tex, Ke, Kd, Ks, Kt, roughness, ior );
        //if( rayDepth > 3 ) payload.rayCone1.setDiffuse(); // clamp cone angle for deeper paths
        //if( rayDepth > 3 ) payload.rayCone2.setDiffuse(); // clamp cone angle for deeper paths
        radiance += weight * Ke;
        if( payload.geom.flipped && ior != 0.0f )
            ior = 1.0f / ior;

        float3 R, bsdf;
        float prob;

        // Sample the BSDF
        float2 xi = float2{ rnd(rseed), rnd(rseed) };
        if( !sampleBsdf( xi, Kd, Ks, Kt, ior, roughness, payload.geom, direction, R ) )
            break;
        bsdf = evalBsdf( Kd, Ks, Kt, ior, roughness, payload.geom, direction, R, prob, payload.geom.curvature );
        
        // Sanity check.  Discard 0 probability and nan samples
        if( !( prob > 0.0f ) || !( dot( bsdf, bsdf ) >= 0.0f ) )
            break;

        // Update the ray cone
        if( params.i[UPDATE_RAY_CONES_ID] )
        {
            bool isTransmission = dot( R, payload.geom.N ) < 0.0f;
            float fs = bsdf.x + bsdf.y + bsdf.z;
            rayConeScatter( payload.geom.curvature, ior, roughness, isTransmission, fs, payload.rayCone1 );
            rayConeScatter( payload.geom.curvature, ior, roughness, isTransmission, fs, payload.rayCone2 );
        }

        weight = weight * bsdf * (1.0f / prob);
        rayDepth++;
        origin = payload.geom.P;
        direction = R;
    }

    // Accumulate sample in the accumulation buffer
    unsigned int image_idx = px.y * params.image_dim.x + px.x;
    float4 accum_color = float4{radiance.x, radiance.y, radiance.z, 1.0f};
    if( subframeId != 0 )
        accum_color += params.accum_buffer[image_idx];
    params.accum_buffer[image_idx] = accum_color;

    // Blend result of ray trace with tile display and put in result buffer
    accum_color *= ( 1.0f / accum_color.w );
    float4 tcolor = tileDisplayColor( params.demand_texture_context, params.display_texture_id, 10, 10, px );
    float4 color  = ( 1.0f - tcolor.w ) * accum_color + tcolor.w * tcolor;
    params.result_buffer[image_idx] = make_color( color );
}

extern "C" __global__ void __miss__ms()
{
    const float4 bg = reinterpret_cast<MissData*>( optixGetSbtDataPointer() )->background_color;
    getRayPayload()->background = float3{bg.x, bg.y, bg.z};
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

    // pinhole and orthographic camera special case for first hit
    bool pinholeSpecialCase = (payload->rayDepth == 0) && (payload->rayCone1.angle >= 0); 

    // Propagate the ray cone and construct ray differentials on surface
    float3 dPdx, dPdy;
    payload->rayCone1 = propagate( payload->rayCone1, rayDistance );
    payload->rayCone2 = propagate( payload->rayCone2, rayDistance );
    if( fabs(payload->rayCone1.width) >= fabs(payload->rayCone2.width) || pinholeSpecialCase ) 
        projectToRayDifferentialsOnSurface( payload->rayCone1.width, D, geom.N, dPdx, dPdy );
    else
        projectToRayDifferentialsOnSurface( payload->rayCone2.width, D, geom.N, dPdx, dPdy );
    computeTexGradientsForTriangle( Va, Vb, Vc, tex_coords[0], tex_coords[1], tex_coords[2], dPdx, dPdy, geom.ddx, geom.ddy );
    geom.ddx *= params.f[MIP_SCALE_ID];
    geom.ddy *= params.f[MIP_SCALE_ID];
}
