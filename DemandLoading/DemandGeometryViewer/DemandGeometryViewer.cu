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

#include "DemandGeometryViewer.h"

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>

namespace demandGeometryViewer {

extern "C" __constant__ Params params;

struct RadiancePRD
{
    float3 result;
};

template <typename T>
__forceinline__ __device__ T* getSbtData()
{
    return reinterpret_cast<T*>( optixGetSbtDataPointer() );
}

template <typename T>
__forceinline__ __device__ uint_t& attr( T& val )
{
    return reinterpret_cast<uint_t&>( val );
}

#define float3Attr( vec_ ) attr( ( vec_ ).x ), attr( ( vec_ ).y ), attr( ( vec_ ).z )

__forceinline__ __device__ uchar4 makeColor( const float3& c )
{
    return make_uchar4( static_cast<unsigned char>( clamp( c.x, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<unsigned char>( clamp( c.y, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<unsigned char>( clamp( c.z, 0.0f, 1.0f ) * 255.0f ), 255u );
}

extern "C" __global__ void __raygen__pinHoleCamera()
{
    const uint3  idx    = optixGetLaunchIndex();
    const auto*  camera = getSbtData<CameraData>();
    const uint_t pixel  = params.width * idx.y + idx.x;

    float2      d         = make_float2( idx.x, idx.y ) / make_float2( params.width, params.height ) * 2.f - 1.f;
    float3      rayOrigin = camera->eye;
    float3      rayDir    = normalize( d.x * camera->U + d.y * camera->V + camera->W );
    RadiancePRD prd{};

    float         tMin         = 0.0f;
    float         tMax         = 1e16f;
    float         rayTime      = 0.0f;
    OptixRayFlags flags        = OPTIX_RAY_FLAG_NONE;
    uint_t        sbtOffset    = RAYTYPE_RADIANCE;
    uint_t        sbtStride    = RAYTYPE_COUNT;
    uint_t        missSbtIndex = RAYTYPE_RADIANCE;
    optixTrace( params.traversable, rayOrigin, rayDir, tMin, tMax, rayTime, OptixVisibilityMask( 255 ), flags,
                sbtOffset, sbtStride, missSbtIndex, float3Attr( prd.result ) );

    params.image[pixel] = makeColor( prd.result );
}

static __forceinline__ __device__ void setRayPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

extern "C" __global__ void __miss__backgroundColor()
{
    const auto* data = getSbtData<MissData>();
    setRayPayload( make_float3( data->background.x, data->background.y, data->background.z ) );
}

static __device__ __inline__ RadiancePRD getRadiancePRD()
{
    RadiancePRD prd;
    prd.result.x = __uint_as_float( optixGetPayload_0() );
    prd.result.y = __uint_as_float( optixGetPayload_1() );
    prd.result.z = __uint_as_float( optixGetPayload_2() );
    return prd;
}

static __device__ __inline__ void setRadiancePRD( const RadiancePRD& prd )
{
    optixSetPayload_0( __float_as_uint( prd.result.x ) );
    optixSetPayload_1( __float_as_uint( prd.result.y ) );
    optixSetPayload_2( __float_as_uint( prd.result.z ) );
}

static __device__ void phongShade( float3 const& p_Kd,
                                   float3 const& p_Ka,
                                   float3 const& p_Ks,
                                   float3 const& p_Kr,
                                   float const&  p_phong_exp,
                                   float3 const& p_normal )
{
    const float3 rayDir = optixGetWorldRayDirection();
    RadiancePRD  prd    = getRadiancePRD();

    // ambient contribution
    float3 result = p_Ka * params.ambientColor;

    // Illuminate using three lights
    for( int i = 0; i < 3; i++ )
    {
        // note that the light "position" (really a direction) passed in is assumed to be normalized
        float3 lightPos = params.lights[i].pos;

        // for directional lights the effect is simply the surface normal dot light position
        float nDl = dot( p_normal, lightPos );

        if( nDl > 0.0f )
        {
            // perform the computation
            float3 phongRes = p_Kd * nDl;

            float3 H   = normalize( lightPos - rayDir );
            float  nDh = dot( p_normal, H );
            if( nDh > 0 )
            {
                float power = pow( nDh, p_phong_exp );
                phongRes += p_Ks * power;
            }
            result += phongRes * params.lights[i].color;
        }
    }

    // pass the color back
    prd.result = result;
    setRadiancePRD( prd );
}

extern "C" __global__ void __closesthit__sphere()
{
    const float tHit = optixGetRayTmax();

    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDir    = optixGetWorldRayDirection();

    const unsigned int           primIdx     = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, primIdx, sbtGASIndex, 0.f, &q );

    const float3 worldRayPos  = rayOrigin + tHit * rayDir;
    const float3 objectRayPos = optixTransformPointFromWorldToObjectSpace( worldRayPos );
    const float3 objectNormal = ( objectRayPos - make_float3( q ) ) / q.w;
    const float3 worldNormal  = normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) ) * 0.5f + 0.5f;

    const PhongMaterial& mat = getSbtData<HitGroupData>()->material;
    phongShade( mat.Kd, mat.Ka, mat.Ks, mat.Kr, mat.phongExp, worldNormal );
}

}  // namespace demandGeometryViewer

namespace demandGeometry {
namespace app {

__device__ Context& getContext()
{
    return demandGeometryViewer::params.demandGeomContext;
}

__device__ const demandLoading::DeviceContext& getDeviceContext()
{
    return demandGeometryViewer::params.demandContext;
}

__device__ void reportClosestHitNormal( float3 ffNormal )
{
    // Use a single material for all proxies.
    const demandGeometryViewer::PhongMaterial& mat = demandGeometryViewer::params.proxyMaterial;
    demandGeometryViewer::phongShade( mat.Kd, mat.Ka, mat.Ks, mat.Kr, mat.phongExp, ffNormal );
}


}  // namespace app
}  // namespace demandGeometry

#include <OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h>
