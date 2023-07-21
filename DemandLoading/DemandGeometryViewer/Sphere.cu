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

#include "DemandGeometryViewer.h"

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>

using namespace otk;  // for vec_math operators

namespace demandGeometryViewer {

extern "C" static __constant__ Params g_params;

template <typename T>
__forceinline__ __device__ T* getSbtData()
{
    return reinterpret_cast<T*>( optixGetSbtDataPointer() );
}

static __forceinline__ __device__ void setRayPayload( float x, float y, float z )
{
    optixSetPayload_0( __float_as_uint( x ) );
    optixSetPayload_1( __float_as_uint( y ) );
    optixSetPayload_2( __float_as_uint( z ) );
}

static __forceinline__ __device__ void setRayPayload( float3 p )
{
    setRayPayload( p.x, p.y, p.z );
}

static __device__ void phongShade( float3 const& p_Kd,
                                   float3 const& p_Ka,
                                   float3 const& p_Ks,
                                   float3 const& p_Kr,
                                   float const&  p_phong_exp,
                                   float3 const& p_normal )
{
    const float3 rayDir = optixGetWorldRayDirection();

    // ambient contribution
    float3 result = p_Ka * g_params.ambientColor;

    // Illuminate using three lights
    for( int i = 0; i < 3; i++ )
    {
        // note that the light "position" (really a direction) passed in is assumed to be normalized
        float3 lightPos = g_params.lights[i].pos;

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
            result += phongRes * g_params.lights[i].color;
        }
    }

    // pass the color back
    setRayPayload( result );
}

static __forceinline__ __device__ bool outsideWindow( uint_t lhs, uint_t rhs, uint_t width )
{
    const uint_t bigger  = max( lhs, rhs );
    const uint_t smaller = min( lhs, rhs );
    return bigger - smaller > width;
}

static __forceinline__ __device__ bool inDebugWindow( const uint3& launchIndex, const uint3& debugIndex, uint_t width )
{
    return !( outsideWindow( launchIndex.x, debugIndex.x, width ) || outsideWindow( launchIndex.y, debugIndex.y, width )
              || outsideWindow( launchIndex.z, debugIndex.z, width ) );
}

static __forceinline__ __device__ bool debugInfo( const Debug& debug, const float4& q, const float3& worldNormal )
{
    if( debug.enabled && debug.debugIndexSet )
    {
        const uint3 launchIndex = optixGetLaunchIndex();
        if( debug.debugIndex == launchIndex )
        {
            const uint_t                 primIdx     = optixGetPrimitiveIndex();
            const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
            const uint_t                 sbtGASIndex = optixGetSbtGASIndex();
            const PhongMaterial&         mat         = getSbtData<HitGroupData>()->material;
            printf(
                "[%u, %u, %u]: primitive index: %u, GAS index: %u, GAS: %llx, q: [%g,%g,%g,%g], N: [%g,%g,%g], D: "
                "[%g,%g,%g]\n",
                launchIndex.x, launchIndex.y, launchIndex.z, primIdx, sbtGASIndex, gas, q.x, q.y, q.z, q.w,
                worldNormal.x, worldNormal.y, worldNormal.z, mat.Kd.x, mat.Kd.y, mat.Kd.z );
            setRayPayload( 1.0f, 0.0f, 0.0f );
            return true;
        }
        if( inDebugWindow( launchIndex, debug.debugIndex, 2 ) )
        {
            setRayPayload( 0.0f, 0.0f, 0.0f );
            return true;
        }
        if( inDebugWindow( launchIndex, debug.debugIndex, 4 ) )
        {
            setRayPayload( 1.0f, 1.0f, 1.0f );
            return true;
        }
    }
    return false;
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
    // Work around a bug in optixGetSphereData in drivers before 545.27
    if( g_params.getSphereData.useOptixGetSphereData )
    {
        // sphere center (q.x, q.y, q.z), sphere radius q.w
        optixGetSphereData( gas, primIdx, sbtGASIndex, 0.f, &q );
    }
    else
    {
        q   = make_float4( g_params.getSphereData.centers[primIdx] );
        q.w = g_params.getSphereData.radii[primIdx];
    }

    const float3 worldRayPos  = rayOrigin + tHit * rayDir;
    const float3 objectRayPos = optixTransformPointFromWorldToObjectSpace( worldRayPos );
    const float3 objectNormal = ( objectRayPos - make_float3( q ) ) / q.w;
    const float3 worldNormal  = normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) ) * 0.5f + 0.5f;

    if( debugInfo( g_params.debug, q, worldNormal ) )
        return;

    const PhongMaterial& mat = getSbtData<HitGroupData>()->material;
    phongShade( mat.Kd, mat.Ka, mat.Ks, mat.Kr, mat.phongExp, worldNormal );
}

}  // namespace demandGeometryViewer
