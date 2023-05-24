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

namespace demandGeometryViewer {

extern "C" __constant__ Params params;

template <typename T>
__forceinline__ __device__ T* getSbtData()
{
    return reinterpret_cast<T*>( optixGetSbtDataPointer() );
}

static __forceinline__ __device__ void setRayPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
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
    setRayPayload( result );
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
