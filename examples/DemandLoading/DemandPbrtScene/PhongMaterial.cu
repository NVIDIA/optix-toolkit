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

#include "DeviceTriangles.h"
#include "Params.h"
#include "PhongShade.h"

#include <optix.h>

#include <vector_functions.h>

using namespace otk;  // for vec_math operators

namespace demandPbrtScene {

extern "C" __global__ void __closesthit__mesh()
{
    const Params& params{ PARAMS_VAR_NAME };
    float3 worldNormal;
    float3 vertices[3];
    getTriangleData( vertices, worldNormal );

    float2 uv = optixGetTriangleBarycentrics();

    if( triMeshMaterialDebugInfo( vertices, worldNormal, uv ) )
        return;

    const PhongMaterial& mat = PARAMS_VAR_NAME.realizedMaterials[optixGetInstanceId()];
    // Hack: Return phong shaded value for PHONG_SHADING, Kd otherwise
    float3 shaded = ( params.renderMode == PHONG_SHADING ) ? phongShade( mat, worldNormal ) : mat.Kd;
    setRayPayload( shaded );

    // Values needed for path tracing
    optixSetPayload_3( __float_as_uint( optixGetRayTmax() ) );
    optixSetPayload_4( __float_as_uint( worldNormal.x ) );
    optixSetPayload_5( __float_as_uint( worldNormal.y ) );
    optixSetPayload_6( __float_as_uint( worldNormal.z ) );
}

static __forceinline__ __device__ bool sphereMaterialDebugInfo( const float4& q, const float3& worldNormal )
{
    return debugInfoDump(
        PARAMS_VAR_NAME.debug,
        [&]( const uint3& launchIndex ) {
            const uint_t                 instIdx     = optixGetInstanceIndex();
            const uint_t                 instId      = optixGetInstanceId();
            const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
            const uint_t                 sbtGASIndex = optixGetSbtGASIndex();
            printf(                                                                                                //
                "[%u, %u, %u]: instance %u (id %u), GAS index: %u, GAS: %llx, q: [%g,%g,%g,%g], N: [%g,%g,%g]\n",  //
                launchIndex.x, launchIndex.y, launchIndex.z,                                                       //
                instIdx, instId, sbtGASIndex, gas,                                                                 //
                q.x, q.y, q.z, q.w,                                                                                //
                worldNormal.x, worldNormal.y, worldNormal.z );                                                     //
        },
        setRayPayload );
}

extern "C" __global__ void __closesthit__sphere()
{
    const Params& params{ PARAMS_VAR_NAME };
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
    const float3 worldNormal  = normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) );

    if( sphereMaterialDebugInfo( q, worldNormal ) )
        return;

    // Hack: Return phong shaded value for PHONG_SHADING, Kd otherwise
    const PhongMaterial& mat = PARAMS_VAR_NAME.realizedMaterials[optixGetInstanceId()];
    float3 shaded = ( params.renderMode == PHONG_SHADING ) ? phongShade( mat, worldNormal ) : mat.Kd;
    setRayPayload( shaded );

    // Values needed for path tracing
    optixSetPayload_3( __float_as_uint( optixGetRayTmax() ) );
    optixSetPayload_4( __float_as_uint( worldNormal.x ) );
    optixSetPayload_5( __float_as_uint( worldNormal.y ) );
    optixSetPayload_6( __float_as_uint( worldNormal.z ) );
}

}  // namespace demandPbrtScene
