// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/DeviceTriangles.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/PhongShade.h"

#include <optix.h>

#include <vector_functions.h>

#include <cassert>

using namespace otk;  // for vec_math operators

namespace demandPbrtScene {

extern "C" __global__ void __closesthit__mesh()
{
    float3 worldNormal;
    float3 vertices[3];
    getTriangleData( vertices, worldNormal );

    float2 uv = optixGetTriangleBarycentrics();

    if( triMeshMaterialDebugInfo( vertices, worldNormal, uv ) )
    {
        return;
    }

    RayPayload* prd       = getRayPayload();
    prd->diffuseTextureId = 0xffffffff;
    const Params&                 params{ PARAMS_VAR_NAME };
    const uint_t                  instanceId{ optixGetInstanceId() };
    const uint_t                  primIdx{ optixGetPrimitiveIndex() };
    const MaterialIndex&          matIdx{ params.materialIndices[instanceId] };
    uint_t                        materialId{};
    const PrimitiveMaterialRange* matRange{ &params.primitiveMaterials[matIdx.primitiveMaterialBegin] };
    for( uint_t i = 0; i < matIdx.numPrimitiveGroups; ++i )
    {
        if( primIdx < matRange[i].primitiveEnd )
        {
            materialId = matRange[i].materialId;
            break;
        }
    }
#ifndef NDEBUG
    if( materialId >= params.numRealizedMaterials )
    {
        printf( "Material id %u exceeds numRealizedMaterials %u\n", materialId, params.numRealizedMaterials );
        assert( materialId < params.numRealizedMaterials );
    }
#endif
    prd->material    = &params.realizedMaterials[materialId];
    prd->normal      = worldNormal;
    prd->rayDistance = optixGetRayTmax();
    prd->color       = float3{ 1.0f, 0.0f, 1.0f };
}

struct SphereMaterialDebugInfo
{
    __forceinline__ __device__ SphereMaterialDebugInfo( const float4& q, const float3& worldNormal )
        : q( q ), worldNormal( worldNormal )
    {
    }

    __forceinline__ __device__ void dump( const uint3& launchIndex ) const
    {
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
    }

    __forceinline__ __device__ void setColor( float r, float g, float b ) const { setRayPayload( r, g, b ); }

    const float4& q;            // sphere center (q.x, q.y, q.z), sphere radius q.w
    const float3& worldNormal;  // world space normal at intersection point
};

static __forceinline__ __device__ bool sphereMaterialDebugInfo( const float4& q, const float3& worldNormal )
{
    return debugInfoDump( PARAMS_VAR_NAME.debug, SphereMaterialDebugInfo{ q, worldNormal } );
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
    const float3 worldNormal  = normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) );

    if( sphereMaterialDebugInfo( q, worldNormal ) )
        return;

    RayPayload* prd       = getRayPayload();
    prd->diffuseTextureId = 0xffffffff;
    const Params&                 params{ PARAMS_VAR_NAME };
    const uint_t                  instanceId{ optixGetInstanceId() };
    const MaterialIndex&          matIdx{ params.materialIndices[instanceId] };
    uint_t                        materialId{};
    const PrimitiveMaterialRange* matRange{ &params.primitiveMaterials[matIdx.primitiveMaterialBegin] };
    for( uint_t i = 0; i < matIdx.numPrimitiveGroups; ++i )
    {
        if( primIdx < matRange[i].primitiveEnd )
        {
            materialId = matRange[i].materialId;
            break;
        }
    }
#ifndef NDEBUG
    if( materialId >= params.numRealizedMaterials )
    {
        printf( "Material id %u exceeds numRealizedMaterials %u\n", materialId, params.numRealizedMaterials );
        assert( materialId < params.numRealizedMaterials );
    }
#endif
    prd->material    = &params.realizedMaterials[materialId];
    prd->normal      = worldNormal;
    prd->rayDistance = optixGetRayTmax();
    prd->color       = float3{ 1.0f, 0.0f, 1.0f };
}

}  // namespace demandPbrtScene
