// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/DeviceTriangles.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/PhongShade.h"

#include <optix.h>

#define TARGET_CODE_USE_CUDA_TYPES
#include <mi/neuraylib/target_code_types.h>

#include <vector_functions.h>

#include <cassert>

using namespace otk;  // for vec_math operators

namespace demandPbrtScene {

// Flip V because PBRT texture coordinate space has (0,0) at the lower left corner.
__device__ __forceinline__ float2 adjustMdlUV( float2 uv )
{
    return make_float2( uv.x, 1.f - uv.y );
}

__device__ __forceinline__ float2 interpolateMdlUVs( const TriangleUVs& uv )
{
    const float2 bc = optixGetTriangleBarycentrics();
    return adjustMdlUV( uv.UV[0] ) * ( 1.0f - bc.x - bc.y ) + adjustMdlUV( uv.UV[1] ) * bc.x + adjustMdlUV( uv.UV[2] ) * bc.y;
}

__device__ __forceinline__ float2 getMdlTriangleUVs( TriangleUVs** uvs, const uint_t index )
{
#ifndef NDEBUG
    static const float2 zero{};
    if( uvs == nullptr )
    {
        printf( "Parameters uvs array is nullptr!\n" );
        return zero;
    }
#endif
    const TriangleUVs* triangleUVs = uvs[index];
#ifndef NDEBUG
    if( triangleUVs == nullptr )
    {
        printf( "Parameters uvs array for material %u is nullptr!\n", index );
        return zero;
    }
#endif
    return interpolateMdlUVs( triangleUVs[optixGetPrimitiveIndex()] );
}

__device__ __forceinline__ uint_t getMdlMaterialId( const Params& params, uint_t instanceId )
{
#ifndef NDEBUG
    if( instanceId >= params.numMaterialIndices )
    {
        printf( "Instance id %u exceeds number of MaterialIndex entries %u\n", instanceId, params.numMaterialIndices );
        assert( instanceId < params.numMaterialIndices );
    }
#endif
    const MaterialIndex groups{ params.materialIndices[instanceId] };
    const uint_t        primIdx{ optixGetPrimitiveIndex() };
    uint_t              matIdx{ groups.primitiveMaterialBegin };
    for( uint_t i = 0; i < groups.numPrimitiveGroups; ++i )
    {
#ifndef NDEBUG
        if( matIdx >= params.numPrimitiveMaterials )
        {
            printf( "Material index %u exceeds number of PrimitiveMaterialRange entries %u\n", matIdx, params.numPrimitiveMaterials );
            assert( matIdx < params.numPrimitiveMaterials );
        }
#endif
        const PrimitiveMaterialRange& group{ params.primitiveMaterials[matIdx] };
        if( primIdx < group.primitiveEnd )
        {
            return group.materialId;
        }
        ++matIdx;
    }
#ifndef NDEBUG
    printf( "Requested material for instance id %u, primitive index %u not found in MaterialIndex{%u, %u}\n",
            instanceId, primIdx, groups.numPrimitiveGroups, groups.primitiveMaterialBegin );
    assert( false );
#endif
    return ~0U;
}

__device__ __forceinline__ bool useMdlShader( const Params& params, uint_t materialId, MdlMaterialShader& shader )
{
    if( params.materialStates == nullptr || materialId >= params.numMaterialStates )
    {
        return false;
    }

    const MaterialState& state{ params.materialStates[materialId] };
    if( state.backend != MaterialBackend::MDL_READY )
    {
        return false;
    }

#ifndef NDEBUG
    if( state.shaderKey >= params.numMdlMaterialShaders )
    {
        printf( "Shader key %u exceeds number of MDL material shader entries %u\n", state.shaderKey, params.numMdlMaterialShaders );
        assert( state.shaderKey < params.numMdlMaterialShaders );
    }
#endif
    if( state.shaderKey >= params.numMdlMaterialShaders || params.mdlMaterialShaders == nullptr )
    {
        return false;
    }

    shader = params.mdlMaterialShaders[state.shaderKey];
    return shader.callableCount == 1U;
}

extern "C" __global__ void __closesthit__mdlMesh()
{
    float3 worldNormal;
    float3 vertices[3];
    getTriangleData( vertices, worldNormal );

    if( triMeshMaterialDebugInfo( vertices, worldNormal, optixGetTriangleBarycentrics() ) )
    {
        return;
    }

    const Params& params{ PARAMS_VAR_NAME };
    const uint_t  instanceId{ optixGetInstanceId() };
    const uint_t  materialId{ getMdlMaterialId( params, instanceId ) };
#ifndef NDEBUG
    if( materialId >= params.numRealizedMaterials )
    {
        printf( "Material id %u exceeds numRealizedMaterials %u\n", materialId, params.numRealizedMaterials );
        assert( materialId < params.numRealizedMaterials );
    }
#endif

    const float3      rayOrigin{ optixGetWorldRayOrigin() };
    const float3      rayDirection{ optixGetWorldRayDirection() };
    const float       rayT{ optixGetRayTmax() };
    PhongMaterial     material{ params.realizedMaterials[materialId] };
    RayPayload* const prd{ getRayPayload() };
    MdlMaterialShader shader{};

    prd->diffuseTextureId = 0xffffffff;
    prd->material         = nullptr;
    prd->normal           = worldNormal;
    prd->rayDistance      = rayT;

    if( !useMdlShader( params, materialId, shader ) )
    {
        prd->color          = phongShade( material, worldNormal, rayDirection );
        prd->hasDirectColor = true;
        return;
    }

    mi::neuraylib::Shading_state_material state{};
    state.normal      = worldNormal;
    state.geom_normal = worldNormal;
    state.position    = rayOrigin + rayT * rayDirection;

    mi::neuraylib::Resource_data resourceData{};
    mi::neuraylib::tct_float3    tint{};
    optixDirectCall<void, void*, const mi::neuraylib::Shading_state_material*, const mi::neuraylib::Resource_data*, const char*>(
        shader.callableBaseIndex, &tint, &state, &resourceData, nullptr );

    material.Kd = make_float3( tint.x, tint.y, tint.z );

    prd->materialCopy   = material;
    prd->color          = phongShade( material, worldNormal, rayDirection );
    prd->hasDirectColor = true;

    if( ( static_cast<uint_t>( material.flags ) & static_cast<uint_t>( MaterialFlags::DIFFUSE_MAP_ALLOCATED ) ) == 0U )
    {
        return;
    }

#ifndef NDEBUG
    if( instanceId >= params.numInstanceUVs )
    {
        printf( "Instance id %u exceeds numInstanceUVs %u\n", instanceId, params.numInstanceUVs );
        assert( instanceId < params.numInstanceUVs );
    }
#endif
    prd->diffuseTextureId = material.diffuseTextureId;
    prd->material         = &prd->materialCopy;
    prd->uv               = getMdlTriangleUVs( params.instanceUVs, instanceId );
    prd->hasDirectColor   = false;

    const float2* uvs          = params.instanceUVs[instanceId][optixGetPrimitiveIndex()].UV;
    const float   a            = otk::length( uvs[2] - uvs[0] ) / otk::length( vertices[2] - vertices[0] );
    const float   b            = otk::length( uvs[2] - uvs[0] ) / otk::length( vertices[2] - vertices[0] );
    const float   c            = otk::length( uvs[2] - uvs[0] ) / otk::length( vertices[2] - vertices[0] );
    prd->worldSpaceTextureSize = ( a + b + c ) / 3.0f;
}

}  // namespace demandPbrtScene
