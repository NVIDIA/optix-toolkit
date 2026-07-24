// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/DeviceTriangles.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/PhongShade.h"

#include <OptiXToolkit/DemandLoading/Texture2D.h>

#include <optix.h>

#define TARGET_CODE_USE_CUDA_TYPES
#include <mi/neuraylib/target_code_types.h>

#include <vector_functions.h>

#include <cassert>
#include <cmath>

using namespace otk;  // for vec_math operators

namespace demandPbrtScene {

constexpr uint_t MDL_BSDF_INIT_CALLABLE_OFFSET{ 1U };
constexpr uint_t MDL_BSDF_EVALUATE_CALLABLE_OFFSET{ 3U };
constexpr uint_t MDL_BSDF_CALLABLE_COUNT{ 4U };
constexpr float  INV_PI{ 0.31830988618379067154f };
constexpr float  PI{ 3.14159265358979323846f };
constexpr float  DISPLAY_GAMMA{ 1.0f / 2.2f };

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

__device__ __forceinline__ const TriangleUVs* getMdlTriangleUVArray( TriangleUVs** uvs, const uint_t index )
{
#ifndef NDEBUG
    if( uvs == nullptr )
    {
        printf( "Parameters uvs array is nullptr!\n" );
        assert( uvs != nullptr );
    }
#endif
    const TriangleUVs* triangleUVs = uvs[index];
#ifndef NDEBUG
    if( triangleUVs == nullptr )
    {
        printf( "Parameters uvs array for material %u is nullptr!\n", index );
        assert( triangleUVs != nullptr );
    }
#endif
    return triangleUVs;
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

__device__ __forceinline__ bool hasMdlBsdfCallables( const MdlMaterialShader& shader )
{
    return shader.callableCount >= 1U + MDL_BSDF_CALLABLE_COUNT;
}

__device__ __forceinline__ bool hasMdlDiffuseTexture( const PhongMaterial& material )
{
    return ( static_cast<uint_t>( material.flags ) & static_cast<uint_t>( MaterialFlags::DIFFUSE_MAP_ALLOCATED ) ) != 0U;
}

__device__ __forceinline__ float3 displayEncodeMdlColor( const float3& color )
{
    return make_float3( powf( fmaxf( color.x, 0.0f ), DISPLAY_GAMMA ),  //
                        powf( fmaxf( color.y, 0.0f ), DISPLAY_GAMMA ),  //
                        powf( fmaxf( color.z, 0.0f ), DISPLAY_GAMMA ) );
}

__device__ __forceinline__ float2 concentricMapping( float2 u )
{
    float a = 2.0f * u.x - 1.0f;
    if( a == 0.0f )
        a = 1.0f;
    float b = 2.0f * u.y - 1.0f;
    if( b == 0.0f )
        b = 1.0f;

    float r, phi;
    if( a * a > b * b )
    {
        r   = a;
        phi = ( PI / 4.0f ) * ( b / a );
    }
    else
    {
        r   = b;
        phi = ( PI / 2.0f ) - ( PI / 4.0f ) * ( a / b );
    }
    return float2{ r * cosf( phi ), r * sinf( phi ) };
}

__device__ __forceinline__ void makeOrthoBasis( float3 n, float3& s, float3& t )
{
    s = ( fabsf( n.x ) + fabsf( n.y ) > fabsf( n.z ) ) ? float3{ -n.y, n.x, 0.0f } : float3{ 0.0f, -n.z, n.y };
    s = otk::normalize( s );
    t = otk::cross( s, n );
}

__device__ __forceinline__ float3 sampleDiffuseDirection( const float4& xi, const float3& n )
{
    float3       s, t;
    const float3 normal{ otk::normalize( n ) };
    makeOrthoBasis( normal, s, t );
    const float2 st{ concentricMapping( make_float2( xi.x, xi.y ) ) };
    return otk::normalize( ( st.x * s ) + ( st.y * t ) + ( sqrtf( 1.0f - otk::dot( st, st ) ) * normal ) );
}

__device__ __forceinline__ float getWorldSpaceTextureSize( const float3 ( &vertices )[3], const TriangleUVs& uvs )
{
    const float2* uv = uvs.UV;
    const float   a  = otk::length( uv[2] - uv[0] ) / otk::length( vertices[2] - vertices[0] );
    const float   b  = otk::length( uv[2] - uv[0] ) / otk::length( vertices[2] - vertices[0] );
    const float   c  = otk::length( uv[2] - uv[0] ) / otk::length( vertices[2] - vertices[0] );
    return ( a + b + c ) / 3.0f;
}

__device__ __forceinline__ void setMdlDiffuseTexturePayload( RayPayload*          prd,
                                                             const PhongMaterial& material,
                                                             const float3&        worldNormal,
                                                             float                rayT,
                                                             uint_t               textureId,
                                                             const float2&        uv,
                                                             float                worldSpaceTextureSize )
{
    prd->materialCopy          = material;
    prd->diffuseTextureId      = textureId;
    prd->material              = &prd->materialCopy;
    prd->normal                = worldNormal;
    prd->rayDistance           = rayT;
    prd->uv                    = uv;
    prd->worldSpaceTextureSize = worldSpaceTextureSize;
    prd->hasDirectColor        = false;
    prd->hasMdlBsdfSample      = false;
}

__device__ __forceinline__ void setMdlMaterialDiffuseTexturePayload( const Params&        params,
                                                                     RayPayload*          prd,
                                                                     const PhongMaterial& material,
                                                                     const float3&        worldNormal,
                                                                     const float3 ( &vertices )[3],
                                                                     uint_t instanceId,
                                                                     float  rayT )
{
#ifndef NDEBUG
    if( instanceId >= params.numInstanceUVs )
    {
        printf( "Instance id %u exceeds numInstanceUVs %u\n", instanceId, params.numInstanceUVs );
        assert( instanceId < params.numInstanceUVs );
    }
    if( material.diffuseTextureId < params.minDiffuseTextureId || material.diffuseTextureId > params.maxDiffuseTextureId )
    {
        printf( "Diffuse texture id %u out of range [%u, %u]\n", material.diffuseTextureId, params.minDiffuseTextureId,
                params.maxDiffuseTextureId );
    }
    assert( material.diffuseTextureId >= params.minDiffuseTextureId && material.diffuseTextureId <= params.maxDiffuseTextureId );
#endif

    const TriangleUVs& triangleUVs{ getMdlTriangleUVArray( params.instanceUVs, instanceId )[optixGetPrimitiveIndex()] };
    setMdlDiffuseTexturePayload( prd, material, worldNormal, rayT, material.diffuseTextureId,
                                 interpolateMdlUVs( triangleUVs ), getWorldSpaceTextureSize( vertices, triangleUVs ) );
}

__device__ __forceinline__ float3 sampleMdlDiffuseTexture( uint_t textureId, const float2& uv, bool& isResident )
{
    isResident = true;
    if( textureId == INVALID_TEXTURE_ID )
    {
        return make_float3( 1.0f );
    }

    const Params& params{ PARAMS_VAR_NAME };
#ifndef NDEBUG
    if( textureId < params.minDiffuseTextureId || textureId > params.maxDiffuseTextureId )
    {
        printf( "MDL diffuse texture id %u out of range [%u, %u]\n", textureId, params.minDiffuseTextureId, params.maxDiffuseTextureId );
        assert( textureId >= params.minDiffuseTextureId && textureId <= params.maxDiffuseTextureId );
    }
#endif
    const float4 texel = demandLoading::tex2D<float4>( params.demandContext, textureId, uv.x, uv.y, &isResident );
    return make_float3( texel.x, texel.y, texel.z );
}

__device__ __forceinline__ void initializeMdlBsdf( const MdlMaterialShader&               shader,
                                                   mi::neuraylib::Shading_state_material& state,
                                                   const mi::neuraylib::Resource_data&    resourceData )
{
    optixDirectCall<void, mi::neuraylib::Shading_state_material*, const mi::neuraylib::Resource_data*, const char*>(
        shader.callableBaseIndex + MDL_BSDF_INIT_CALLABLE_OFFSET, &state, &resourceData, nullptr );
}

__device__ __forceinline__ float3 evaluateMdlBsdf( const MdlMaterialShader&                     shader,
                                                   const mi::neuraylib::Shading_state_material& state,
                                                   const mi::neuraylib::Resource_data&          resourceData,
                                                   const float3&                                outgoing,
                                                   const float3&                                incoming,
                                                   const float3&                                textureScale )
{
    mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> evalData{};
    evalData.ior1  = make_float3( 1.0f );
    evalData.ior2  = make_float3( 1.0f );
    evalData.k1    = outgoing;
    evalData.k2    = incoming;
    evalData.flags = mi::neuraylib::DF_FLAGS_ALLOW_REFLECT;
    optixDirectCall<void, mi::neuraylib::Bsdf_evaluate_data_base*, const mi::neuraylib::Shading_state_material*,
                    const mi::neuraylib::Resource_data*, const char*>( shader.callableBaseIndex + MDL_BSDF_EVALUATE_CALLABLE_OFFSET,
                                                                       &evalData, &state, &resourceData, nullptr );
    return ( evalData.bsdf_diffuse + evalData.bsdf_glossy ) * textureScale;
}

__device__ __forceinline__ float3 shadeMdlBsdf( const MdlMaterialShader&                     shader,
                                                const mi::neuraylib::Shading_state_material& state,
                                                const mi::neuraylib::Resource_data&          resourceData,
                                                const float3&                                worldNormal,
                                                const float3&                                rayDirection,
                                                const float3&                                textureScale )
{
    float3        result{};
    const float3  outgoing{ -rayDirection };
    const Params& params{ PARAMS_VAR_NAME };

    for( uint_t i = 0; i < params.numDirectionalLights; ++i )
    {
        const DirectionalLight& light{ params.directionalLights[i] };
        if( otk::dot( worldNormal, light.direction ) > 0.0f )
        {
            result += evaluateMdlBsdf( shader, state, resourceData, outgoing, light.direction, textureScale ) * light.color;
        }
    }

    for( uint_t i = 0; i < params.numInfiniteLights; ++i )
    {
        const InfiniteLight& light{ params.infiniteLights[i] };
        result +=
            evaluateMdlBsdf( shader, state, resourceData, outgoing, worldNormal, textureScale ) * light.color * light.scale;
    }

    return result;
}

__device__ __forceinline__ bool sampleMdlBsdf( const MdlMaterialShader&                     shader,
                                               const mi::neuraylib::Shading_state_material& state,
                                               const mi::neuraylib::Resource_data&          resourceData,
                                               const float3&                                worldNormal,
                                               const float3&                                outgoing,
                                               const float4&                                xi,
                                               const float3&                                textureScale,
                                               float3&                                      direction,
                                               float3&                                      throughput )
{
    direction       = sampleDiffuseDirection( xi, worldNormal );
    const float pdf = fmaxf( otk::dot( worldNormal, direction ), 0.0f ) * INV_PI;
    if( pdf <= 0.0f )
    {
        return false;
    }
    throughput = evaluateMdlBsdf( shader, state, resourceData, outgoing, direction, textureScale ) / pdf;
    return otk::dot( throughput, throughput ) > 0.0f;
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
    return shader.callableCount >= 1U;
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

    prd->diffuseTextureId = INVALID_TEXTURE_ID;
    prd->material         = nullptr;
    prd->normal           = worldNormal;
    prd->rayDistance      = rayT;
    prd->hasMdlBsdfSample = false;

    if( !useMdlShader( params, materialId, shader ) )
    {
        if( hasMdlDiffuseTexture( material ) )
        {
            setMdlMaterialDiffuseTexturePayload( params, prd, material, worldNormal, vertices, instanceId, rayT );
            return;
        }
        prd->color          = phongShade( material, worldNormal, rayDirection );
        prd->hasDirectColor = true;
        return;
    }

    mi::neuraylib::Shading_state_material state{};
    state.normal      = worldNormal;
    state.geom_normal = worldNormal;
    state.position    = rayOrigin + rayT * rayDirection;

    const bool                hasDiffuseTexture{ hasMdlDiffuseTexture( material ) };
    const bool                useMdlDiffuseTexture{ shader.usesDiffuseTexture && hasDiffuseTexture };
    float2                    uv{};
    float                     worldSpaceTextureSize{};
    mi::neuraylib::tct_float3 textCoords[1]{};
    if( hasDiffuseTexture )
    {
#ifndef NDEBUG
        if( instanceId >= params.numInstanceUVs )
        {
            printf( "Instance id %u exceeds numInstanceUVs %u\n", instanceId, params.numInstanceUVs );
            assert( instanceId < params.numInstanceUVs );
        }
#endif
        const TriangleUVs& triangleUVs{ getMdlTriangleUVArray( params.instanceUVs, instanceId )[optixGetPrimitiveIndex()] };
        uv                    = interpolateMdlUVs( triangleUVs );
        worldSpaceTextureSize = getWorldSpaceTextureSize( vertices, triangleUVs );
        textCoords[0]         = make_float3( uv.x, uv.y, 0.0f );
        state.text_coords     = textCoords;
    }

    mi::neuraylib::Resource_data resourceData{};
    mi::neuraylib::tct_float3    tint{};
    optixDirectCall<void, void*, const mi::neuraylib::Shading_state_material*, const mi::neuraylib::Resource_data*, const char*>(
        shader.callableBaseIndex, &tint, &state, &resourceData, nullptr );

    material.Kd = make_float3( tint.x, tint.y, tint.z );

    prd->materialCopy   = material;
    prd->color          = phongShade( material, worldNormal, rayDirection );
    prd->hasDirectColor = true;

    bool diffuseTextureResident{};
    const float3 diffuseTextureScale{ sampleMdlDiffuseTexture( useMdlDiffuseTexture ? material.diffuseTextureId : INVALID_TEXTURE_ID,
                                                               uv, diffuseTextureResident ) };
    if( useMdlDiffuseTexture && !diffuseTextureResident )
    {
        setMdlDiffuseTexturePayload( prd, material, worldNormal, rayT, material.diffuseTextureId, uv, worldSpaceTextureSize );
        return;
    }

    if( hasMdlBsdfCallables( shader ) && ( !hasDiffuseTexture || useMdlDiffuseTexture ) )
    {
        initializeMdlBsdf( shader, state, resourceData );
        prd->color = displayEncodeMdlColor( shadeMdlBsdf( shader, state, resourceData, worldNormal, rayDirection, diffuseTextureScale ) );
        prd->hasDirectColor = true;
        prd->hasMdlBsdfSample = sampleMdlBsdf( shader, state, resourceData, worldNormal, -rayDirection, prd->mdlBsdfSampleXi,
                                               diffuseTextureScale, prd->mdlBsdfSampleDirection, prd->mdlBsdfSampleThroughput );
        return;
    }

    if( !hasDiffuseTexture )
    {
        return;
    }

    setMdlDiffuseTexturePayload( prd, material, worldNormal, rayT, material.diffuseTextureId, uv, worldSpaceTextureSize );
}

}  // namespace demandPbrtScene
