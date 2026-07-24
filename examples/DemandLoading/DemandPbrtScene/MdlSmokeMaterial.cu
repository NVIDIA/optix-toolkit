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
constexpr uint_t MDL_BSDF_SAMPLE_CALLABLE_OFFSET{ 2U };
constexpr uint_t MDL_BSDF_EVALUATE_CALLABLE_OFFSET{ 3U };
constexpr uint_t MDL_BSDF_CALLABLE_COUNT{ 4U };
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

__device__ __forceinline__ float3 makeMdlTangentU( const float3& normal )
{
    const float3 helper{ fabsf( normal.z ) < 0.999f ? make_float3( 0.0f, 0.0f, 1.0f ) : make_float3( 1.0f, 0.0f, 0.0f ) };
    return otk::normalize( otk::cross( helper, normal ) );
}

__device__ __forceinline__ float3 makeMdlTangentV( const float3& normal, const float3& tangentU )
{
    return otk::normalize( otk::cross( normal, tangentU ) );
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

__device__ __forceinline__ bool hasMdlMaterialTexture( const MdlMaterialShader& shader, uint_t index )
{
    return shader.textureBindingCount > index && shader.textureBindings[index].textureId != INVALID_TEXTURE_ID;
}

__device__ __forceinline__ bool hasMdlDiffuseTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX );
}

__device__ __forceinline__ bool hasMdlMaterialTextures( const MdlMaterialShader& shader )
{
    for( uint_t i = 0; i < shader.textureBindingCount && i < MDL_MATERIAL_TEXTURE_BINDING_COUNT; ++i )
    {
        if( hasMdlMaterialTexture( shader, i ) )
        {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool hasAllocatedDiffuseMap( const PhongMaterial& material )
{
    return ( static_cast<uint_t>( material.flags ) & static_cast<uint_t>( MaterialFlags::DIFFUSE_MAP_ALLOCATED ) ) != 0U;
}

__device__ __forceinline__ float3 displayEncodeMdlColor( const float3& color )
{
    return make_float3( powf( fmaxf( color.x, 0.0f ), DISPLAY_GAMMA ),  //
                        powf( fmaxf( color.y, 0.0f ), DISPLAY_GAMMA ),  //
                        powf( fmaxf( color.z, 0.0f ), DISPLAY_GAMMA ) );
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

struct MdlMaterialTextureSamples
{
    float3 kd;
    float3 ks;
    float3 kr;
};

__device__ __forceinline__ MdlMaterialTextureSamples makeMdlMaterialTextureSamples()
{
    MdlMaterialTextureSamples samples{};
    samples.kd = make_float3( 1.0f );
    samples.ks = make_float3( 1.0f );
    samples.kr = make_float3( 1.0f );
    return samples;
}

__device__ __forceinline__ float3 sampleMdlMaterialTexture( const MdlMaterialShader& shader, uint_t index, const float2& uv, bool& isResident )
{
    isResident = true;
    if( !hasMdlMaterialTexture( shader, index ) )
    {
        return make_float3( 1.0f );
    }

    const Params&                    params{ PARAMS_VAR_NAME };
    const MdlMaterialTextureBinding& binding{ shader.textureBindings[index] };
#ifndef NDEBUG
    if( binding.textureId < params.minDiffuseTextureId || binding.textureId > params.maxDiffuseTextureId )
    {
        printf( "MDL material texture id %u out of range [%u, %u]\n", binding.textureId, params.minDiffuseTextureId,
                params.maxDiffuseTextureId );
        assert( binding.textureId >= params.minDiffuseTextureId && binding.textureId <= params.maxDiffuseTextureId );
    }
#endif
    const float4 texel = demandLoading::tex2D<float4>( params.demandContext, binding.textureId, uv.x, uv.y, &isResident );
    return make_float3( texel.x, texel.y, texel.z ) * binding.scale + binding.bias;
}

__device__ __forceinline__ void setMdlMaterialTextureSample( MdlMaterialTextureSamples& samples, uint_t index, const float3& value )
{
    switch( index )
    {
        case MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX:
            samples.kd = value;
            return;
        case MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX:
            samples.ks = value;
            return;
        case MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX:
            samples.kr = value;
            return;
    }
}

__device__ __forceinline__ bool sampleMdlMaterialTextures( const MdlMaterialShader&   shader,
                                                           const float2&              uv,
                                                           MdlMaterialTextureSamples& samples,
                                                           uint_t&                    nonResidentTextureId )
{
    samples = makeMdlMaterialTextureSamples();
    for( uint_t i = 0; i < shader.textureBindingCount && i < MDL_MATERIAL_TEXTURE_BINDING_COUNT; ++i )
    {
        if( !hasMdlMaterialTexture( shader, i ) )
        {
            continue;
        }
        bool         isResident{};
        const float3 value{ sampleMdlMaterialTexture( shader, i, uv, isResident ) };
        if( !isResident )
        {
            nonResidentTextureId = shader.textureBindings[i].textureId;
            return false;
        }
        setMdlMaterialTextureSample( samples, i, value );
    }
    return true;
}

__device__ __forceinline__ float3 mdlDiffuseTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX ) ? samples.kd : make_float3( 1.0f );
}

__device__ __forceinline__ bool hasMdlGlossyTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX );
}

__device__ __forceinline__ float3 mdlGlossyTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    const bool hasKs{ hasMdlMaterialTexture( shader, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX ) };
    const bool hasKr{ hasMdlMaterialTexture( shader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX ) };
    if( hasKs && hasKr )
    {
        return ( samples.ks + samples.kr ) * 0.5f;
    }
    if( hasKs )
    {
        return samples.ks;
    }
    if( hasKr )
    {
        return samples.kr;
    }
    return make_float3( 1.0f );
}

__device__ __forceinline__ float3 mdlSpecularTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX ) )
    {
        return samples.kr;
    }
    return mdlGlossyTextureScale( shader, samples );
}

__device__ __forceinline__ float mdlRuntimeDiffuseMixCompensation( const MdlMaterialShader& shader )
{
    if( !hasMdlGlossyTexture( shader ) )
    {
        return 1.0f;
    }

    // Demand textures use unit MDL placeholder weights and apply samples after
    // evaluation, so partially compensate the diffuse lobe for placeholder mix
    // normalization when runtime glossy/specular texture slots are active.
    float compensation{ 1.0f };
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX ) )
    {
        compensation += 0.5f;
    }
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX ) )
    {
        compensation += 0.5f;
    }
    return compensation;
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
                                                   const MdlMaterialTextureSamples&             textureSamples )
{
    mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> evalData{};
    evalData.ior1  = make_float3( 1.0f );
    evalData.ior2  = make_float3( MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR );
    evalData.k1    = outgoing;
    evalData.k2    = incoming;
    evalData.flags = mi::neuraylib::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    optixDirectCall<void, mi::neuraylib::Bsdf_evaluate_data_base*, const mi::neuraylib::Shading_state_material*,
                    const mi::neuraylib::Resource_data*, const char*>( shader.callableBaseIndex + MDL_BSDF_EVALUATE_CALLABLE_OFFSET,
                                                                       &evalData, &state, &resourceData, nullptr );
    float3 glossyScale{ mdlGlossyTextureScale( shader, textureSamples ) };
    if( !hasMdlGlossyTexture( shader ) && hasMdlMaterialTexture( shader, MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX )
        && otk::dot( evalData.bsdf_diffuse, evalData.bsdf_diffuse ) == 0.0f )
    {
        glossyScale = textureSamples.kd;
    }
    return evalData.bsdf_diffuse * mdlDiffuseTextureScale( shader, textureSamples ) * mdlRuntimeDiffuseMixCompensation( shader )
           + evalData.bsdf_glossy * glossyScale;
}

__device__ __forceinline__ float3 shadeMdlBsdf( const MdlMaterialShader&                     shader,
                                                const mi::neuraylib::Shading_state_material& state,
                                                const mi::neuraylib::Resource_data&          resourceData,
                                                const float3&                                worldNormal,
                                                const float3&                                rayDirection,
                                                const MdlMaterialTextureSamples&             textureSamples )
{
    float3        result{};
    const float3  outgoing{ -rayDirection };
    const Params& params{ PARAMS_VAR_NAME };

    for( uint_t i = 0; i < params.numDirectionalLights; ++i )
    {
        const DirectionalLight& light{ params.directionalLights[i] };
        if( otk::dot( worldNormal, light.direction ) > 0.0f )
        {
            result += evaluateMdlBsdf( shader, state, resourceData, outgoing, light.direction, textureSamples ) * light.color;
        }
    }

    for( uint_t i = 0; i < params.numInfiniteLights; ++i )
    {
        const InfiniteLight& light{ params.infiniteLights[i] };
        result +=
            evaluateMdlBsdf( shader, state, resourceData, outgoing, worldNormal, textureSamples ) * light.color * light.scale;
    }

    return result;
}

__device__ __forceinline__ bool sampleMdlBsdf( const MdlMaterialShader&                     shader,
                                               const mi::neuraylib::Shading_state_material& state,
                                               const mi::neuraylib::Resource_data&          resourceData,
                                               const float3&                                outgoing,
                                               const float4&                                xi,
                                               const MdlMaterialTextureSamples&             textureSamples,
                                               float3&                                      direction,
                                               float3&                                      throughput )
{
    mi::neuraylib::Bsdf_sample_data sampleData{};
    sampleData.ior1  = make_float3( 1.0f );
    sampleData.ior2  = make_float3( MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR );
    sampleData.k1    = outgoing;
    sampleData.xi    = xi;
    sampleData.flags = mi::neuraylib::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    optixDirectCall<void, mi::neuraylib::Bsdf_sample_data*, const mi::neuraylib::Shading_state_material*, const mi::neuraylib::Resource_data*, const char*>(
        shader.callableBaseIndex + MDL_BSDF_SAMPLE_CALLABLE_OFFSET, &sampleData, &state, &resourceData, nullptr );

    if( sampleData.event_type == mi::neuraylib::BSDF_EVENT_ABSORB )
    {
        return false;
    }

    direction = sampleData.k2;
    const bool diffuseEvent{ ( sampleData.event_type & mi::neuraylib::BSDF_EVENT_DIFFUSE ) != 0 };
    const bool reflectedGlossyEvent{ ( sampleData.event_type & mi::neuraylib::BSDF_EVENT_REFLECTION ) != 0
                                     && ( sampleData.event_type & mi::neuraylib::BSDF_EVENT_GLOSSY ) != 0 };
    const bool reflectedSpecularEvent{ ( sampleData.event_type & mi::neuraylib::BSDF_EVENT_REFLECTION ) != 0
                                       && ( sampleData.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR ) != 0 };
    float3     textureScale{ make_float3( 1.0f ) };
    if( diffuseEvent )
    {
        textureScale = mdlDiffuseTextureScale( shader, textureSamples ) * mdlRuntimeDiffuseMixCompensation( shader );
    }
    else if( reflectedGlossyEvent )
    {
        textureScale = mdlGlossyTextureScale( shader, textureSamples );
        if( !hasMdlGlossyTexture( shader ) && hasMdlMaterialTexture( shader, MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX ) )
        {
            textureScale = textureSamples.kd;
        }
    }
    else if( reflectedSpecularEvent )
    {
        textureScale = mdlSpecularTextureScale( shader, textureSamples );
    }
    throughput = sampleData.bsdf_over_pdf * textureScale;
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
        if( hasAllocatedDiffuseMap( material ) )
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

    mi::neuraylib::Resource_data resourceData{};
    mi::neuraylib::tct_float3    tint{};
    const bool                   hasDiffuseTexture{ hasAllocatedDiffuseMap( material ) };
    const bool                   useMdlDiffuseTexture{ hasMdlDiffuseTexture( shader ) && hasDiffuseTexture };
    const bool                   hasMaterialTextures{ hasMdlMaterialTextures( shader ) };
    float2                       uv{};
    float                        worldSpaceTextureSize{};
    mi::neuraylib::tct_float3    textCoords[1]{};
    mi::neuraylib::tct_float3    tangentU[1]{};
    mi::neuraylib::tct_float3    tangentV[1]{};
    tangentU[0]       = makeMdlTangentU( worldNormal );
    tangentV[0]       = makeMdlTangentV( worldNormal, tangentU[0] );
    state.text_coords = textCoords;
    state.tangent_u   = tangentU;
    state.tangent_v   = tangentV;
    if( hasMaterialTextures || hasDiffuseTexture )
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
    }

    optixDirectCall<void, void*, const mi::neuraylib::Shading_state_material*, const mi::neuraylib::Resource_data*, const char*>(
        shader.callableBaseIndex, &tint, &state, &resourceData, nullptr );

    material.Kd = make_float3( tint.x, tint.y, tint.z );

    prd->materialCopy   = material;
    prd->color          = phongShade( material, worldNormal, rayDirection );
    prd->hasDirectColor = true;

    MdlMaterialTextureSamples textureSamples{};
    uint_t                    nonResidentTextureId{};
    if( !sampleMdlMaterialTextures( shader, uv, textureSamples, nonResidentTextureId ) )
    {
        setMdlDiffuseTexturePayload( prd, material, worldNormal, rayT, nonResidentTextureId, uv, worldSpaceTextureSize );
        return;
    }

    if( hasMdlBsdfCallables( shader ) && ( !hasDiffuseTexture || useMdlDiffuseTexture ) )
    {
        initializeMdlBsdf( shader, state, resourceData );
        prd->color = displayEncodeMdlColor( shadeMdlBsdf( shader, state, resourceData, worldNormal, rayDirection, textureSamples ) );
        prd->hasDirectColor = true;
        prd->hasMdlBsdfSample = PARAMS_VAR_NAME.renderMode == RenderMode::PATH_TRACING
                                && sampleMdlBsdf( shader, state, resourceData, -rayDirection, prd->mdlBsdfSampleXi,
                                                  textureSamples, prd->mdlBsdfSampleDirection, prd->mdlBsdfSampleThroughput );
        return;
    }

    if( !hasAllocatedDiffuseMap( material ) )
    {
        return;
    }

    setMdlMaterialDiffuseTexturePayload( params, prd, material, worldNormal, vertices, instanceId, rayT );
}

}  // namespace demandPbrtScene
