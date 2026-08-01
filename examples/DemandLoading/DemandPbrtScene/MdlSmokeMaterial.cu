// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/DeviceTriangles.h"
#include "DemandPbrtScene/MdlBumpMap.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/PhongShade.h"

#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/ShaderUtil/ray_cone.h>

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

constexpr float DISPLAY_GAMMA{ 1.0f / 2.2f };

__device__ __forceinline__ float3 displayEncodeMdlColor( const float3& color )
{
    return make_float3( powf( fmaxf( color.x, 0.0f ), DISPLAY_GAMMA ), powf( fmaxf( color.y, 0.0f ), DISPLAY_GAMMA ),
                        powf( fmaxf( color.z, 0.0f ), DISPLAY_GAMMA ) );
}

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

__device__ __forceinline__ float2 mdlSphericalUV( const float3& direction )
{
    constexpr float PI{ 3.141592729f };
    float           phi{ atan2f( direction.y, direction.x ) };
    if( phi < 0.0f )
    {
        phi += 2.0f * PI;
    }
    return make_float2( phi / ( 2.0f * PI ), acosf( otk::clamp( direction.z, -1.0f, 1.0f ) ) / PI );
}

__device__ __forceinline__ float3 sampleMdlCosineHemisphere( const float2& xi, const float3& normal, float& cosine )
{
    constexpr float PI{ 3.141592729f };
    const float     radius{ sqrtf( xi.x ) };
    const float     phi{ 2.0f * PI * xi.y };
    const float3    tangentU{ makeMdlTangentU( normal ) };
    const float3    tangentV{ makeMdlTangentV( normal, tangentU ) };
    cosine = sqrtf( fmaxf( 0.0f, 1.0f - xi.x ) );
    return radius * cosf( phi ) * tangentU + radius * sinf( phi ) * tangentV + cosine * normal;
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
    float3 kt;
    float  roughness;
    float  uroughness;
    float  vroughness;
    float3 mixNamedKd[2];
    float3 mixNamedKs[2];
    float3 mixNamedKr[2];
    float  mixNamedAlpha[2];
};

struct MdlBumpMapSamples
{
    float height;
    float heightU;
    float heightV;
};

__device__ __forceinline__ MdlMaterialTextureSamples makeMdlMaterialTextureSamples()
{
    MdlMaterialTextureSamples samples{};
    samples.kd         = make_float3( 1.0f );
    samples.ks         = make_float3( 1.0f );
    samples.kr         = make_float3( 1.0f );
    samples.kt         = make_float3( 1.0f );
    samples.roughness  = 0.1f;
    samples.uroughness = -1.0f;
    samples.vroughness = -1.0f;
    for( uint_t i = 0; i < 2U; ++i )
    {
        samples.mixNamedKd[i]    = make_float3( 1.0f );
        samples.mixNamedKs[i]    = make_float3( 1.0f );
        samples.mixNamedKr[i]    = make_float3( 1.0f );
        samples.mixNamedAlpha[i] = 1.0f;
    }
    return samples;
}

__device__ __forceinline__ float mdlLuminance( const float3& color )
{
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
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

__device__ __forceinline__ float3 sampleMdlMaterialTextureGrad( const MdlMaterialShader& shader,
                                                                uint_t                   index,
                                                                const float2&            uv,
                                                                const float2&            ddx,
                                                                const float2&            ddy,
                                                                bool&                    isResident )
{
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
    const float4 texel{ demandLoading::tex2DGrad<float4>( params.demandContext, binding.textureId, uv.x, uv.y, ddx, ddy, &isResident ) };
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
        case MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX:
            samples.kt = value;
            return;
        case MDL_MATERIAL_ROUGHNESS_TEXTURE_BINDING_INDEX:
            samples.roughness = mdlLuminance( value );
            return;
        case MDL_MATERIAL_UROUGHNESS_TEXTURE_BINDING_INDEX:
            samples.uroughness = mdlLuminance( value );
            return;
        case MDL_MATERIAL_VROUGHNESS_TEXTURE_BINDING_INDEX:
            samples.vroughness = mdlLuminance( value );
            return;
        case MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX:
            samples.mixNamedKd[0] = value;
            return;
        case MDL_MATERIAL_MIX_NAMED_0_KS_TEXTURE_BINDING_INDEX:
            samples.mixNamedKs[0] = value;
            return;
        case MDL_MATERIAL_MIX_NAMED_0_KR_TEXTURE_BINDING_INDEX:
            samples.mixNamedKr[0] = value;
            return;
        case MDL_MATERIAL_MIX_NAMED_0_ALPHA_TEXTURE_BINDING_INDEX:
            samples.mixNamedAlpha[0] = mdlLuminance( value );
            return;
        case MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX:
            samples.mixNamedKd[1] = value;
            return;
        case MDL_MATERIAL_MIX_NAMED_1_KS_TEXTURE_BINDING_INDEX:
            samples.mixNamedKs[1] = value;
            return;
        case MDL_MATERIAL_MIX_NAMED_1_KR_TEXTURE_BINDING_INDEX:
            samples.mixNamedKr[1] = value;
            return;
        case MDL_MATERIAL_MIX_NAMED_1_ALPHA_TEXTURE_BINDING_INDEX:
            samples.mixNamedAlpha[1] = mdlLuminance( value );
            return;
    }
}

__device__ __forceinline__ bool isMdlBumpMapTextureBinding( uint_t index )
{
    return index == MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX || index == MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX
           || index == MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX;
}

__device__ __forceinline__ bool hasMdlBumpMapTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX );
}

__device__ __forceinline__ bool sampleMdlMaterialTextures( const MdlMaterialShader&   shader,
                                                           const float2&              uv,
                                                           MdlMaterialTextureSamples& samples,
                                                           uint_t&                    nonResidentTextureId )
{
    samples = makeMdlMaterialTextureSamples();
    for( uint_t i = 0; i < shader.textureBindingCount && i < MDL_MATERIAL_TEXTURE_BINDING_COUNT; ++i )
    {
        if( isMdlBumpMapTextureBinding( i ) || !hasMdlMaterialTexture( shader, i ) )
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

__device__ __forceinline__ bool sampleMdlBumpMapTexture( const MdlMaterialShader& shader,
                                                         uint_t                   index,
                                                         const float2&            uv,
                                                         const float2&            ddx,
                                                         const float2&            ddy,
                                                         float                    du,
                                                         float                    dv,
                                                         MdlBumpMapSamples&       samples,
                                                         uint_t&                  nonResidentTextureId )
{
    bool resident{};
    bool allResident{ true };
    samples.height = mdlLuminance( sampleMdlMaterialTextureGrad( shader, index, uv, ddx, ddy, resident ) );
    allResident    = allResident && resident;
    samples.heightU =
        mdlLuminance( sampleMdlMaterialTextureGrad( shader, index, uv + make_float2( du, 0.0f ), ddx, ddy, resident ) );
    allResident = allResident && resident;
    samples.heightV =
        mdlLuminance( sampleMdlMaterialTextureGrad( shader, index, uv + make_float2( 0.0f, dv ), ddx, ddy, resident ) );
    allResident = allResident && resident;
    if( !allResident )
    {
        nonResidentTextureId = shader.textureBindings[index].textureId;
    }
    return allResident;
}

__device__ __forceinline__ bool sampleMdlBumpMap( const MdlMaterialShader& shader,
                                                  const float2&            uv,
                                                  const float2&            ddx,
                                                  const float2&            ddy,
                                                  float                    du,
                                                  float                    dv,
                                                  MdlBumpMapSamples&       samples,
                                                  uint_t&                  nonResidentTextureId )
{
    const uint_t bumpMapIndices[] = {
        MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX,
        MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX,
        MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX,
    };
    uint_t numSamples{};
    for( uint_t index : bumpMapIndices )
    {
        if( !hasMdlMaterialTexture( shader, index ) )
        {
            continue;
        }
        MdlBumpMapSamples current{};
        if( !sampleMdlBumpMapTexture( shader, index, uv, ddx, ddy, du, dv, current, nonResidentTextureId ) )
        {
            return false;
        }
        samples.height += current.height;
        samples.heightU += current.heightU;
        samples.heightV += current.heightV;
        ++numSamples;
    }
    if( numSamples > 1U )
    {
        const float scale{ 1.0f / static_cast<float>( numSamples ) };
        samples.height *= scale;
        samples.heightU *= scale;
        samples.heightV *= scale;
    }
    return true;
}

__device__ __forceinline__ bool hasMdlMixNamedKdTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX );
}

__device__ __forceinline__ bool hasMdlMixNamedKsTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_0_KS_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_1_KS_TEXTURE_BINDING_INDEX );
}

__device__ __forceinline__ bool hasMdlMixNamedKrTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_0_KR_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_1_KR_TEXTURE_BINDING_INDEX );
}

__device__ __forceinline__ float3 mdlAverageOptionalTexturePair( bool hasFirst, const float3& first, bool hasSecond, const float3& second )
{
    if( hasFirst && hasSecond )
    {
        return ( first + second ) * 0.5f;
    }
    if( hasFirst )
    {
        return first;
    }
    if( hasSecond )
    {
        return second;
    }
    return make_float3( 1.0f );
}

__device__ __forceinline__ float3 mdlMixNamedKdTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    return mdlAverageOptionalTexturePair( hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX ),
                                          samples.mixNamedKd[0] * samples.mixNamedAlpha[0],
                                          hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX ),
                                          samples.mixNamedKd[1] * samples.mixNamedAlpha[1] );
}

__device__ __forceinline__ float3 mdlMixNamedKsTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    return mdlAverageOptionalTexturePair( hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_0_KS_TEXTURE_BINDING_INDEX ),
                                          samples.mixNamedKs[0] * samples.mixNamedAlpha[0],
                                          hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_1_KS_TEXTURE_BINDING_INDEX ),
                                          samples.mixNamedKs[1] * samples.mixNamedAlpha[1] );
}

__device__ __forceinline__ float3 mdlMixNamedKrTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    return mdlAverageOptionalTexturePair( hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_0_KR_TEXTURE_BINDING_INDEX ),
                                          samples.mixNamedKr[0] * samples.mixNamedAlpha[0],
                                          hasMdlMaterialTexture( shader, MDL_MATERIAL_MIX_NAMED_1_KR_TEXTURE_BINDING_INDEX ),
                                          samples.mixNamedKr[1] * samples.mixNamedAlpha[1] );
}

__device__ __forceinline__ float3 mdlDiffuseTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX ) )
    {
        return samples.kd;
    }
    return hasMdlMixNamedKdTexture( shader ) ? mdlMixNamedKdTextureScale( shader, samples ) : make_float3( 1.0f );
}

__device__ __forceinline__ bool hasMdlGlossyTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX )
           || hasMdlMixNamedKsTexture( shader ) || hasMdlMixNamedKrTexture( shader );
}

__device__ __forceinline__ float3 mdlGlossyTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    const bool hasKs{ hasMdlMaterialTexture( shader, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX ) };
    const bool hasKr{ hasMdlMaterialTexture( shader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX ) };
    if( hasKs || hasKr )
    {
        return mdlAverageOptionalTexturePair( hasKs, samples.ks, hasKr, samples.kr );
    }
    const bool hasNamedKs{ hasMdlMixNamedKsTexture( shader ) };
    const bool hasNamedKr{ hasMdlMixNamedKrTexture( shader ) };
    return mdlAverageOptionalTexturePair( hasNamedKs, mdlMixNamedKsTextureScale( shader, samples ), hasNamedKr,
                                          mdlMixNamedKrTextureScale( shader, samples ) );
}

__device__ __forceinline__ float3 mdlSpecularTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX ) )
    {
        return samples.kr;
    }
    if( hasMdlMixNamedKrTexture( shader ) )
    {
        return mdlMixNamedKrTextureScale( shader, samples );
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
    if( hasMdlMixNamedKsTexture( shader ) )
    {
        compensation += 0.5f;
    }
    if( hasMdlMixNamedKrTexture( shader ) )
    {
        compensation += 0.5f;
    }
    return compensation;
}

__device__ __forceinline__ float3 mdlTransmissionTextureScale( const MdlMaterialShader& shader, const MdlMaterialTextureSamples& samples )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX ) ? samples.kt : make_float3( 1.0f );
}

__device__ __forceinline__ bool hasMdlRoughnessTexture( const MdlMaterialShader& shader )
{
    return hasMdlMaterialTexture( shader, MDL_MATERIAL_ROUGHNESS_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_UROUGHNESS_TEXTURE_BINDING_INDEX )
           || hasMdlMaterialTexture( shader, MDL_MATERIAL_VROUGHNESS_TEXTURE_BINDING_INDEX );
}

__device__ __forceinline__ void copyMdlArgumentBlock( char* dst, const char* src, uint_t size )
{
    for( uint_t i = 0; i < size; ++i )
    {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void writeMdlArgumentBlockFloat( char* data, uint_t offset, float value )
{
    *reinterpret_cast<float*>( data + offset ) = value;
}

__device__ __forceinline__ const char* makeMdlBsdfArgumentBlock( const MdlMaterialShader&         shader,
                                                                 const MdlMaterialTextureSamples& samples,
                                                                 char*                            storage )
{
    const char* const argumentBlock{ reinterpret_cast<const char*>( shader.bsdfArgumentBlock ) };
    if( !hasMdlRoughnessTexture( shader ) || shader.bsdfArgumentBlock == 0U || shader.bsdfArgumentBlockSize == 0U
        || shader.bsdfArgumentBlockSize > MDL_MATERIAL_ARGUMENT_BLOCK_STACK_SIZE )
    {
        return argumentBlock;
    }

    copyMdlArgumentBlock( storage, argumentBlock, shader.bsdfArgumentBlockSize );
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_ROUGHNESS_TEXTURE_BINDING_INDEX )
        && shader.roughnessArgumentBlockOffset != INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
    {
        writeMdlArgumentBlockFloat( storage, shader.roughnessArgumentBlockOffset, samples.roughness );
    }
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_UROUGHNESS_TEXTURE_BINDING_INDEX )
        && shader.uRoughnessArgumentBlockOffset != INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
    {
        writeMdlArgumentBlockFloat( storage, shader.uRoughnessArgumentBlockOffset, samples.uroughness );
    }
    if( hasMdlMaterialTexture( shader, MDL_MATERIAL_VROUGHNESS_TEXTURE_BINDING_INDEX )
        && shader.vRoughnessArgumentBlockOffset != INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
    {
        writeMdlArgumentBlockFloat( storage, shader.vRoughnessArgumentBlockOffset, samples.vroughness );
    }
    return storage;
}

__device__ __forceinline__ void initializeMdlBsdf( const MdlMaterialShader&               shader,
                                                   mi::neuraylib::Shading_state_material& state,
                                                   const mi::neuraylib::Resource_data&    resourceData,
                                                   const char*                            argumentBlock )
{
    optixDirectCall<void, mi::neuraylib::Shading_state_material*, const mi::neuraylib::Resource_data*, const char*>(
        shader.callableBaseIndex + MDL_BSDF_INIT_CALLABLE_OFFSET, &state, &resourceData, argumentBlock );
}

__device__ __forceinline__ float3 evaluateMdlBsdf( const MdlMaterialShader&                     shader,
                                                   const mi::neuraylib::Shading_state_material& state,
                                                   const mi::neuraylib::Resource_data&          resourceData,
                                                   const float3&                                outgoing,
                                                   const float3&                                incoming,
                                                   const MdlMaterialTextureSamples&             textureSamples,
                                                   const char*                                  argumentBlock )
{
    mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> evalData{};
    evalData.ior1  = make_float3( 1.0f );
    evalData.ior2  = make_float3( MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR );
    evalData.k1    = outgoing;
    evalData.k2    = incoming;
    evalData.flags = mi::neuraylib::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    optixDirectCall<void, mi::neuraylib::Bsdf_evaluate_data_base*, const mi::neuraylib::Shading_state_material*,
                    const mi::neuraylib::Resource_data*, const char*>( shader.callableBaseIndex + MDL_BSDF_EVALUATE_CALLABLE_OFFSET,
                                                                       &evalData, &state, &resourceData, argumentBlock );
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
                                                const MdlMaterialTextureSamples&             textureSamples,
                                                const float2&                                environmentXi,
                                                const char*                                  argumentBlock )
{
    constexpr float PI{ 3.141592729f };
    float3        result{};
    const float3  outgoing{ -rayDirection };
    const Params& params{ PARAMS_VAR_NAME };

    for( uint_t i = 0; i < params.numDirectionalLights; ++i )
    {
        const DirectionalLight& light{ params.directionalLights[i] };
        result += evaluateMdlBsdf( shader, state, resourceData, outgoing, light.direction, textureSamples, argumentBlock )
                  * light.color;
    }
    for( uint_t i = 0; i < params.numInfiniteLights; ++i )
    {
        const InfiniteLight& light{ params.infiniteLights[i] };
        float                cosine{};
        const float3         incoming{ sampleMdlCosineHemisphere( environmentXi, worldNormal, cosine ) };
        float3               radiance{ light.color * light.scale };
        if( light.skyboxTextureId != 0U )
        {
            bool         isResident{};
            const float2 environmentUV{ mdlSphericalUV( incoming ) };
            const float4 texel{ demandLoading::tex2D<float4>( params.demandContext, light.skyboxTextureId - 1U,
                                                               environmentUV.x, environmentUV.y, &isResident ) };
            if( !isResident )
            {
                continue;
            }
            radiance *= make_float3( texel.x, texel.y, texel.z );
        }
        result += evaluateMdlBsdf( shader, state, resourceData, outgoing, incoming, textureSamples, argumentBlock ) * radiance
                  * ( PI / fmaxf( cosine, 1.0e-6f ) );
    }
    return result;
}

__device__ __forceinline__ bool sampleMdlBsdf( const MdlMaterialShader&                     shader,
                                               const mi::neuraylib::Shading_state_material& state,
                                               const mi::neuraylib::Resource_data&          resourceData,
                                               const float3&                                outgoing,
                                               const float4&                                xi,
                                               const MdlMaterialTextureSamples&             textureSamples,
                                               const char*                                  argumentBlock,
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
        shader.callableBaseIndex + MDL_BSDF_SAMPLE_CALLABLE_OFFSET, &sampleData, &state, &resourceData, argumentBlock );

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
    const bool transmittedEvent{ ( sampleData.event_type & mi::neuraylib::BSDF_EVENT_TRANSMISSION ) != 0 };
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
    else if( transmittedEvent )
    {
        textureScale = mdlTransmissionTextureScale( shader, textureSamples );
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
    if( materialId >= params.numMdlMaterialShaders )
    {
        printf( "Material id %u exceeds number of MDL material shader entries %u\n", materialId, params.numMdlMaterialShaders );
        assert( materialId < params.numMdlMaterialShaders );
    }
#endif
    if( materialId >= params.numMdlMaterialShaders || params.mdlMaterialShaders == nullptr )
    {
        return false;
    }

    shader = params.mdlMaterialShaders[materialId];
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
    const TriangleUVs*           triangleUVs{};
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
        triangleUVs              = &getMdlTriangleUVArray( params.instanceUVs, instanceId )[optixGetPrimitiveIndex()];
        uv                       = interpolateMdlUVs( *triangleUVs );
        worldSpaceTextureSize    = getWorldSpaceTextureSize( vertices, *triangleUVs );
        textCoords[0]         = make_float3( uv.x, uv.y, 0.0f );
    }
    if( params.renderMode == RenderMode::PATH_TRACING || !hasDiffuseTexture )
    {
        optixDirectCall<void, void*, const mi::neuraylib::Shading_state_material*, const mi::neuraylib::Resource_data*, const char*>(
            shader.callableBaseIndex, &tint, &state, &resourceData,
            reinterpret_cast<const char*>( shader.tintArgumentBlock ) );
        material.Kd = make_float3( tint.x, tint.y, tint.z );
    }

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

    float3 shadingNormal{ worldNormal };
    if( hasMdlBumpMapTexture( shader ) )
    {
#ifndef NDEBUG
        assert( triangleUVs != nullptr );
#endif
        float3 worldVertices[3];
        float3 worldNormals[3] = { worldNormal, worldNormal, worldNormal };
        float2 adjustedUVs[3];
        for( uint_t i = 0; i < 3U; ++i )
        {
            worldVertices[i] = optixTransformPointFromObjectToWorldSpace( vertices[i] );
            adjustedUVs[i]   = adjustMdlUV( triangleUVs->UV[i] );
        }
        if( params.instanceNormals != nullptr && params.instanceNormals[instanceId] != nullptr )
        {
            const TriangleNormals& normals{ params.instanceNormals[instanceId][optixGetPrimitiveIndex()] };
            for( uint_t i = 0; i < 3U; ++i )
            {
                worldNormals[i] = otk::normalize( optixTransformNormalFromObjectToWorldSpace( normals.N[i] ) );
                if( params.useFaceForward && optixIsBackFaceHit() )
                {
                    worldNormals[i] = -worldNormals[i];
                }
            }
        }

        const MdlBumpDifferentialGeometry geometry{ makeMdlBumpDifferentialGeometry( worldVertices, adjustedUVs, worldNormals ) };
        float3                            dPdx{};
        float3                            dPdy{};
        const float rayConeWidth{ fabsf( prd->mdlRayConeWidth + prd->mdlRayConeAngle * rayT ) };
        projectToRayDifferentialsOnSurface( rayConeWidth, rayDirection, worldNormal, dPdx, dPdy );
        float2 ddx{};
        float2 ddy{};
        computeTexGradientsForTriangle( worldVertices[0], worldVertices[1], worldVertices[2], adjustedUVs[0], adjustedUVs[1],
                                        adjustedUVs[2], dPdx, dPdy, ddx, ddy );
        const float       du{ mdlBumpOffset( ddx.x, ddy.x ) };
        const float       dv{ mdlBumpOffset( ddx.y, ddy.y ) };
        MdlBumpMapSamples bumpSamples{};
        if( !sampleMdlBumpMap( shader, uv, ddx, ddy, du, dv, bumpSamples, nonResidentTextureId ) )
        {
            setMdlDiffuseTexturePayload( prd, material, worldNormal, rayT, nonResidentTextureId, uv, worldSpaceTextureSize );
            return;
        }
        const MdlBumpDifferentialGeometry bumpedGeometry{
            applyMdlBumpMap( geometry, worldNormal, bumpSamples.height, bumpSamples.heightU, bumpSamples.heightV, du, dv ) };
        shadingNormal = mdlBumpNormal( bumpedGeometry, worldNormal );
        tangentU[0]   = otk::normalize( bumpedGeometry.dpdu );
        tangentV[0]   = otk::normalize( bumpedGeometry.dpdv );
    }
    state.normal = shadingNormal;
    prd->normal  = shadingNormal;
    prd->color   = phongShade( material, shadingNormal, rayDirection );

    if( hasMdlBsdfCallables( shader ) && ( !hasDiffuseTexture || useMdlDiffuseTexture ) )
    {
        alignas( 16 ) char bsdfArgumentBlockStorage[MDL_MATERIAL_ARGUMENT_BLOCK_STACK_SIZE];
        const char* const bsdfArgumentBlock{ makeMdlBsdfArgumentBlock( shader, textureSamples, bsdfArgumentBlockStorage ) };
        initializeMdlBsdf( shader, state, resourceData, bsdfArgumentBlock );
        prd->color = displayEncodeMdlColor( shadeMdlBsdf( shader, state, resourceData, shadingNormal, rayDirection,
                                                          textureSamples,
                                                          make_float2( prd->mdlBsdfSampleXi.z, prd->mdlBsdfSampleXi.w ),
                                                          bsdfArgumentBlock ) );
        prd->hasDirectColor = true;
        prd->hasMdlBsdfSample =
            PARAMS_VAR_NAME.renderMode == RenderMode::PATH_TRACING
            && sampleMdlBsdf( shader, state, resourceData, -rayDirection, prd->mdlBsdfSampleXi, textureSamples,
                              bsdfArgumentBlock, prd->mdlBsdfSampleDirection, prd->mdlBsdfSampleThroughput );
        return;
    }

    if( !hasAllocatedDiffuseMap( material ) )
    {
        return;
    }

    setMdlMaterialDiffuseTexturePayload( params, prd, material, shadingNormal, vertices, instanceId, rayT );
}

}  // namespace demandPbrtScene
