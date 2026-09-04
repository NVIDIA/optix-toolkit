// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"
#include "DemandPbrtScene/RenderMode.h"

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>
#include <OptiXToolkit/ShaderUtil/DebugLocation.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>
#include <vector_types.h>

namespace demandPbrtScene {

using uint_t = unsigned int;

constexpr uint_t INVALID_TEXTURE_ID{ 0xffffffffU };
#ifdef OTK_USE_MDL
constexpr uint_t INVALID_FOURIER_BSDF_TABLE_RESOURCE_ID{ 0U };
constexpr uint_t INVALID_MDL_ARGUMENT_BLOCK_OFFSET{ 0xffffffffU };
constexpr uint_t MDL_MATERIAL_ARGUMENT_BLOCK_STACK_SIZE{ 512U };
constexpr uint_t MDL_MATERIAL_TEXTURE_BINDING_COUNT{ 19U };
constexpr uint_t MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX{ 0U };
constexpr uint_t MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX{ 1U };
constexpr uint_t MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX{ 2U };
constexpr uint_t MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX{ 3U };
constexpr uint_t MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX{ MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_TEXTURE_BINDING_COUNT{ 5U };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE{ 4U };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_1_TEXTURE_BINDING_BASE{ MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE
                                                                + MDL_MATERIAL_MIX_NAMED_TEXTURE_BINDING_COUNT };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_KD_TEXTURE_BINDING_OFFSET{ 0U };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_KS_TEXTURE_BINDING_OFFSET{ 1U };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_KR_TEXTURE_BINDING_OFFSET{ 2U };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_ALPHA_TEXTURE_BINDING_OFFSET{ 3U };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_BUMPMAP_TEXTURE_BINDING_OFFSET{ 4U };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX{ MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE
                                                                    + MDL_MATERIAL_MIX_NAMED_KD_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_0_KS_TEXTURE_BINDING_INDEX{ MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE
                                                                    + MDL_MATERIAL_MIX_NAMED_KS_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_0_KR_TEXTURE_BINDING_INDEX{ MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE
                                                                    + MDL_MATERIAL_MIX_NAMED_KR_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_0_ALPHA_TEXTURE_BINDING_INDEX{
    MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE + MDL_MATERIAL_MIX_NAMED_ALPHA_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX{
    MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE + MDL_MATERIAL_MIX_NAMED_BUMPMAP_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX{ MDL_MATERIAL_MIX_NAMED_1_TEXTURE_BINDING_BASE
                                                                    + MDL_MATERIAL_MIX_NAMED_KD_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_1_KS_TEXTURE_BINDING_INDEX{ MDL_MATERIAL_MIX_NAMED_1_TEXTURE_BINDING_BASE
                                                                    + MDL_MATERIAL_MIX_NAMED_KS_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_1_KR_TEXTURE_BINDING_INDEX{ MDL_MATERIAL_MIX_NAMED_1_TEXTURE_BINDING_BASE
                                                                    + MDL_MATERIAL_MIX_NAMED_KR_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_1_ALPHA_TEXTURE_BINDING_INDEX{
    MDL_MATERIAL_MIX_NAMED_1_TEXTURE_BINDING_BASE + MDL_MATERIAL_MIX_NAMED_ALPHA_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX{
    MDL_MATERIAL_MIX_NAMED_1_TEXTURE_BINDING_BASE + MDL_MATERIAL_MIX_NAMED_BUMPMAP_TEXTURE_BINDING_OFFSET };
constexpr uint_t MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX{ 14U };
constexpr uint_t MDL_MATERIAL_ROUGHNESS_TEXTURE_BINDING_INDEX{ 15U };
constexpr uint_t MDL_MATERIAL_UROUGHNESS_TEXTURE_BINDING_INDEX{ 16U };
constexpr uint_t MDL_MATERIAL_VROUGHNESS_TEXTURE_BINDING_INDEX{ 17U };
constexpr uint_t MDL_MATERIAL_MIX_AMOUNT_TEXTURE_BINDING_INDEX{ 18U };
#endif

enum RayType
{
    RAYTYPE_RADIANCE = 0,
    RAYTYPE_COUNT
};

enum class ProgramGroupIndex : uint_t
{
    RAYGEN                                 = 0,
    MISS                                   = 1,
    HITGROUP_START                         = 2,
    HITGROUP_PROXY_GEOMETRY                = HITGROUP_START,
    HITGROUP_PROXY_MATERIAL_TRIANGLE       = 3,
    HITGROUP_PROXY_MATERIAL_TRIANGLE_ALPHA = 4,
    HITGROUP_PROXY_MATERIAL_SPHERE         = 5,
    HITGROUP_PROXY_MATERIAL_SPHERE_ALPHA   = 6,
    HITGROUP_REALIZED_MATERIAL_START       = 7,
    NUM_STATIC_PROGRAM_GROUPS              = HITGROUP_REALIZED_MATERIAL_START,
};

// Least noisy way to get uint_t for the enum
constexpr uint_t operator+( ProgramGroupIndex value )
{
    return static_cast<uint_t>( value );
}

enum class HitGroupIndex : uint_t
{
    PROXY_GEOMETRY                = 0,
    PROXY_MATERIAL_TRIANGLE       = 1,
    PROXY_MATERIAL_TRIANGLE_ALPHA = 2,
    PROXY_MATERIAL_SPHERE         = 3,
    PROXY_MATERIAL_SPHERE_ALPHA   = 4,
    REALIZED_MATERIAL_START       = 5,
};

// Least noisy way to get uint_t for the enum
inline uint_t operator+( HitGroupIndex value )
{
    return static_cast<uint_t>( value );
}

#ifdef OTK_USE_MDL
struct MdlMaterialTextureBinding
{
    uint_t textureId;
    float3 scale;
    float3 bias;
};

inline bool operator==( const MdlMaterialTextureBinding& lhs, const MdlMaterialTextureBinding& rhs )
{
    return lhs.textureId == rhs.textureId && lhs.scale == rhs.scale && lhs.bias == rhs.bias;
}

inline bool operator!=( const MdlMaterialTextureBinding& lhs, const MdlMaterialTextureBinding& rhs )
{
    return !( lhs == rhs );
}

__host__ __device__ inline MdlMaterialTextureBinding invalidMdlMaterialTextureBinding()
{
    return MdlMaterialTextureBinding{ INVALID_TEXTURE_ID, make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) };
}
#endif

#ifdef OTK_USE_MDL
struct FourierBsdfTableDeviceData
{
    int         flags;
    int         nMu;
    int         nCoefficients;
    int         maxOrder;
    int         nChannels;
    int         nBases;
    float       eta;
    uint_t      trailingByteCount;
    uint_t      gridSize;
    CUdeviceptr mu;
    CUdeviceptr cdf;
    CUdeviceptr coefficientOffsets;
    CUdeviceptr coefficientCounts;
    CUdeviceptr zeroOrderCoefficients;
    CUdeviceptr coefficients;
};

struct FourierMaterialResource
{
    uint_t      resourceId;
    CUdeviceptr table;
};

__host__ __device__ inline bool hasFourierBsdfTableResource( const FourierMaterialResource& resource )
{
    return resource.resourceId != INVALID_FOURIER_BSDF_TABLE_RESOURCE_ID && resource.table != CUdeviceptr{};
}

__host__ __device__ inline FourierMaterialResource makeFourierMaterialResource( uint_t resourceId, CUdeviceptr table )
{
    return FourierMaterialResource{ resourceId, table };
}

__host__ __device__ inline bool operator==( const FourierMaterialResource& lhs, const FourierMaterialResource& rhs )
{
    return lhs.resourceId == rhs.resourceId && lhs.table == rhs.table;
}

__host__ __device__ inline bool operator!=( const FourierMaterialResource& lhs, const FourierMaterialResource& rhs )
{
    return !( lhs == rhs );
}
#endif

struct DirectionalLight
{
    float3 direction;
    float3 color;
};

inline bool operator==( const DirectionalLight& lhs, const DirectionalLight& rhs )
{
    return lhs.direction == rhs.direction && lhs.color == rhs.color;
}
inline bool operator!=( const DirectionalLight& lhs, const DirectionalLight& rhs )
{
    return !( lhs == rhs );
}

struct InfiniteLight
{
    float3 color;            // color of the light
    float3 scale;            // scaling factor applied to value from texture
    uint_t skyboxTextureId;  // one greater than the texture ID, or zero if there is no skybox texture
};

inline bool operator==( const InfiniteLight& lhs, const InfiniteLight& rhs )
{
    return lhs.color == rhs.color     //
           && lhs.scale == rhs.scale  //
           && lhs.skyboxTextureId == rhs.skyboxTextureId;
}
inline bool operator!=( const InfiniteLight& lhs, const InfiniteLight& rhs )
{
    return !( lhs == rhs );
}

enum class MaterialFlags : uint_t
{
    NONE                  = 0,
    ALPHA_MAP             = 1,
    DIFFUSE_MAP           = 2,
    ALPHA_MAP_ALLOCATED   = 4,
    DIFFUSE_MAP_ALLOCATED = 8,
    MASK                  = 0xF,
};

// least noisy way to convert to uint_t
inline uint_t operator+( MaterialFlags value )
{
    return static_cast<uint_t>( value );
}

// bit operators for flags enum
inline MaterialFlags operator|( MaterialFlags lhs, MaterialFlags rhs )
{
    return static_cast<MaterialFlags>( +lhs | +rhs );
}
inline MaterialFlags& operator|=( MaterialFlags& lhs, MaterialFlags rhs )
{
    lhs = lhs | rhs;
    return lhs;
}
inline MaterialFlags operator&( MaterialFlags lhs, MaterialFlags rhs )
{
    return static_cast<MaterialFlags>( +lhs & +rhs );
}
inline MaterialFlags& operator&=( MaterialFlags& lhs, MaterialFlags rhs )
{
    lhs = lhs & rhs;
    return lhs;
}
inline MaterialFlags operator^( MaterialFlags lhs, MaterialFlags rhs )
{
    return static_cast<MaterialFlags>( +lhs ^ +rhs );
}
inline MaterialFlags& operator^=( MaterialFlags& lhs, MaterialFlags rhs )
{
    lhs = lhs ^ rhs;
    return lhs;
}
inline MaterialFlags operator~( MaterialFlags value )
{
    return static_cast<MaterialFlags>( ~( +value ) & +MaterialFlags::MASK );
}
inline bool flagSet( MaterialFlags value, MaterialFlags flag )
{
    return ( value & flag ) == flag;
}

struct PartialMaterial
{
    uint_t alphaTextureId;
};
inline bool operator==( const PartialMaterial& lhs, const PartialMaterial& rhs )
{
    return lhs.alphaTextureId == rhs.alphaTextureId;
}
inline bool operator!=( const PartialMaterial& lhs, const PartialMaterial& rhs )
{
    return !( lhs == rhs );
}

enum class MaterialBackend : uint_t
{
    NONE                = 0,
    LOCAL_FALLBACK      = 1,
    MDL_READY           = 2,
    MDL_PENDING         = 3,
    MDL_FAILED          = 4,
    FOURIER_TABLE_READY = 5,
};

inline uint_t operator+( MaterialBackend value )
{
    return static_cast<uint_t>( value );
}

enum class MaterialFallbackReason : uint_t
{
    NONE           = 0,
    NO_MDL_BACKEND = 1,
    MDL_PENDING    = 2,
    MDL_FAILED     = 3,
    UNSUPPORTED    = 4,
};

inline uint_t operator+( MaterialFallbackReason value )
{
    return static_cast<uint_t>( value );
}

struct MaterialState
{
    uint_t                 materialId;
    MaterialBackend        backend;
    uint_t                 shaderKey;
    MaterialFallbackReason fallbackReason;
};

inline bool operator==( const MaterialState& lhs, const MaterialState& rhs )
{
    return lhs.materialId == rhs.materialId  //
           && lhs.backend == rhs.backend     //
           && lhs.shaderKey == rhs.shaderKey && lhs.fallbackReason == rhs.fallbackReason;
}

inline bool operator!=( const MaterialState& lhs, const MaterialState& rhs )
{
    return !( lhs == rhs );
}

inline MaterialFallbackReason defaultFallbackReason( MaterialBackend backend )
{
    switch( backend )
    {
        case MaterialBackend::NONE:
            return MaterialFallbackReason::UNSUPPORTED;
        case MaterialBackend::LOCAL_FALLBACK:
            return MaterialFallbackReason::NO_MDL_BACKEND;
        case MaterialBackend::MDL_READY:
            return MaterialFallbackReason::NONE;
        case MaterialBackend::MDL_PENDING:
            return MaterialFallbackReason::MDL_PENDING;
        case MaterialBackend::MDL_FAILED:
            return MaterialFallbackReason::MDL_FAILED;
        case MaterialBackend::FOURIER_TABLE_READY:
            return MaterialFallbackReason::NONE;
    }
    return MaterialFallbackReason::UNSUPPORTED;
}

inline MaterialState makeMaterialState( uint_t                 materialId,
                                        MaterialBackend        backend,
                                        uint_t                 shaderKey      = 0U,
                                        MaterialFallbackReason fallbackReason = MaterialFallbackReason::NONE )
{
    if( fallbackReason == MaterialFallbackReason::NONE )
    {
        fallbackReason = defaultFallbackReason( backend );
    }
    return MaterialState{ materialId, backend, shaderKey, fallbackReason };
}

inline bool usesFallbackShader( const MaterialState& state )
{
    return state.backend != MaterialBackend::MDL_READY && state.backend != MaterialBackend::FOURIER_TABLE_READY;
}

#ifdef OTK_USE_MDL
struct MdlMaterialShader
{
    uint_t      callableBaseIndex;
    uint_t      callableCount;
    CUdeviceptr tintArgumentBlock;
    CUdeviceptr bsdfArgumentBlock;
    uint_t      bsdfArgumentBlockSize;
    uint_t      roughnessArgumentBlockOffset;
    uint_t      uRoughnessArgumentBlockOffset;
    uint_t      vRoughnessArgumentBlockOffset;
    uint_t      mixAmountArgumentBlockOffset;

    // Per-instance shader data lives here rather than in hitgroup SBT records.
    uint_t                    textureBindingCount;
    MdlMaterialTextureBinding textureBindings[MDL_MATERIAL_TEXTURE_BINDING_COUNT];

    __host__ __device__ void clearTextureBindings()
    {
        textureBindingCount = 0U;
        for( uint_t i = 0; i < MDL_MATERIAL_TEXTURE_BINDING_COUNT; ++i )
        {
            textureBindings[i] = invalidMdlMaterialTextureBinding();
        }
    }

    __host__ __device__ bool setTextureBinding( uint_t index, uint_t textureId, const float3& scale, const float3& bias )
    {
        if( index >= MDL_MATERIAL_TEXTURE_BINDING_COUNT )
        {
            return false;
        }
        textureBindings[index] = MdlMaterialTextureBinding{ textureId, scale, bias };
        if( textureId != INVALID_TEXTURE_ID && textureBindingCount <= index )
        {
            textureBindingCount = index + 1U;
        }
        return true;
    }

    __host__ __device__ MdlMaterialShader()
        : callableBaseIndex( 0U )
        , callableCount( 0U )
        , tintArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlockSize( 0U )
        , roughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , uRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , vRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , mixAmountArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , textureBindingCount( 0U )
    {
        clearTextureBindings();
    }

    __host__ __device__ MdlMaterialShader( uint_t callableBaseIndex_, uint_t callableCount_ )
        : callableBaseIndex( callableBaseIndex_ )
        , callableCount( callableCount_ )
        , tintArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlockSize( 0U )
        , roughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , uRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , vRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , mixAmountArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , textureBindingCount( 0U )
    {
        clearTextureBindings();
    }

    __host__ __device__ MdlMaterialShader( uint_t callableBaseIndex_, uint_t callableCount_, const float3& diffuseTextureScale_ )
        : callableBaseIndex( callableBaseIndex_ )
        , callableCount( callableCount_ )
        , tintArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlockSize( 0U )
        , roughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , uRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , vRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , mixAmountArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , textureBindingCount( 0U )
    {
        clearTextureBindings();
        setTextureBinding( MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX, INVALID_TEXTURE_ID, diffuseTextureScale_,
                           make_float3( 0.0f, 0.0f, 0.0f ) );
    }

    __host__ __device__ MdlMaterialShader( uint_t        callableBaseIndex_,
                                           uint_t        callableCount_,
                                           const float3& diffuseTextureScale_,
                                           const float3& diffuseTextureBias_ )
        : callableBaseIndex( callableBaseIndex_ )
        , callableCount( callableCount_ )
        , tintArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlockSize( 0U )
        , roughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , uRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , vRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , mixAmountArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , textureBindingCount( 0U )
    {
        clearTextureBindings();
        setTextureBinding( MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX, INVALID_TEXTURE_ID, diffuseTextureScale_, diffuseTextureBias_ );
    }

    __host__ __device__ MdlMaterialShader( uint_t        callableBaseIndex_,
                                           uint_t        callableCount_,
                                           uint_t        diffuseTextureId_,
                                           const float3& diffuseTextureScale_,
                                           const float3& diffuseTextureBias_ )
        : callableBaseIndex( callableBaseIndex_ )
        , callableCount( callableCount_ )
        , tintArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlock( CUdeviceptr{} )
        , bsdfArgumentBlockSize( 0U )
        , roughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , uRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , vRoughnessArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , mixAmountArgumentBlockOffset( INVALID_MDL_ARGUMENT_BLOCK_OFFSET )
        , textureBindingCount( 0U )
    {
        clearTextureBindings();
        setTextureBinding( MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX, diffuseTextureId_, diffuseTextureScale_, diffuseTextureBias_ );
    }
};

inline void clearMdlMaterialTextureBindings( MdlMaterialShader& shader )
{
    shader.clearTextureBindings();
}

inline bool setMdlMaterialTextureBinding( MdlMaterialShader& shader, uint_t index, uint_t textureId, const float3& scale, const float3& bias )
{
    return shader.setTextureBinding( index, textureId, scale, bias );
}

inline bool operator==( const MdlMaterialShader& lhs, const MdlMaterialShader& rhs )
{
    if( lhs.callableBaseIndex != rhs.callableBaseIndex || lhs.callableCount != rhs.callableCount
        || lhs.tintArgumentBlock != rhs.tintArgumentBlock || lhs.bsdfArgumentBlock != rhs.bsdfArgumentBlock
        || lhs.bsdfArgumentBlockSize != rhs.bsdfArgumentBlockSize || lhs.roughnessArgumentBlockOffset != rhs.roughnessArgumentBlockOffset
        || lhs.uRoughnessArgumentBlockOffset != rhs.uRoughnessArgumentBlockOffset
        || lhs.vRoughnessArgumentBlockOffset != rhs.vRoughnessArgumentBlockOffset
        || lhs.mixAmountArgumentBlockOffset != rhs.mixAmountArgumentBlockOffset || lhs.textureBindingCount != rhs.textureBindingCount )
    {
        return false;
    }
    for( uint_t i = 0; i < MDL_MATERIAL_TEXTURE_BINDING_COUNT; ++i )
    {
        if( lhs.textureBindings[i] != rhs.textureBindings[i] )
        {
            return false;
        }
    }
    return true;
}

inline bool operator!=( const MdlMaterialShader& lhs, const MdlMaterialShader& rhs )
{
    return !( lhs == rhs );
}
#endif

struct PhongMaterial
{
    float3        Ka;
    float3        Kd;
    float3        Ks;
    float3        Kr;
    float         phongExp;
    MaterialFlags flags;
    uint_t        alphaTextureId;
    uint_t        diffuseTextureId;
};

inline bool operator==( const PhongMaterial& lhs, const PhongMaterial& rhs )
{
    // clang-format off
    return lhs.Ka               == rhs.Ka
        && lhs.Kd               == rhs.Kd
        && lhs.Ks               == rhs.Ks
        && lhs.Kr               == rhs.Kr
        && lhs.phongExp         == rhs.phongExp
        && lhs.flags            == rhs.flags
        && lhs.alphaTextureId   == rhs.alphaTextureId
        && lhs.diffuseTextureId == rhs.diffuseTextureId;
    // clang-format on
}

inline bool operator!=( const PhongMaterial& lhs, const PhongMaterial& rhs )
{
    return !( lhs == rhs );
}

enum class FallbackShaderFeature : uint_t
{
    NONE             = 0,
    CONSTANT_KD      = 1,
    DIFFUSE_TEXTURE  = 2,
    ALPHA_CUTOUT     = 4,
    DIAGNOSTIC_COLOR = 8,
    MASK             = 0xF,
};

inline uint_t operator+( FallbackShaderFeature value )
{
    return static_cast<uint_t>( value );
}

inline FallbackShaderFeature operator|( FallbackShaderFeature lhs, FallbackShaderFeature rhs )
{
    return static_cast<FallbackShaderFeature>( +lhs | +rhs );
}

inline FallbackShaderFeature& operator|=( FallbackShaderFeature& lhs, FallbackShaderFeature rhs )
{
    lhs = lhs | rhs;
    return lhs;
}

inline FallbackShaderFeature operator&( FallbackShaderFeature lhs, FallbackShaderFeature rhs )
{
    return static_cast<FallbackShaderFeature>( +lhs & +rhs );
}

inline bool flagSet( FallbackShaderFeature value, FallbackShaderFeature flag )
{
    return ( value & flag ) == flag;
}

struct FallbackShaderContract
{
    FallbackShaderFeature  featureMask;
    MaterialFallbackReason reason;
};

inline bool operator==( const FallbackShaderContract& lhs, const FallbackShaderContract& rhs )
{
    return lhs.featureMask == rhs.featureMask && lhs.reason == rhs.reason;
}

inline bool operator!=( const FallbackShaderContract& lhs, const FallbackShaderContract& rhs )
{
    return !( lhs == rhs );
}

inline FallbackShaderFeature fallbackFeaturesForMaterial( const PhongMaterial& material )
{
    FallbackShaderFeature features{ FallbackShaderFeature::CONSTANT_KD };
    if( flagSet( material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
    {
        features |= FallbackShaderFeature::DIFFUSE_TEXTURE;
    }
    if( flagSet( material.flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
    {
        features |= FallbackShaderFeature::ALPHA_CUTOUT;
    }
    return features;
}

inline FallbackShaderContract fallbackShaderContract( const MaterialState& state, const PhongMaterial& material )
{
    FallbackShaderFeature features{ fallbackFeaturesForMaterial( material ) };
    if( usesFallbackShader( state ) && state.fallbackReason != MaterialFallbackReason::NO_MDL_BACKEND )
    {
        features |= FallbackShaderFeature::DIAGNOSTIC_COLOR;
    }
    return FallbackShaderContract{ features, state.fallbackReason };
}

struct TriangleUVs
{
    float2 UV[3];
};

inline bool operator==( const TriangleUVs& lhs, const TriangleUVs& rhs )
{
    return lhs.UV[0] == rhs.UV[0] && lhs.UV[1] == rhs.UV[1] && lhs.UV[2] == rhs.UV[2];
}
inline bool operator!=( const TriangleUVs& lhs, const TriangleUVs& rhs )
{
    return !( lhs == rhs );
}

struct TriangleNormals
{
    float3 N[3];
};

inline bool operator==( const TriangleNormals& lhs, const TriangleNormals& rhs )
{
    return lhs.N[0] == rhs.N[0] && lhs.N[1] == rhs.N[1] && lhs.N[2] == rhs.N[2];
}
inline bool operator!=( const TriangleNormals& lhs, const TriangleNormals& rhs )
{
    return !( lhs == rhs );
}

struct LookAtParams
{
    float3 lookAt;
    float3 eye;
    float3 up;
};

struct PerspectiveCamera
{
    float fovY;
    float aspectRatio;
};

struct MaterialIndex
{
    uint_t numPrimitiveGroups;      // number of groups of primitives with different materials
    uint_t primitiveMaterialBegin;  // starting index into PrimitiveMaterialRange array
};

struct PrimitiveMaterialRange
{
    uint_t primitiveEnd;
    uint_t materialId;
};

struct Params
{
    otk::DebugLocation            debug;
    uchar4*                       image;
    float4*                       accumulator;
    uint_t                        width;
    uint_t                        height;
    RenderMode                    renderMode;
    LookAtParams                  lookAt;
    PerspectiveCamera             camera;
    float3                        background;
    uint_t                        numDirectionalLights;
    const DirectionalLight*       directionalLights;
    uint_t                        numInfiniteLights;
    const InfiniteLight*          infiniteLights;
    float3                        ambientColor;
    float3                        proxyFaceColors[6];
    float                         sceneEpsilon;
    bool                          useFaceForward;
    OptixTraversableHandle        traversable;
    demandLoading::DeviceContext  demandContext;
    demandGeometry::Context       demandGeomContext;
    float3                        demandMaterialColor;
    uint_t                        numMaterialStates;      //
    const MaterialState*          materialStates;         // indexed by materialId
#ifdef OTK_USE_MDL
    uint_t                         numMdlMaterialShaders;        //
    const MdlMaterialShader*       mdlMaterialShaders;           // indexed by materialId
    uint_t                         numFourierMaterialResources;  //
    const FourierMaterialResource* fourierMaterialResources;     // indexed by materialId
#endif
    uint_t                        numPartialMaterials;    //
    const PartialMaterial*        partialMaterials;       // indexed by materialId
    uint_t                        numRealizedMaterials;   //
    const PhongMaterial*          realizedMaterials;      // indexed by materialId
    uint_t                        numMaterialIndices;     //
    const MaterialIndex*          materialIndices;        // indexed by instanceId, one entry per instance
    uint_t                        numPrimitiveMaterials;  // one entry per material group per instance
    const PrimitiveMaterialRange* primitiveMaterials;     // indexed by MaterialIndex::primitiveMaterialBegin

    // An array of pointers to arrays of per-face data, one per geometry instance.
    // If the pointer is nullptr, then the instance has no per-face data.
    uint_t            numInstanceNormals;  //
    TriangleNormals** instanceNormals;     // indexed by instanceId, then by primitive index
    uint_t            numInstanceUVs;      //
    TriangleUVs**     instanceUVs;         // indexed by instanceId, then by primitive index
    uint_t            numPartialUVs;       //
    TriangleUVs**     partialUVs;          // indexed by materialId, then by primitive index

    uint_t minAlphaTextureId;
    uint_t maxAlphaTextureId;
    uint_t minDiffuseTextureId;
    uint_t maxDiffuseTextureId;
};

#define PARAMS_STRINGIFY_IMPL( x_ ) #x_
#define PARAMS_STRINGIFY( x_ ) PARAMS_STRINGIFY_IMPL( x_ )
#define PARAMS_VAR_NAME g_params
#define PARAMS_STRING_NAME PARAMS_STRINGIFY( PARAMS_VAR_NAME )

#if __CUDACC__
extern "C" {
__constant__ Params PARAMS_VAR_NAME;
}
#endif

struct RayPayload
{
    float                rayDistance;
    float3               normal;
    float3               color;
    float2               uv;
    const PhongMaterial* material;
    PhongMaterial        materialCopy;
    unsigned int         diffuseTextureId;
    float                worldSpaceTextureSize;
    bool                 isDebug;
    bool                 isBackground;
    bool                 discardRay;
    bool                 hasDirectColor;
#ifdef OTK_USE_MDL
    bool                 hasMdlBsdfSample;
    float                mdlRayConeAngle;
    float                mdlRayConeWidth;
    float4               mdlBsdfSampleXi;
    float3               mdlBsdfSampleDirection;
    float3               mdlBsdfSampleThroughput;
#endif
};

#if __CUDACC__
static __forceinline__ __device__ RayPayload* getRayPayload()
{
    const unsigned int       u0   = optixGetPayload_0();
    const unsigned int       u1   = optixGetPayload_1();
    const unsigned long long uptr = static_cast<unsigned long long>( u0 ) << 32 | u1;
    return reinterpret_cast<RayPayload*>( uptr );
}
#endif

}  // namespace demandPbrtScene
