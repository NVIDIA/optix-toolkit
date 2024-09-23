// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/RenderMode.h"

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>
#include <OptiXToolkit/ShaderUtil/DebugLocation.h>
#include <OptiXToolkit/ShaderUtil/Transform4.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>
#include <vector_types.h>

namespace demandPbrtScene {

using uint_t = unsigned int;

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
    uint_t skyboxTextureId;  // non-zero if there is a skybox texture
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
    float           fovY;
    float           focalDistance;
    float           lensRadius;
    float           aspectRatio;
    otk::Transform4 cameraToWorld;
    otk::Transform4 worldToCamera;
    otk::Transform4 cameraToScreen;
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
    DirectionalLight*             directionalLights;
    uint_t                        numInfiniteLights;
    InfiniteLight*                infiniteLights;
    float3                        ambientColor;
    float3                        proxyFaceColors[6];
    float                         sceneEpsilon;
    bool                          usePinholeCamera;
    bool                          useFaceForward;
    OptixTraversableHandle        traversable;
    demandLoading::DeviceContext  demandContext;
    demandGeometry::Context       demandGeomContext;
    float3                        demandMaterialColor;
    uint_t                        numPartialMaterials;     //
    PartialMaterial*              partialMaterials;        // indexed by instanceId
    uint_t                        numRealizedMaterials;    //
    PhongMaterial*                realizedMaterials;       // indexed by materialId
    uint_t                        numInstanceMaterialIds;  //
    uint_t*                       instanceMaterialIds;     // indexed by instanceId, material id per instance
    uint_t                        numMaterialIndices;      //
    const MaterialIndex*          materialIndices;         // indexed by instanceId, one entry per instance
    uint_t                        numPrimitiveMaterials;   // one entry per material group per instance
    const PrimitiveMaterialRange* primitiveMaterials;      // indexed by MaterialIndex::primitiveMaterialBegin

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
    unsigned int         diffuseTextureId;
    bool                 isBackground;
    float                worldSpaceTextureSize;
    bool                 discardRay;
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
