// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandGeometryViewer.h"

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>

using namespace otk;  // for vec_math operators

namespace demandGeometryViewer {

extern "C" {
__constant__ Params g_params;
}

template <typename T>
__forceinline__ __device__ T* getSbtData()
{
    return reinterpret_cast<T*>( optixGetSbtDataPointer() );
}

template <typename T>
__forceinline__ __device__ uint_t& attr( T& val )
{
    return reinterpret_cast<uint_t&>( val );
}

#define float3Attr( vec_ ) attr( ( vec_ ).x ), attr( ( vec_ ).y ), attr( ( vec_ ).z )

__forceinline__ __device__ uchar4 makeColor( const float3& c )
{
    return make_uchar4( static_cast<unsigned char>( clamp( c.x, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<unsigned char>( clamp( c.y, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<unsigned char>( clamp( c.z, 0.0f, 1.0f ) * 255.0f ), 255u );
}

extern "C" __global__ void __raygen__pinHoleCamera()
{
    const uint3  idx    = optixGetLaunchIndex();
    const auto*  camera = getSbtData<CameraData>();
    const uint_t pixel  = g_params.width * idx.y + idx.x;

    float2 d         = make_float2( idx.x, idx.y ) / make_float2( g_params.width, g_params.height ) * 2.f - 1.f;
    float3 rayOrigin = camera->eye;
    float3 rayDir    = normalize( d.x * camera->U + d.y * camera->V + camera->W );
    float3 result{};

    float         tMin         = 0.0f;
    float         tMax         = 1e16f;
    float         rayTime      = 0.0f;
    OptixRayFlags flags        = OPTIX_RAY_FLAG_NONE;
    uint_t        sbtOffset    = RAYTYPE_RADIANCE;
    uint_t        sbtStride    = RAYTYPE_COUNT;
    uint_t        missSbtIndex = RAYTYPE_RADIANCE;
    optixTrace( g_params.traversable, rayOrigin, rayDir, tMin, tMax, rayTime, OptixVisibilityMask( 255 ), flags,
                sbtOffset, sbtStride, missSbtIndex, float3Attr( result ) );

    g_params.image[pixel] = makeColor( result );
}

static __forceinline__ __device__ void setRayPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

extern "C" __global__ void __miss__backgroundColor()
{
    const auto* data = getSbtData<MissData>();
    setRayPayload( make_float3( data->background.x, data->background.y, data->background.z ) );
}

}  // namespace demandGeometryViewer

namespace demandGeometry {
namespace app {

__device__ Context& getContext()
{
    return demandGeometryViewer::g_params.demandGeomContext;
}

__device__ const demandLoading::DeviceContext& getDeviceContext()
{
    return demandGeometryViewer::g_params.demandContext;
}

__device__ void reportClosestHitNormal( float3 ffNormal )
{
    // Color the proxy faces by a solid color per face.
    const float3* colors = demandGeometryViewer::g_params.proxyFaceColors;
    uint_t        index{};
    if( ffNormal.x > 0.5f )
        index = 0;
    else if( ffNormal.x < -0.5f )
        index = 1;
    else if( ffNormal.y > 0.5f )
        index = 2;
    else if( ffNormal.y < -0.5f )
        index = 3;
    else if( ffNormal.z > 0.5f )
        index = 4;
    else if( ffNormal.z < -0.5f )
        index = 5;

    demandGeometryViewer::setRayPayload( colors[index] );
}

}  // namespace app
}  // namespace demandGeometry

namespace demandMaterial {
namespace app {

__device__ __forceinline__ const demandLoading::DeviceContext& getDeviceContext()
{
    return demandGeometryViewer::g_params.demandContext;
}

__device__ __forceinline__ unsigned int getMaterialId()
{
    return demandGeometryViewer::g_params.demandMaterialPageIds[optixGetPrimitiveIndex()];
}

__device__ __forceinline__ void reportClosestHit( unsigned int /*materialId*/, bool /*isResident*/ )
{
    demandGeometryViewer::setRayPayload( demandGeometryViewer::g_params.demandMaterialColor );
}

}  // namespace app
}  // namespace demandMaterial

#include <OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoaderImpl.h>
