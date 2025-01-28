// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/DeviceTriangles.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/PhongShade.h"

#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <cuda_fp16.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cassert>
#include <cmath>

namespace demandPbrtScene {

static __forceinline__ __device__ void packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0                            = uptr >> 32;
    i1                            = uptr & 0x00000000ffffffff;
}

template <typename T>
__forceinline__ __device__ uint_t& attr( T& val )
{
    return reinterpret_cast<uint_t&>( val );
}

/// Use this in the parameter list to optixTrace.
#define float3Attr( vec_ ) attr( ( vec_ ).x ), attr( ( vec_ ).y ), attr( ( vec_ ).z )

__forceinline__ __device__ uchar4 makeColor( const float3& c )
{
    return make_uchar4( static_cast<unsigned char>( otk::clamp( c.x, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<unsigned char>( otk::clamp( c.y, 0.0f, 1.0f ) * 255.0f ),
                        static_cast<unsigned char>( otk::clamp( c.z, 0.0f, 1.0f ) * 255.0f ), 255u );
}

constexpr unsigned int SBT_STRIDE_COLLAPSE = 0;

__forceinline__ __device__ void uvwFrame( const LookAtParams& lookAt, const PerspectiveCamera& camera, float3& U, float3& V, float3& W )
{
    W          = lookAt.lookAt - lookAt.eye;  // Do not normalize W -- it implies focal length
    float wlen = otk::length( W );
    U          = otk::normalize( otk::cross( W, lookAt.up ) );
    V          = otk::normalize( otk::cross( U, W ) );

    float vlen = wlen * tanf( 0.5f * camera.fovY * M_PIf / 180.0f );
    V *= vlen;
    float ulen = vlen * camera.aspectRatio;
    U *= ulen;
}


// Seed a random number generator for a given pixel and launch index
static __forceinline__ __device__ unsigned int srand( unsigned int x, unsigned int y, unsigned int z )
{
    const unsigned int xyz = 84721u * x + 28411u * y + 339341u * z + 23478;
    return xyz + xyz * xyz * xyz;
}

// Get a random number in [0,1)
static __forceinline__ __device__ float rnd( unsigned int& prev )
{
    prev = 1664525u * prev + 1013904223u;
    return static_cast<float>( prev & 0x00FFFFFF ) / 0x01000000;
}

// Concentric mapping from Ray Tracing Gems : Sampling Transformations Zoo
static __forceinline__ __device__ float2 concentricMapping( float2 u )
{
    float a = 2.0f * u.x - 1.0f;
    float b = 2.0f * u.y - 1.0f;
    if( b == 0.0f )
        b = 1.0f;
    float r, phi;

    if( a * a > b * b )
    {
        r   = a;
        phi = ( M_PIf / 4.0f ) * ( b / a );
    }
    else
    {
        r   = b;
        phi = ( M_PIf / 2.0f ) - ( M_PIf / 4.0f ) * ( a / b );
    }
    return float2{ r * cos( phi ), r * sin( phi ) };
}

// Make an orthonormal basis from unit length vector N.
static __forceinline__ __device__ void makeOrthoBasis( float3 N, float3& S, float3& T )
{
    using namespace otk;
    S = ( fabsf( N.x ) + fabsf( N.y ) > fabsf( N.z ) ) ? float3{ -N.y, N.x, 0.0f } : float3{ 0.0f, -N.z, N.y };
    S = normalize( S );
    T = cross( S, N );
}

// Sample a cosine-weighted hemisphere
static __forceinline__ __device__ float3 sampleDiffuse( float2 xi, float3 N )
{
    float3 S, T;
    makeOrthoBasis( N, S, T );
    const float2 st = concentricMapping( xi );
    return ( st.x * S ) + ( st.y * T ) + ( sqrtf( 1.0f - otk::dot( st, st ) ) * N );
}

// Accumulate a value
static __forceinline__ __device__ void accumulateValue( int pixel, float3 val )
{
    const Params& params{ PARAMS_VAR_NAME };
    float4        accVal      = params.accumulator[pixel] + float4{ val.x, val.y, val.z, 1.0f };
    params.accumulator[pixel] = accVal;
    params.image[pixel]       = makeColor( float3{ accVal.x, accVal.y, accVal.z } / accVal.w );
}

// Show the accumulated value without accumulating anything
static __forceinline__ __device__ void showAccumulatorValue( int pixel, float3 substituteVal )
{
    const Params& params{ PARAMS_VAR_NAME };
    const float4  accVal{ params.accumulator[pixel] };
    if( accVal.w > 0.0f )
        params.image[pixel] = makeColor( float3{ accVal.x, accVal.y, accVal.z } / accVal.w );
    else
        params.image[pixel] = makeColor( substituteVal );
}


extern "C" __global__ void __raygen__perspectiveCamera()
{
    const uint3              idx{ optixGetLaunchIndex() };
    const Params&            params{ PARAMS_VAR_NAME };
    const LookAtParams&      lookAt{ params.lookAt };
    const PerspectiveCamera& camera{ params.camera };
    const float2             imageSize{ static_cast<float>( params.width ), static_cast<float>( params.height ) };
    const uint_t             pixel{ params.width * idx.y + idx.x };
    const RenderMode         renderMode{ params.renderMode };
    const int                subframe{ static_cast<int>( params.accumulator[pixel].w ) };
    unsigned int             rseed{ srand( idx.x, idx.y, subframe ) };

    // Compute ray as pbrt would.
    const otk::Transform4 screenToRaster{ otk::scale( imageSize.x, imageSize.y, 1.0f )
                                          * otk::scale( 1.0f / imageSize.x, -1.0f / imageSize.y, 1.0f )
                                          * otk::translate( 0.0f, -imageSize.y, 0.0f ) };
    const otk::Transform4 rasterToScreen{ inverse( screenToRaster ) };
    const otk::Transform4 rasterToCamera{ inverse( camera.cameraToScreen ) * rasterToScreen };
    const float3          filmPos{ static_cast<float>( idx.x ), static_cast<float>( idx.y ), 0.0f };
    const float4          cameraPos{ rasterToCamera * filmPos };
    const float4          pbrtRayDir{ camera.cameraToWorld * otk::normalize( cameraPos ) };
    const float4          pbrtRayPos{ camera.cameraToWorld * make_float4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    // Compute ray using look at parameters.
    const float2 d = make_float2( idx.x + rnd( rseed ), idx.y + rnd( rseed ) ) / imageSize * 2.f - 1.f;
    float3       u;
    float3       v;
    float3       w;
    uvwFrame( lookAt, camera, u, v, w );
    const float3 pinholeRayOrigin = lookAt.eye;
    const float3 pinholeRayDir    = otk::normalize( d.x * u + d.y * v + w );

    const float         tMin         = params.sceneEpsilon;
    const float         tMax         = 1e16f;
    const float         rayTime      = 0.0f;
    const OptixRayFlags flags        = OPTIX_RAY_FLAG_NONE;
    const uint_t        sbtOffset    = RAYTYPE_RADIANCE;
    const uint_t        missSbtIndex = RAYTYPE_RADIANCE;

    const bool usePinhole{ params.usePinholeCamera };
    if( otk::atDebugIndex( params.debug, idx ) )
    {
        if( usePinhole )
        {
            printf(
                "pinhole raygen [%u,%u] "                                    //
                "org: (%g, %g, %g), "                                        //
                "dir: <%g, %g, %g>, "                                        //
                "LA: E(%g, %g, %g), "                                        //
                "A(%g, %g, %g), "                                            //
                "U<%g, %g, %g>, "                                            //
                "Cam: U:<%3g, %3g, %3g>, "                                   //
                "V:<%3g, %3g, %3g>, "                                        //
                "W:<%3g, %3g, %3g>\n",                                       //
                idx.x, idx.y,                                                //
                pinholeRayOrigin.x, pinholeRayOrigin.y, pinholeRayOrigin.z,  //
                pinholeRayDir.x, pinholeRayDir.y, pinholeRayDir.z,           //
                lookAt.eye.x, lookAt.eye.y, lookAt.eye.z,                    //
                lookAt.lookAt.x, lookAt.lookAt.y, lookAt.lookAt.z,           //
                lookAt.up.x, lookAt.up.y, lookAt.up.z,                       //
                u.x, u.y, u.z,                                               //
                v.x, v.y, v.z,                                               //
                w.x, w.y, w.z                                                //
            );
        }
        else
        {
            printf(
                "pbrt raygen [%g,%g] "                                   //
                "org: (%g, %g, %g, %g), "                                //
                "dir: <%g, %g, %g, %g> "                                 //
                "filmPos: (%g, %g, %g), "                                //
                "cameraPos: (%g, %g, %g, %g)\n",                         //
                filmPos.x, filmPos.y,                                    //
                pbrtRayPos.x, pbrtRayPos.y, pbrtRayPos.z, pbrtRayPos.w,  //
                pbrtRayDir.x, pbrtRayDir.y, pbrtRayDir.z, pbrtRayDir.w,  //
                filmPos.x, filmPos.y, filmPos.z,                         //
                cameraPos.x, cameraPos.y, cameraPos.z, cameraPos.w       //
            );
            auto printMatrix = []( const char* label, const otk::Transform4& transform ) {
                printf(
                    "    %s:\n"                                                              //
                    "        %g, %g, %g, %g\n"                                               //
                    "        %g, %g, %g, %g\n"                                               //
                    "        %g, %g, %g, %g\n"                                               //
                    "        %g, %g, %g, %g\n",                                              //
                    label,                                                                   //
                    transform.m[0].x, transform.m[0].y, transform.m[0].z, transform.m[0].w,  //
                    transform.m[1].x, transform.m[1].y, transform.m[1].z, transform.m[1].w,  //
                    transform.m[2].x, transform.m[2].y, transform.m[2].z, transform.m[2].w,  //
                    transform.m[3].x, transform.m[3].y, transform.m[3].z, transform.m[3].w   //
                );
            };
            printf( "pbrt matrices\n" );
            printMatrix( "screenToRaster", screenToRaster );
            printMatrix( "cameraToScreen", camera.cameraToScreen );
            printMatrix( "rasterToScreen", rasterToScreen );
            printMatrix( "rasterToCamera", rasterToCamera );
        }
    }

    RayPayload   prd{};
    unsigned int u0;
    unsigned int u1;
    packPointer( &prd, u0, u1 );

    prd.rayDistance = tMax;
    float3  rayOrigin{ usePinhole ? pinholeRayOrigin : make_float3( pbrtRayPos.x, pbrtRayPos.y, pbrtRayPos.z ) };
    float3  rayDirection{ usePinhole ? pinholeRayDir : make_float3( pbrtRayDir.x, pbrtRayDir.y, pbrtRayDir.z ) };
    RayCone rayCone = initRayConePinholeCamera( u, v, w, uint2{ params.width, params.height }, rayDirection );
    optixTrace( params.traversable, rayOrigin, rayDirection, tMin, tMax, rayTime, OptixVisibilityMask( 255 ), flags,
                sbtOffset, SBT_STRIDE_COLLAPSE, missSbtIndex, attr( u0 ), attr( u1 ) );
    rayCone       = propagate( rayCone, prd.rayDistance );
    rayCone.angle = MAX_CONE_ANGLE;

    // Render background and electric bounding boxes
    if( otk::dot( prd.normal, prd.normal ) == 0.0f )
    {
        if( prd.isBackground )
            accumulateValue( pixel, prd.color );
        else  // electric bounding box
            showAccumulatorValue( pixel, prd.color );
        return;
    }

    if( prd.discardRay )
    {
        showAccumulatorValue( pixel, float3{ 1.0f, 1.0f, 0.0f } );
        return;
    }

    // Phong shading
    if( renderMode == RenderMode::PRIMARY_RAY )
    {
        if( prd.material == nullptr )
        {
            showAccumulatorValue( pixel, float3{ 1.0f, 1.0f, 0.0f } );
            return;
        }
        PhongMaterial mat{ *prd.material };
        if( prd.diffuseTextureId != 0xffffffff )
        {
            const float  dd{ rayCone.width / prd.worldSpaceTextureSize };
            const float2 ddx{ dd, 0.0f };
            const float2 ddy{ 0.0f, dd };
            bool         isResident{};
            const float4 texel{ demandLoading::tex2DGrad<float4>( params.demandContext, prd.diffuseTextureId, prd.uv.x,
                                                                  prd.uv.y, ddx, ddy, &isResident ) };
            if( !isResident )
            {
                showAccumulatorValue( pixel, float3{ 1.0f, 1.0f, 0.0f } );
                return;
            }
            mat.Kd *= float3{ texel.x, texel.y, texel.z };
        }

        const float3 color{ phongShade( mat, prd.normal, rayDirection ) };
        accumulateValue( pixel, color );
        return;
    }

    // Path tracing starts out with diffuse color, ambient occlusion with white
    float3 sampleColor = float3{ 1.0f, 1.0f, 1.0f };
    if( renderMode == RenderMode::PATH_TRACING && prd.diffuseTextureId != 0xffffffff )
    {
        bool isResident;
        float4 texel = demandLoading::tex2D<float4>( params.demandContext, prd.diffuseTextureId, prd.uv.x, prd.uv.y, &isResident );
        if( !isResident )
        {
            showAccumulatorValue( pixel, float3{ 1.0f, 1.0f, 0.0f } );
            return;
        }
        sampleColor *= float3{ texel.x, texel.y, texel.z };
    }

    // Ambient Occlusion and Diffuse Path Tracing
    const float rayTmax     = ( renderMode == RenderMode::NEAR_AO ) ? 128.0f : 1000000.0f;
    const float maxRayDepth = ( renderMode != RenderMode::PATH_TRACING ) ? 1 : 3;

    for( int rayDepth = 0; rayDepth < maxRayDepth; ++rayDepth )
    {
        // Compute ray scatter direction
        rayOrigin = rayOrigin + prd.rayDistance * rayDirection;
        if( otk::dot( prd.normal, rayDirection ) > 0.0f )
            prd.normal = -prd.normal;
        rayDirection = sampleDiffuse( float2{ rnd( rseed ), rnd( rseed ) }, prd.normal );

        // Trace ray
        prd.rayDistance = rayTmax;
        optixTrace( params.traversable, rayOrigin, rayDirection, tMin, rayTmax, rayTime, OptixVisibilityMask( 255 ),
                    flags, sbtOffset, SBT_STRIDE_COLLAPSE, missSbtIndex, attr( u0 ), attr( u1 ) );
        rayCone       = propagate( rayCone, prd.rayDistance );
        rayCone.angle = MAX_CONE_ANGLE;

        // Ray hit sky, break
        if( prd.rayDistance >= rayTmax )
            break;

        // Multiply by diffuse color
        if( prd.diffuseTextureId != 0xffffffff )
        {
            bool   isResident;
            float  dd    = rayCone.width / prd.worldSpaceTextureSize;
            float2 ddx   = float2{ dd, 0.0f };
            float2 ddy   = float2{ 0.0f, dd };
            float4 texel = demandLoading::tex2DGrad<float4>( params.demandContext, prd.diffuseTextureId, prd.uv.x,
                                                             prd.uv.y, ddx, ddy, &isResident );
            if( isResident )
            {
                sampleColor *= float3{ texel.x, texel.y, texel.z };
            }
        }

        // Max depth reached. Make sample black.
        if( rayDepth == maxRayDepth - 1 )
            sampleColor *= 0.0f;
    }

    if( renderMode == RenderMode::PATH_TRACING )
    {
        // Background is the light source in path tracing mode
        sampleColor *= params.background;
    }
    else
    {
        // Tint sample color for ambient occlusion modes
        float x     = sampleColor.x;
        sampleColor = ( 1.0f - x ) * float3{ 0.0f, 0.0f, 0.3f } + x * float3{ 1.0f, 1.0f, 0.9f };
    }

    accumulateValue( pixel, sampleColor );
}

__forceinline__ __device__ float2 sphericalCoordFromRayDirection()
{
    constexpr float Pi{ 3.141592729f };
    constexpr float TwoPi{ 2.0f * Pi };
    constexpr float InvPi{ 1.0f / Pi };
    constexpr float Inv2Pi{ 1.0f / TwoPi };
    auto            clamp = []( float val, float low, float high ) {
        if( val < low )
            return low;
        if( val > high )
            return high;
        return val;
    };
    auto sphericalPhi = [=]( const float3& v ) {
        const float p = std::atan2( v.y, v.x );
        return ( p < 0.0f ) ? ( p + 2.0f * Pi ) : p;
    };
    auto         sphericalTheta = [=]( const float3& v ) { return std::acos( clamp( v.z, -1.0f, 1.0f ) ); };
    const float3 dir{ optixGetWorldRayDirection() };
    return make_float2( sphericalPhi( dir ) * Inv2Pi, sphericalTheta( dir ) * InvPi );
}

extern "C" __global__ void __miss__backgroundColor()
{
    const Params& params{ PARAMS_VAR_NAME };
    if( otk::debugInfoDump(
            params.debug, []( const uint3& pixel ) { printf( "Miss at [%u, %u]\n", pixel.x, pixel.y ); }, setRayPayload ) )
    {
        return;
    }

    float3 background{};
    if( params.numInfiniteLights > 0 )
    {
        bool isResident{};
        bool first{ true };
        for( uint_t i = 0; i < params.numInfiniteLights; ++i )
        {
            const InfiniteLight& light{ params.infiniteLights[i] };
            if( light.skyboxTextureId != 0 && first )
            {
                const float2 uv = sphericalCoordFromRayDirection();
                float4 texel = demandLoading::tex2D<float4>( params.demandContext, light.skyboxTextureId, uv.x, uv.y, &isResident );
                if( isResident )
                {
                    background += light.color * light.scale * make_float3( texel.x, texel.y, texel.z );
                }
                first = false;
            }
            else
            {
                background += light.color;
            }
        }
    }
    else
    {
        background = params.background;
    }

    getRayPayload()->color        = background;
    getRayPayload()->isBackground = true;
}

}  // namespace demandPbrtScene

namespace demandPbrtScene {

__device__ __forceinline__ uint_t getMaterialId( const Params& params, uint_t instanceId )
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
    matIdx = groups.primitiveMaterialBegin;
    for( uint_t i = 0; i < groups.numPrimitiveGroups; ++i )
    {
        const PrimitiveMaterialRange& group{ params.primitiveMaterials[matIdx] };
        printf( "    %u PrimitiveMaterialRange[%u]{%u, %u}\n", instanceId, matIdx, group.primitiveEnd, group.materialId );
    }
    assert( false );
#endif
    return ~0U;
}

}  // namespace demandPbrtScene

namespace demandGeometry {
namespace app {

__device__ Context& getContext()
{
    return demandPbrtScene::PARAMS_VAR_NAME.demandGeomContext;
}

__device__ const demandLoading::DeviceContext& getDeviceContext()
{
    return demandPbrtScene::PARAMS_VAR_NAME.demandContext;
}

__device__ void reportClosestHitNormal( float3 ffNormal )
{
    // Color the proxy faces by a solid color per face.
    const ::demandPbrtScene::Params& params{ ::demandPbrtScene::PARAMS_VAR_NAME };
    const float3*                    colors = params.proxyFaceColors;
    uint_t                           index{};
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

    if( otk::debugInfoDump(
            params.debug,
            [=]( const uint3& launchIndex ) {
                printf( "Proxy geometry %u at [%u, %u]: N(%g,%g,%g) index %u, C(%g,%g,%g)\n", optixGetAttribute_3(),
                        launchIndex.x, launchIndex.y, ffNormal.x, ffNormal.y, ffNormal.z, index, colors[index].x,
                        colors[index].y, colors[index].z );
            },
            demandPbrtScene::setRayPayload ) )
    {
        return;
    }

    demandPbrtScene::getRayPayload()->color       = colors[index];
    demandPbrtScene::getRayPayload()->rayDistance = optixGetRayTmax();
}

}  // namespace app
}  // namespace demandGeometry

namespace demandMaterial {
namespace app {

__device__ __forceinline__ const demandLoading::DeviceContext& getDeviceContext()
{
    return demandPbrtScene::PARAMS_VAR_NAME.demandContext;
}

__device__ __forceinline__ unsigned int getMaterialId()
{
    const demandPbrtScene::uint_t instanceId{ optixGetInstanceId() };
    const demandPbrtScene::uint_t materialId{ demandPbrtScene::getMaterialId( demandPbrtScene::PARAMS_VAR_NAME, instanceId ) };
#ifndef NDEBUG
    assert( materialId > 0 );
#endif
    return materialId;
}

__device__ __forceinline__ bool proxyMaterialDebugInfo( unsigned int pageId, bool isResident )
{
    return otk::debugInfoDump(
        demandPbrtScene::PARAMS_VAR_NAME.debug,
        [=]( const uint3& launchIndex ) {
            printf( "Demand material at [%u, %u]: materialId %u, isResident %s\n", launchIndex.x, launchIndex.y, pageId,
                    isResident ? "true" : "false" );
        },
        demandPbrtScene::setRayPayload );
}

__device__ __forceinline__ void reportClosestHit( unsigned int materialId, bool isResident )
{
    if( proxyMaterialDebugInfo( materialId, isResident ) )
        return;

    demandPbrtScene::getRayPayload()->color = demandPbrtScene::PARAMS_VAR_NAME.demandMaterialColor;
}

}  // namespace app
}  // namespace demandMaterial

namespace demandPbrtScene {

__device__ __forceinline__ bool alphaCutOutDebugInfo( unsigned int textureId, const float2& uv, bool isResident, unsigned char texel, bool ignored )
{
    return otk::debugInfoDump(
        demandPbrtScene::PARAMS_VAR_NAME.debug,
        [=]( const uint3& launchIndex ) {
            printf( "Alpha cutout at [%u, %u]: textureId %u, uv(%g, %g), isResident %s, texel %u: %s\n",  //
                    launchIndex.x, launchIndex.y,                                                         //
                    textureId,                                                                            //
                    uv.x, uv.y,                                                                           //
                    isResident ? "true" : "false",                                                        //
                    static_cast<unsigned int>( texel ),                                                   //
                    ignored ? "ignored" : "accepted" );                                                   //
        },
        demandPbrtScene::setRayPayload );
}

// Flip V because PBRT texture coordinate space has (0,0) at the lower left corner.
__device__ __forceinline__ float2 adjustUV( float2 uv )
{
    return make_float2( uv.x, 1.f - uv.y );
}

__device__ __forceinline__ float2 interpolateUVs( const TriangleUVs& uv )
{
    const float2 bc = optixGetTriangleBarycentrics();
    return adjustUV( uv.UV[0] ) * ( 1.0f - bc.x - bc.y ) + adjustUV( uv.UV[1] ) * bc.x + adjustUV( uv.UV[2] ) * bc.y;
}

__device__ __forceinline__ uint_t getPartialAlphaTextureId()
{
    // use PARAMS_VAR_NAME.partialMaterials[demandMaterial::app::getMaterialId()].alphaTextureId to sample alpha texture
    const uint_t           materialId{ demandMaterial::app::getMaterialId() };
    const Params&          params{ PARAMS_VAR_NAME };
    const PartialMaterial* partialMaterials{ params.partialMaterials };
#ifndef NDEBUG
    if( partialMaterials == nullptr )
    {
        printf( "Parameters partialMaterials array is nullptr!\n" );
        return 0xdeadbeefU;
    }
    if( materialId >= params.numPartialMaterials )
    {
        printf( "Material id %u exceeds size %u num partial materials\n", materialId, params.numPartialMaterials );
    }
    assert( materialId < params.numPartialMaterials );
#endif
    const uint_t alphaTextureId = partialMaterials[materialId].alphaTextureId;
#ifndef NDEBUG
    if( alphaTextureId < params.minAlphaTextureId || alphaTextureId > params.maxAlphaTextureId )
    {
        printf( "Alpha texture id %u out of range [%u, %u]\n", alphaTextureId, params.minAlphaTextureId, params.maxAlphaTextureId );
    }
    assert( alphaTextureId >= params.minAlphaTextureId && alphaTextureId <= params.maxAlphaTextureId );
#endif
    return alphaTextureId;
}

__device__ __forceinline__ float2 getTriangleUVs( TriangleUVs** uvs, const uint_t index )
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
    return interpolateUVs( triangleUVs[optixGetPrimitiveIndex()] );
}


// Use UVs from partialMaterial array
extern "C" __global__ void __anyhit__alphaCutOutPartialMesh()
{
    const Params& params{ PARAMS_VAR_NAME };
    const uint_t  textureId = getPartialAlphaTextureId();
    const float2  uv        = getTriangleUVs( params.partialUVs, demandMaterial::app::getMaterialId() );
    bool          isResident{};
    const float   texel   = demandLoading::tex2D<float>( params.demandContext, textureId, uv.x, uv.y, &isResident );
    const bool    ignored = isResident && ( texel == 0.0f );
    if( !isResident )
    {
        getRayPayload()->discardRay = true;
    }
    alphaCutOutDebugInfo( textureId, uv, isResident, texel, ignored );
    if( ignored )
    {
        optixIgnoreIntersection();
    }
}

__device__ __forceinline__ const PhongMaterial& getRealizedMaterial()
{
    const Params&        params{ PARAMS_VAR_NAME };
    const PhongMaterial* realizedMaterials{ params.realizedMaterials };
#ifndef NDEBUG
    if( realizedMaterials == nullptr )
    {
        printf( "Parameters realizedMaterials array is nullptr!\n" );
        static PhongMaterial oops{};
        return oops;
    }
#endif
    const uint_t instanceId{ optixGetInstanceId() };
    const uint_t materialId{ getMaterialId( params, instanceId ) };
#ifndef NDEBUG
    if( materialId >= params.numRealizedMaterials )
    {
        printf( "Material id %u exceeds %u num realized materials\n", materialId, params.numRealizedMaterials );
    }
    assert( materialId < params.numRealizedMaterials );
#endif
    return realizedMaterials[materialId];
}

// Use UVs from realized material array
extern "C" __global__ void __anyhit__alphaCutOutMesh()
{
    const Params& params{ PARAMS_VAR_NAME };
    const uint_t  textureId{ getRealizedMaterial().alphaTextureId };
    const uint_t  instanceId{ optixGetInstanceId() };
#ifndef NDEBUG
    if( instanceId >= params.numInstanceUVs )
    {
        printf( "Instance id %u exceeds numInstanceUVs %u\n", instanceId, params.numInstanceUVs );
        assert( instanceId < params.numInstanceUVs );
    }
#endif
    const float2 uv{ getTriangleUVs( params.instanceUVs, instanceId ) };
    bool         isResident{};
    const float  texel{ demandLoading::tex2D<float>( params.demandContext, textureId, uv.x, uv.y, &isResident ) };
    const bool   ignored{ isResident && ( texel == 0.0f ) };
    if( !isResident )
    {
        getRayPayload()->discardRay = true;
    }
    alphaCutOutDebugInfo( textureId, uv, isResident, texel, ignored );
    if( ignored )
    {
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__sphere()
{
    // TODO: support spheres with alpha cutout maps
    // use PARAMS_VAR_NAME.partialMaterials[demandMaterial::app::getMaterialId()].alphaTextureId to sample alpha texture
}

extern "C" __global__ void __closesthit__texturedMesh()
{
    float3 worldNormal;
    float3 vertices[3];
    getTriangleData( vertices, worldNormal );

    const Params& params{ PARAMS_VAR_NAME };
    const uint_t  instanceId{ optixGetInstanceId() };
#ifndef NDEBUG
    if( instanceId >= params.numInstanceUVs )
    {
        printf( "Instance id %u exceeds numInstanceUVs %u\n", instanceId, params.numInstanceUVs );
        assert( instanceId < params.numInstanceUVs );
    }
#endif
    const float2 uv = getTriangleUVs( params.instanceUVs, instanceId );

    if( triMeshMaterialDebugInfo( vertices, worldNormal, uv ) )
        return;

    RayPayload*          prd{ getRayPayload() };
    const PhongMaterial& material{ getRealizedMaterial() };
    const uint_t         diffuseTextureId = material.diffuseTextureId;
#ifndef NDEBUG
    if( diffuseTextureId < params.minDiffuseTextureId || diffuseTextureId > params.maxDiffuseTextureId )
    {
        printf( "Diffuse texture id %u out of range [%u, %u]\n", diffuseTextureId, params.minDiffuseTextureId, params.maxDiffuseTextureId );
    }
    assert( diffuseTextureId >= params.minDiffuseTextureId && diffuseTextureId <= params.maxDiffuseTextureId );
#endif

    prd->diffuseTextureId = diffuseTextureId;
    prd->material         = &material;
    prd->normal           = worldNormal;
    prd->uv               = uv;
    prd->rayDistance      = optixGetRayTmax();
    prd->color            = float3{ 1.0f, 0.0f, 1.0f };

    const float2* uvs          = params.instanceUVs[instanceId][optixGetPrimitiveIndex()].UV;
    const float   a            = otk::length( uvs[2] - uvs[0] ) / otk::length( vertices[2] - vertices[0] );
    const float   b            = otk::length( uvs[2] - uvs[0] ) / otk::length( vertices[2] - vertices[0] );
    const float   c            = otk::length( uvs[2] - uvs[0] ) / otk::length( vertices[2] - vertices[0] );
    prd->worldSpaceTextureSize = ( a + b + c ) / 3.0f;
}

}  // namespace demandPbrtScene

#include <OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoaderImpl.h>
