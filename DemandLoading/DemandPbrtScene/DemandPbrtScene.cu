
//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <cuda_fp16.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cmath>

namespace demandPbrtScene {

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

extern "C" __global__ void __raygen__perspectiveCamera()
{
    const uint3              idx{ optixGetLaunchIndex() };
    const Params&            params{ PARAMS_VAR_NAME };
    const LookAtParams&      lookAt{ params.lookAt };
    const PerspectiveCamera& camera{ params.camera };
    const float2             imageSize{ static_cast<float>( params.width ), static_cast<float>( params.height ) };

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
    const float2 d = make_float2( idx.x, idx.y ) / imageSize * 2.f - 1.f;
    float3       u, v, w;
    uvwFrame( lookAt, camera, u, v, w );
    const float3 pinholeRayOrigin = lookAt.eye;
    const float3 pinholeRayDir    = otk::normalize( d.x * u + d.y * v + w );

    float3              result{};
    const float         tMin         = 0.0f;
    const float         tMax         = 1e16f;
    const float         rayTime      = 0.0f;
    const OptixRayFlags flags        = OPTIX_RAY_FLAG_NONE;
    const uint_t        sbtOffset    = RAYTYPE_RADIANCE;
    const uint_t        missSbtIndex = RAYTYPE_RADIANCE;

    const bool usePinhole{ params.usePinholeCamera };
    if( otk::atDebugIndex( params.debug, idx ) && !params.debug.dumpSuppressed )
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

    const float3 rayOrigin{ usePinhole ? pinholeRayOrigin : make_float3( pbrtRayPos.x, pbrtRayPos.y, pbrtRayPos.z ) };
    const float3 rayDirection{ usePinhole ? pinholeRayDir : make_float3( pbrtRayDir.x, pbrtRayDir.y, pbrtRayDir.z ) };
    optixTrace( params.traversable, rayOrigin, rayDirection, tMin, tMax, rayTime, OptixVisibilityMask( 255 ), flags,
                sbtOffset, SBT_STRIDE_COLLAPSE, missSbtIndex, float3Attr( result ) );

    const uint_t pixel{ params.width * idx.y + idx.x };
    params.image[pixel] = makeColor( result );
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
    if( otk::debugInfoDump(
            PARAMS_VAR_NAME.debug, []( const uint3& pixel ) { printf( "Miss at [%u, %u]\n", pixel.x, pixel.y ); }, setRayPayload ) )
    {
        return;
    }

    float3 background{};
    if( 0 /*XXX PARAMS_VAR_NAME.numInfiniteLights > 0 */ )
    {
        bool isResident{};
        bool first{ true };
        for( uint_t i = 0; i < PARAMS_VAR_NAME.numInfiniteLights; ++i )
        {
            const InfiniteLight& light{ PARAMS_VAR_NAME.infiniteLights[i] };
            if( light.skyboxTextureId != 0 && first )
            {
                const float2 uv = sphericalCoordFromRayDirection();
                float4 texel = demandLoading::tex2D<float4>( PARAMS_VAR_NAME.demandContext, light.skyboxTextureId, uv.x,
                                                             uv.y, &isResident );
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
        // XXX background = PARAMS_VAR_NAME.background;
        background = make_float3( .5f, .8f, .9f );
    }

    setRayPayload( background );
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
    const float3* colors = demandPbrtScene::PARAMS_VAR_NAME.proxyFaceColors;
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

    if( otk::debugInfoDump(
            demandPbrtScene::PARAMS_VAR_NAME.debug,
            [=]( const uint3& launchIndex ) {
                printf( "Proxy geometry %u at [%u, %u]: N(%g,%g,%g) index %u, C(%g,%g,%g)\n", optixGetAttribute_3(), launchIndex.x, launchIndex.y,
                        ffNormal.x, ffNormal.y, ffNormal.z, index, colors[index].x, colors[index].y, colors[index].z );
            },
            demandPbrtScene::setRayPayload ) )
    {
        return;
    }

    demandPbrtScene::setRayPayload( colors[index] );
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
    return optixGetInstanceId();
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

    demandPbrtScene::setRayPayload( demandPbrtScene::PARAMS_VAR_NAME.demandMaterialColor );
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
    const uint_t     materialId       = demandMaterial::app::getMaterialId();
    PartialMaterial* partialMaterials = PARAMS_VAR_NAME.partialMaterials;
#ifndef NDEBUG
    if( partialMaterials == nullptr )
    {
        printf( "Parameters partialMaterials array is nullptr!\n" );
        return 0xdeadbeefU;
    }
#endif
    return partialMaterials[materialId].alphaTextureId;
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
    const uint_t textureId = getPartialAlphaTextureId();
    const float2 uv        = getTriangleUVs( PARAMS_VAR_NAME.partialUVs, demandMaterial::app::getMaterialId() );
    bool         isResident{};
    const float texel = demandLoading::tex2D<float>( PARAMS_VAR_NAME.demandContext, textureId, uv.x, uv.y, &isResident );
    const bool ignored = !isResident || texel == 0.0f;
    alphaCutOutDebugInfo( textureId, uv, isResident, texel, ignored );
    if( ignored )
    {
        optixIgnoreIntersection();
    }
}

__device__ __forceinline__ uint_t getRealizedAlphaTextureId()
{
    PhongMaterial* realizedMaterials = PARAMS_VAR_NAME.realizedMaterials;
#ifndef NDEBUG
    if( realizedMaterials == nullptr )
    {
        printf( "Parameters realizedMaterials array is nullptr!\n" );
        return 0xdeadbeefU;
    }
#endif
    return realizedMaterials[optixGetInstanceId()].alphaTextureId;
}

// Use UVs from realized material array
extern "C" __global__ void __anyhit__alphaCutOutMesh()
{
    const uint_t textureId = getRealizedAlphaTextureId();
    const float2 uv        = getTriangleUVs( PARAMS_VAR_NAME.instanceUVs, optixGetInstanceId() );
    bool         isResident{};
    const float texel = demandLoading::tex2D<float>( PARAMS_VAR_NAME.demandContext, textureId, uv.x, uv.y, &isResident );
    const bool ignored = !isResident || texel == 0.0f;
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

__device__ __forceinline__ uint_t getRealizedDiffuseTextureId()
{
    const PhongMaterial* realizedMaterials = PARAMS_VAR_NAME.realizedMaterials;
#ifndef NDEBUG
    if( realizedMaterials == nullptr )
    {
        printf( "Parameters realizedMaterials array is nullptr!\n" );
        return 0xdeadbeefU;
    }
#endif
    return realizedMaterials[optixGetInstanceId()].diffuseTextureId;
}

extern "C" __global__ void __closesthit__texturedMesh()
{
    float3 worldNormal;
    float3 vertices[3];
    getTriangleData(vertices, worldNormal);

    const uint_t instanceId = optixGetInstanceId();
    const float2 uv = getTriangleUVs( PARAMS_VAR_NAME.instanceUVs, instanceId );

    if( triMeshMaterialDebugInfo( vertices, worldNormal, uv ) )
        return;

    const uint_t textureId = getRealizedDiffuseTextureId();
    bool         isResident{};
    const float4 texel = demandLoading::tex2D<float4>( PARAMS_VAR_NAME.demandContext, textureId, uv.x, uv.y, &isResident );
    if( isResident )
    {
        PhongMaterial mat = PARAMS_VAR_NAME.realizedMaterials[instanceId];
        mat.Kd *= make_float3( texel.x, texel.y, texel.z );
        const float3 shaded = phongShade( mat, worldNormal );
        setRayPayload( shaded );
    }
    else
    {
        const float3 yellow( make_float3( 1.0f, 0.0f, 1.0f ) );
        setRayPayload( yellow );
    }
}

}  // namespace demandPbrtScene

#include <OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoaderImpl.h>
