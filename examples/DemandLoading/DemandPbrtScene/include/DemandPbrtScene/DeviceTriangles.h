// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Params.h"

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>
#include <OptiXToolkit/ShaderUtil/DebugLocation.h>
#include <OptiXToolkit/ShaderUtil/Transform4.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>
#include <vector_types.h>

#include <stdio.h>

namespace demandPbrtScene {

static __forceinline__ __device__ void setRayPayload( float x, float y, float z )
{
    optixSetPayload_0( __float_as_uint( x ) );
    optixSetPayload_1( __float_as_uint( y ) );
    optixSetPayload_2( __float_as_uint( z ) );
}

static __forceinline__ __device__ void setRayPayload( const float3 &p )
{
    setRayPayload( p.x, p.y, p.z );
}

static __device__ void getTriangleData( float3 ( &vertices )[3], float3& worldNormal )
{
    const Params&                params{ PARAMS_VAR_NAME };
    const uint_t                 instanceId{ optixGetInstanceId() };
    const uint_t                 primIdx{ optixGetPrimitiveIndex() };
    const OptixTraversableHandle gas{ optixGetGASTraversableHandle() };
    const uint_t                 sbtGASIndex{ optixGetSbtGASIndex() };
    optixGetTriangleVertexData( gas, primIdx, sbtGASIndex, 0.f, vertices );

    if( params.instanceNormals != nullptr && params.instanceNormals[instanceId] != nullptr )
    {
        const TriangleNormals& normals{ params.instanceNormals[instanceId][primIdx] };
        const float2           uv{ optixGetTriangleBarycentrics() };
        const float3           uDir( normals.N[1] - normals.N[0] );
        const float3           vDir( normals.N[2] - normals.N[0] );
        worldNormal = optixTransformNormalFromObjectToWorldSpace( normals.N[0] + uDir * uv.x + vDir * uv.y );
        worldNormal = otk::normalize( worldNormal );
    }
    else
    {
        const float3 p12( vertices[1] - vertices[0] );
        const float3 p13( vertices[2] - vertices[0] );
        const float3 objectNormal{ otk::cross( p12, p13 ) };
        worldNormal               = otk::normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) );
    }
    if( params.useFaceForward && optixIsBackFaceHit() )
    {
        worldNormal = -worldNormal;
    }
}

struct TriMeshMaterialDebugInfo
{
    __forceinline__ __device__ TriMeshMaterialDebugInfo( const Params& params, const float3 *vertices, const float3& worldNormal, const float2& uv )
        : params( params )
        , vertices( vertices )
        , worldNormal( worldNormal )
        , uv( uv )
    {
    }

    __forceinline__ __device__ void dump( const uint3& launchIndex ) const
    {
        const uint_t                 instanceId{ optixGetInstanceId() };
        const uint_t                 primIdx{ optixGetPrimitiveIndex() };
        const OptixTraversableHandle gas{ optixGetGASTraversableHandle() };
        const uint_t                 sbtGASIndex{ optixGetSbtGASIndex() };
        const bool                   front{ optixIsFrontFaceHit() };
        const bool                   back{ optixIsBackFaceHit() };
        if( params.instanceNormals != nullptr && params.instanceNormals[instanceId] != nullptr )
        {
            const TriangleNormals& normals{ params.instanceNormals[instanceId][primIdx] };
            printf(                                                                                                  //
                "[%u, %u]: prim: %u, GAS index: %u, GAS: %llx, F: %s B: %s\n"                                        //
                "    P0: [%g,%g,%g], P1: [%g,%g,%g], P2: [%g,%g,%g],\n"                                              //
                "    N0: [%g,%g,%g], N1: [%g,%g,%g], N2: [%g,%g,%g],\n"                                              //
                "    uv: (%g, %g), N: [%g,%g,%g]\n",                                                                 //
                launchIndex.x, launchIndex.y, primIdx, sbtGASIndex, gas, front ? "yes" : "no", back ? "yes" : "no",  //
                vertices[0].x, vertices[0].y, vertices[0].z,                                                         //
                vertices[1].x, vertices[1].y, vertices[1].z,                                                         //
                vertices[2].x, vertices[2].y, vertices[2].z,                                                         //
                normals.N[0].x, normals.N[0].y, normals.N[0].z,                                                      //
                normals.N[1].x, normals.N[1].y, normals.N[1].z,                                                      //
                normals.N[2].x, normals.N[2].y, normals.N[2].z,                                                      //
                uv.x, uv.y,                                                                                          //
                worldNormal.x, worldNormal.y, worldNormal.z );                                                       //
        }
        else
        {
            printf(                                                                                                  //
                "[%u, %u]: prim: %u, GAS index: %u, GAS: %llx, F: %s B: %s\n"                                        //
                "    P0: [%g,%g,%g], P1: [%g,%g,%g], P2: [%g,%g,%g], uv: (%g, %g), N: [%g,%g,%g]\n",                 //
                launchIndex.x, launchIndex.y, primIdx, sbtGASIndex, gas, front ? "yes" : "no", back ? "yes" : "no",  //
                vertices[0].x, vertices[0].y, vertices[0].z,                                                         //
                vertices[1].x, vertices[1].y, vertices[1].z,                                                         //
                vertices[2].x, vertices[2].y, vertices[2].z,                                                         //
                uv.x, uv.y,                                                                                          //
                worldNormal.x, worldNormal.y, worldNormal.z );                                                       //
        }
        const MaterialIndex&   matIdx{ params.materialIndices[instanceId] };
        PrimitiveMaterialRange matRange{};
        for( uint_t i = 0; i < matIdx.numPrimitiveGroups; ++i )
        {
            if( primIdx < params.primitiveMaterials[i + matIdx.primitiveMaterialBegin].primitiveEnd )
            {
                matRange = params.primitiveMaterials[i];
                break;
            }
        }
        printf( "Instance id: %u, MaterialIndex{%u, %u}, PrimitiveMaterial{%u, %u}\n", instanceId,  //
                matIdx.numPrimitiveGroups, matIdx.primitiveMaterialBegin,                           //
                matRange.primitiveEnd, matRange.materialId );
        const PhongMaterial& mat{ params.realizedMaterials[matRange.materialId] };
        printf(
            "    Ka: (%g, %g, %g), "                         //
            "Kd: (%g, %g, %g), "                             //
            "Ks: (%g, %g, %g), "                             //
            "exp: %g, flags: %x, alpha: %u, diffuse: %u\n",  //
            mat.Ka.x, mat.Ka.y, mat.Ka.z,                    //
            mat.Kd.x, mat.Kd.y, mat.Kd.z,                    //
            mat.Ks.x, mat.Ks.y, mat.Ks.z,                    //
            mat.phongExp,                                    //
            static_cast<unsigned>( mat.flags ),              //
            mat.alphaTextureId,                              //
            mat.diffuseTextureId );                          //
        if( params.numDirectionalLights > 0 )
        {
            printf( "Directional lights:\n" );
            for( unsigned int i = 0; i < params.numDirectionalLights; ++i )
            {
                const DirectionalLight& light{ params.directionalLights[i] };
                printf( "    %u: <%g, %g, %g> R:%g, G:%g, B:%g\n",                //
                        i,                                                        //
                        light.direction.x, light.direction.y, light.direction.z,  //
                        light.color.x, light.color.y, light.color.z );            //
            }
        }
        if( params.numInfiniteLights > 0 )
        {
            printf( "Infinite lights:\n" );
            for( unsigned int i = 0; i < params.numInfiniteLights; ++i )
            {
                const InfiniteLight& light{ params.infiniteLights[i] };
                printf( "    %u: color: (%g, %g, %g), scale: (%g, %g, %g), skybox: %u\n",  //
                        i,                                                                 //
                        light.color.x, light.color.y, light.color.z,                       //
                        light.scale.x, light.scale.y, light.scale.z,                       //
                        light.skyboxTextureId );                                           //
            }
        }
    }

    __forceinline__ __device__ void setColor( float r, float g, float b ) const
    {
        setRayPayload( r, g, b );
    }

    const Params& params;
    const float3* vertices;
    const float3& worldNormal;
    const float2& uv;
};

static __forceinline__ __device__ bool triMeshMaterialDebugInfo( const float3 vertices[3], const float3& worldNormal, const float2& uv )
{
    const Params& params{ PARAMS_VAR_NAME };
    return debugInfoDump( params.debug, TriMeshMaterialDebugInfo{ params, vertices, worldNormal, uv } );
}

}  // namespace demandPbrtScene
