// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/DeviceTriangles.h"
#include "DemandPbrtScene/FourierBsdfEval.h"

#include <optix.h>

#include <cassert>

using namespace otk;  // for vec_math operators

namespace demandPbrtScene {

constexpr float FOURIER_DISPLAY_GAMMA{ 1.0f / 2.2f };

__device__ __forceinline__ uint_t getFourierMaterialId( const Params& params, uint_t instanceId )
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
    printf( "Requested Fourier material for instance id %u, primitive index %u not found in MaterialIndex{%u, %u}\n",
            instanceId, primIdx, groups.numPrimitiveGroups, groups.primitiveMaterialBegin );
    assert( false );
#endif
    return ~0U;
}

__device__ __forceinline__ const FourierMaterialResource* getFourierMaterialResource( const Params& params, uint_t materialId )
{
#ifndef NDEBUG
    if( materialId >= params.numFourierMaterialResources )
    {
        printf( "Material id %u exceeds number of Fourier material resources %u\n", materialId, params.numFourierMaterialResources );
        assert( materialId < params.numFourierMaterialResources );
    }
    if( params.fourierMaterialResources == nullptr )
    {
        printf( "Parameters Fourier material resources array is nullptr!\n" );
        assert( params.fourierMaterialResources != nullptr );
    }
#endif
    if( materialId >= params.numFourierMaterialResources || params.fourierMaterialResources == nullptr )
    {
        return nullptr;
    }
    return params.fourierMaterialResources + materialId;
}

__device__ __forceinline__ float3 makeFourierTangentU( const float3& normal )
{
    const float3 helper{ fabsf( normal.z ) < 0.999f ? make_float3( 0.0f, 0.0f, 1.0f ) : make_float3( 1.0f, 0.0f, 0.0f ) };
    return otk::normalize( otk::cross( helper, normal ) );
}

__device__ __forceinline__ float3 makeFourierTangentV( const float3& normal, const float3& tangentU )
{
    return otk::normalize( otk::cross( normal, tangentU ) );
}

__device__ __forceinline__ float3 fourierWorldToLocal( const float3& value, const float3& tangentU, const float3& tangentV, const float3& normal )
{
    return make_float3( otk::dot( value, tangentU ), otk::dot( value, tangentV ), otk::dot( value, normal ) );
}

__device__ __forceinline__ float3 fourierLocalToWorld( const float3& value, const float3& tangentU, const float3& tangentV, const float3& normal )
{
    return value.x * tangentU + value.y * tangentV + value.z * normal;
}

__device__ __forceinline__ float3 displayEncodeFourierColor( const float3& color )
{
    return make_float3( powf( fmaxf( color.x, 0.0f ), FOURIER_DISPLAY_GAMMA ),  //
                        powf( fmaxf( color.y, 0.0f ), FOURIER_DISPLAY_GAMMA ),  //
                        powf( fmaxf( color.z, 0.0f ), FOURIER_DISPLAY_GAMMA ) );
}

__device__ __forceinline__ float3 shadeFourierBsdf( const FourierMaterialResource& resource,
                                                    const float3&                  tangentU,
                                                    const float3&                  tangentV,
                                                    const float3&                  normal,
                                                    const float3&                  rayDirection )
{
    float3        result{};
    const float3  outgoing{ fourierWorldToLocal( -rayDirection, tangentU, tangentV, normal ) };
    const Params& params{ PARAMS_VAR_NAME };

    for( uint_t i = 0; i < params.numDirectionalLights; ++i )
    {
        const DirectionalLight&     light{ params.directionalLights[i] };
        const FourierBsdfEvalResult eval{
            evaluateFourierBsdf( resource, outgoing, fourierWorldToLocal( light.direction, tangentU, tangentV, normal ),
                                 FourierBsdfTransportMode::RADIANCE ) };
        result += eval.value * light.color;
    }

    for( uint_t i = 0; i < params.numInfiniteLights; ++i )
    {
        const InfiniteLight&        light{ params.infiniteLights[i] };
        const FourierBsdfEvalResult eval{ evaluateFourierBsdf( resource, outgoing, make_float3( 0.0f, 0.0f, 1.0f ),
                                                               FourierBsdfTransportMode::RADIANCE ) };
        result += eval.value * light.color * light.scale;
    }

    return result;
}

__device__ __forceinline__ bool sampleFourierBsdfPath( const FourierMaterialResource& resource,
                                                       const float3&                  tangentU,
                                                       const float3&                  tangentV,
                                                       const float3&                  normal,
                                                       const float3&                  rayDirection,
                                                       const float4&                  xi,
                                                       float3&                        direction,
                                                       float3&                        throughput )
{
    const float3                  outgoing{ fourierWorldToLocal( -rayDirection, tangentU, tangentV, normal ) };
    const FourierBsdfSampleResult sample{
        sampleFourierBsdf( resource, outgoing, make_float2( xi.x, xi.y ), FourierBsdfTransportMode::RADIANCE ) };
    if( !sample.valid )
    {
        return false;
    }

    direction  = otk::normalize( fourierLocalToWorld( sample.direction, tangentU, tangentV, normal ) );
    throughput = sample.throughput;
    return true;
}

extern "C" __device__ float3 __direct_callable__fourierBsdfEvaluate( const FourierMaterialResource* resource,
                                                                     const float3                   outgoing,
                                                                     const float3                   incoming )
{
    if( resource == nullptr )
    {
        return make_float3( 0.0f, 0.0f, 0.0f );
    }
    return evaluateFourierBsdf( *resource, outgoing, incoming, FourierBsdfTransportMode::IMPORTANCE ).value;
}

extern "C" __device__ float __direct_callable__fourierBsdfPdf( const FourierMaterialResource* resource,
                                                               const float3                   outgoing,
                                                               const float3                   incoming )
{
    if( resource == nullptr )
    {
        return 0.0f;
    }
    return evaluateFourierBsdf( *resource, outgoing, incoming, FourierBsdfTransportMode::IMPORTANCE ).pdf;
}

extern "C" __device__ void __direct_callable__fourierBsdfSample( const FourierMaterialResource* resource,
                                                                 const float3                   outgoing,
                                                                 const float2                   u,
                                                                 FourierBsdfSampleResult*       result )
{
    if( result == nullptr )
    {
        return;
    }
    if( resource == nullptr )
    {
        *result = FourierBsdfSampleResult{ false, make_float3( 0.0f, 0.0f, 0.0f ), 0.0f,
                                           make_float3( 0.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 0.0f ) };
        return;
    }
    *result = sampleFourierBsdf( *resource, outgoing, u, FourierBsdfTransportMode::IMPORTANCE );
}

extern "C" __global__ void __closesthit__fourierMesh()
{
    float3 worldNormal;
    float3 vertices[3];
    getTriangleData( vertices, worldNormal );

    if( triMeshMaterialDebugInfo( vertices, worldNormal, optixGetTriangleBarycentrics() ) )
    {
        return;
    }

    const Params&                  params{ PARAMS_VAR_NAME };
    const uint_t                   instanceId{ optixGetInstanceId() };
    const uint_t                   materialId{ getFourierMaterialId( params, instanceId ) };
    const FourierMaterialResource* resource{ getFourierMaterialResource( params, materialId ) };
    const FourierMaterialResource  invalidResource{};
    const FourierMaterialResource& fourierResource{ resource != nullptr ? *resource : invalidResource };
    const float3                   rayDirection{ optixGetWorldRayDirection() };
    const float                    rayT{ optixGetRayTmax() };
    const float3                   tangentU{ makeFourierTangentU( worldNormal ) };
    const float3                   tangentV{ makeFourierTangentV( worldNormal, tangentU ) };

    RayPayload* prd       = getRayPayload();
    prd->diffuseTextureId = INVALID_TEXTURE_ID;
    prd->material         = nullptr;
    prd->normal           = worldNormal;
    prd->rayDistance      = rayT;
    prd->hasDirectColor   = true;
    prd->hasMdlBsdfSample = false;

    prd->color = displayEncodeFourierColor( shadeFourierBsdf( fourierResource, tangentU, tangentV, worldNormal, rayDirection ) );
    if( params.renderMode == RenderMode::PATH_TRACING )
    {
        prd->hasMdlBsdfSample =
            sampleFourierBsdfPath( fourierResource, tangentU, tangentV, worldNormal, rayDirection, prd->mdlBsdfSampleXi,
                                   prd->mdlBsdfSampleDirection, prd->mdlBsdfSampleThroughput );
    }
}

}  // namespace demandPbrtScene
