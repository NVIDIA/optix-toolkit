// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// Include this implementation file at the end of your CUDA source file in order
/// to supply the implementation of the proxy instances intersection and closest hit
/// programs.

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>
#include <OptiXToolkit/DemandGeometry/intersectAabb.h>
#include <OptiXToolkit/DemandLoading/Paging.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

namespace demandGeometry {

template <typename T>
__forceinline__ __device__ uint_t& attr( T& val )
{
    return reinterpret_cast<uint_t&>( val );
}

extern "C" __global__ void __intersection__electricBoundingBox()
{
    const uint_t     instanceIdx = optixGetInstanceIndex();
    const OptixAabb& proxy       = app::getContext().proxies[instanceIdx];

    float  tIntersect{};
    float3 normal{};
    int    face;
    if( intersectAabb( optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax(),
                       proxy, tIntersect, normal, face ) )
    {
        optixReportIntersection( tIntersect, 0U, attr( normal.x ), attr( normal.y ), attr( normal.z ), optixGetInstanceId() );
    }
}

extern "C" __global__ void __closesthit__electricBoundingBox()
{
    using namespace otk;

    float3 objectNormal = make_float3( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ),
                                       __uint_as_float( optixGetAttribute_2() ) );
    const uint_t pageId = optixGetAttribute_3();

    bool isResident{};
    const unsigned long long pageTableEntry = demandLoading::pagingMapOrRequest( app::getDeviceContext(), pageId, &isResident );
    // We don't actually care about the value of isResident or pageTableEntry.

    float3 worldNormal = normalize( optixTransformNormalFromObjectToWorldSpace( objectNormal ) );
    float3 ffNormal    = faceforward( worldNormal, -optixGetWorldRayDirection(), worldNormal );
    app::reportClosestHitNormal( ffNormal );
}

}  // namespace demandGeometry
