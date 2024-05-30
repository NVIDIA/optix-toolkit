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
