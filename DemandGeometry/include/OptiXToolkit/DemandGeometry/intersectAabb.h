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

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>

namespace demandGeometry {
namespace detail {

struct Parallelogram
{
    __device__ Parallelogram( float3 v1_, float3 v2_, float3 anchor )
        : v1( v1_ )
        , v2( v2_ )
        , anchor( anchor )
        , normal( normalize( cross( v1_, v2_ ) ) )
    {
        float d = dot( normal, anchor );
        v1 *= 1.0f / dot( v1_, v1_ );
        v2 *= 1.0f / dot( v2_, v2_ );
        plane = make_float4( normal, d );
    }

    float4 plane;
    float3 v1;
    float3 v2;
    float3 anchor;
    float3 normal;
};

__forceinline__ bool __device__ intersectParallelogram( const float3& rayOrigin,
                                                        const float3& rayDir,
                                                        float         rayTMin,
                                                        float         rayTMax,
                                                        const Parallelogram& floor,
                                                        float& tIntersect )
{
    float3 n  = make_float3( floor.plane );
    float  dt = dot( rayDir, n );
    float  t  = ( floor.plane.w - dot( n, rayOrigin ) ) / dt;
    if( t > rayTMin && t < rayTMax )
    {
        float3 p  = rayOrigin + rayDir * t;
        float3 vi = p - floor.anchor;
        float  a1 = dot( floor.v1, vi );
        if( a1 >= 0 && a1 <= 1 )
        {
            float a2 = dot( floor.v2, vi );
            if( a2 >= 0 && a2 <= 1 )
            {
                tIntersect = t;
                return true;
            }
        }
    }
    return false;
}

__forceinline__ __device__ Parallelogram makeParallelogram( const float3& p0,
                                                            const float3& p1,
                                                            const float3& anchor )
{
    const float3 v1 = p0 - anchor;
    const float3 v2 = p1 - anchor;
    return { v1, v2, anchor };
}

}  // namespace detail

__forceinline__ bool __device__ intersectAabb( const float3&    rayOrigin,
                                               const float3&    rayDir,
                                               float            rayTMin,
                                               float            rayTMax,
                                               const OptixAabb& aabb,
                                               float&           tIntersect,
                                               float3&          normal,
                                               int&             face )
{
    // Points on the AABB are labeled with letters.
    // Faces are labeled with numbers that correspond to array indices.
    //
    //      H-2---5--B    A: p0.X, p0.Y, minZ       0: A, C, G
    //     /|       /|    B: maxX, maxY, maxZ       1: C, D, B
    //    5 3      5 1    C: B.x, A.y, A.z          2: D, E, H
    //   3  |     1  |    D: B.x, A.y, B.z          3: E, A, F
    //  /   2    /   2    E: A.x, A.y, B.z          4: A, E, D
    // F--0--5--G    |    F: A.x, B.y, A.z          5: F, G, B
    // |    E-2-|-4--D    G: B.x, B.y, A.z
    // 3   /    1   /     H: A.x, B.y, B.z
    // |  4     |  4
    // 0 3      0 1
    // |/       |/
    // A--0--4--C
    //
    const float3 A = make_float3( aabb.minX, aabb.minY, aabb.minZ );
    const float3 B = make_float3( aabb.maxX, aabb.maxY, aabb.maxZ );
    const float3 C = make_float3( B.x, A.y, A.z );
    const float3 D = make_float3( B.x, A.y, B.z );
    const float3 E = make_float3( A.x, A.y, B.z );
    const float3 F = make_float3( A.x, B.y, A.z );
    const float3 G = make_float3( B.x, B.y, A.z );
    const float3 H = make_float3( A.x, B.y, B.z );

    const detail::Parallelogram parallelograms[6]{
        // clang-format off
        // 0
        detail::makeParallelogram( A, C, G ),
        // 1
        detail::makeParallelogram( C, B, D ),
        // 2
        detail::makeParallelogram( D, E, H ),
        // 3
        detail::makeParallelogram( E, F, A ),
        // 4
        detail::makeParallelogram( A, D, E ),
        // 5
        detail::makeParallelogram( F, B, G ) };
        // clang-format on
    float currentTIntersect{ rayTMax };
    bool  intersected{ false };
    int   i = 0;
    for( const detail::Parallelogram& p : parallelograms )
    {
        if( intersectParallelogram( rayOrigin, rayDir, rayTMin, rayTMax, p, tIntersect ) )
        {
            if( tIntersect < currentTIntersect )
            {
                normal            = p.normal;
                face              = i;
                currentTIntersect = tIntersect;
                intersected       = true;
            }
        }
        ++i;
    }
    return intersected;
}

}
