//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/ShaderUtil/Preprocessor.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <vector_functions.h>
#include <vector_types.h>

namespace otk {

/// 4x4 homogeneous coordinate transformation matrix
class Transform4
{
public:
    float4 m[4];
};

OTK_INLINE OTK_HOSTDEVICE bool operator==( const Transform4& t1, const Transform4& t2 )
{
    const float4( &lhs )[4] = t1.m;
    const float4( &rhs )[4] = t2.m;

    return lhs[0] == rhs[0] &&  //
           lhs[1] == rhs[1] &&  //
           lhs[2] == rhs[2] &&  //
           lhs[3] == rhs[3];
}

OTK_INLINE OTK_HOSTDEVICE bool operator!=( const Transform4& lhs, const Transform4& rhs )
{
    return !( lhs == rhs );
}

OTK_INLINE OTK_HOSTDEVICE Transform4 identity()
{
    constexpr float zero{};
    return Transform4{ make_float4( 1.0f, zero, zero, zero ), make_float4( zero, 1.0f, zero, zero ),
        make_float4( zero, zero, 1.0f, zero ), make_float4( zero, zero, zero, 1.0f ) };
}

OTK_INLINE OTK_HOSTDEVICE Transform4 translate(float x = 0.0f, float y = 0.0f, float z = 0.0f)
{
    Transform4 result{identity()};
    result.m[0].w = x;
    result.m[1].w = y;
    result.m[2].w = z;
    return result;
}

OTK_INLINE OTK_HOSTDEVICE Transform4 scale(float x = 0.0f, float y = 0.0f, float z = 0.0f)
{
    Transform4 result{identity()};
    result.m[0].x = x;
    result.m[1].y = y;
    result.m[2].z = z;
    return result;
}

OTK_INLINE OTK_HOSTDEVICE float4 operator*( const Transform4& lhs, const float4& rhs )
{
    return make_float4( dot( lhs.m[0], rhs ), dot( lhs.m[1], rhs ), dot( lhs.m[2], rhs ), dot( lhs.m[3], rhs ) );
}

OTK_INLINE OTK_HOSTDEVICE Transform4 operator*( const Transform4& lhs, const Transform4& rhs )
{
    const float4 rhsColumns[4]{ make_float4( rhs.m[0].x, rhs.m[1].x, rhs.m[2].x, rhs.m[3].x ),    //
                                make_float4( rhs.m[0].y, rhs.m[1].y, rhs.m[2].y, rhs.m[3].y ),    //
                                make_float4( rhs.m[0].z, rhs.m[1].z, rhs.m[2].z, rhs.m[3].z ),    //
                                make_float4( rhs.m[0].w, rhs.m[1].w, rhs.m[2].w, rhs.m[3].w ) };  //

    Transform4 result{};
    for (int row = 0; row < 4; ++row)
    {
        result.m[row].x = dot( lhs.m[row], rhsColumns[0] );
        result.m[row].y = dot( lhs.m[row], rhsColumns[1] );
        result.m[row].z = dot( lhs.m[row], rhsColumns[2] );
        result.m[row].w = dot( lhs.m[row], rhsColumns[3] );
    }
    return result;
}

// cribbed from Stack Overflow: https://stackoverflow.com/questions/2624422/efficient-4x4-matrix-inverse-affine-transform
OTK_INLINE OTK_HOSTDEVICE Transform4 inverse( const Transform4& transform )
{
    const float4 (&a)[4] = transform.m;

    const float s0 = a[0].x * a[1].y - a[1].x * a[0].y;
    const float s1 = a[0].x * a[1].z - a[1].x * a[0].z;
    const float s2 = a[0].x * a[1].w - a[1].x * a[0].w;
    const float s3 = a[0].y * a[1].z - a[1].y * a[0].z;
    const float s4 = a[0].y * a[1].w - a[1].y * a[0].w;
    const float s5 = a[0].z * a[1].w - a[1].z * a[0].w;

    const float c5 = a[2].z * a[3].w - a[3].z * a[2].w;
    const float c4 = a[2].y * a[3].w - a[3].y * a[2].w;
    const float c3 = a[2].y * a[3].z - a[3].y * a[2].z;
    const float c2 = a[2].x * a[3].w - a[3].x * a[2].w;
    const float c1 = a[2].x * a[3].z - a[3].x * a[2].z;
    const float c0 = a[2].x * a[3].y - a[3].x * a[2].y;

    // Should check for 0 determinant
    const float invdet = 1.0f / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);

    Transform4 result;
    float4 (&b)[4] = result.m;

    b[0].x = ( a[1].y * c5 - a[1].z * c4 + a[1].w * c3) * invdet;
    b[0].y = (-a[0].y * c5 + a[0].z * c4 - a[0].w * c3) * invdet;
    b[0].z = ( a[3].y * s5 - a[3].z * s4 + a[3].w * s3) * invdet;
    b[0].w = (-a[2].y * s5 + a[2].z * s4 - a[2].w * s3) * invdet;

    b[1].x = (-a[1].x * c5 + a[1].z * c2 - a[1].w * c1) * invdet;
    b[1].y = ( a[0].x * c5 - a[0].z * c2 + a[0].w * c1) * invdet;
    b[1].z = (-a[3].x * s5 + a[3].z * s2 - a[3].w * s1) * invdet;
    b[1].w = ( a[2].x * s5 - a[2].z * s2 + a[2].w * s1) * invdet;

    b[2].x = ( a[1].x * c4 - a[1].y * c2 + a[1].w * c0) * invdet;
    b[2].y = (-a[0].x * c4 + a[0].y * c2 - a[0].w * c0) * invdet;
    b[2].z = ( a[3].x * s4 - a[3].y * s2 + a[3].w * s0) * invdet;
    b[2].w = (-a[2].x * s4 + a[2].y * s2 - a[2].w * s0) * invdet;

    b[3].x = (-a[1].x * c3 + a[1].y * c1 - a[1].z * c0) * invdet;
    b[3].y = ( a[0].x * c3 - a[0].y * c1 + a[0].z * c0) * invdet;
    b[3].z = (-a[3].x * s3 + a[3].y * s1 - a[3].z * s0) * invdet;
    b[3].w = ( a[2].x * s3 - a[2].y * s1 + a[2].z * s0) * invdet;

    return result;
}

}  // namespace otk
