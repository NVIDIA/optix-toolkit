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

}  // namespace otk
