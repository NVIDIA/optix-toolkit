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

#include <OptiXToolkit/ShaderUtil/Transform4.h>

#include <gmock/gmock.h>

using namespace otk;
using namespace testing;

constexpr float zero{};

TEST( TestTransform4, identity )
{
    Transform4 xform{ identity() };

    EXPECT_THAT( xform.m, ElementsAre( make_float4( 1.0f, zero, zero, zero ),  //
                                       make_float4( zero, 1.0f, zero, zero ),  //
                                       make_float4( zero, zero, 1.0f, zero ),  //
                                       make_float4( zero, zero, zero, 1.0f ) ) );
}

TEST( TestTransform4, translate )
{
    Transform4 xform{ translate( 11.0f, 22.0f, 33.0f ) };

    EXPECT_THAT( xform.m, ElementsAre( make_float4( 1.0f, zero, zero, 11.0f ),  //
                                       make_float4( zero, 1.0f, zero, 22.0f ),  //
                                       make_float4( zero, zero, 1.0f, 33.0f ),  //
                                       make_float4( zero, zero, zero, 1.0f ) ) );
}

TEST( TestTransform4, scale )
{
    Transform4 xform{ scale( 11.0f, 22.0f, 33.0f ) };

    EXPECT_THAT( xform.m, ElementsAre( make_float4( 11.0f, zero, zero, zero ),  //
                                       make_float4( zero, 22.0f, zero, zero ),  //
                                       make_float4( zero, zero, 33.0f, zero ),  //
                                       make_float4( zero, zero, zero, 1.0f ) ) );
}

TEST( TestTransform4, inverseScale )
{
    Transform4 xform{ inverse( scale( 10.0f, 20.0f, 40.0f ) ) };

    EXPECT_THAT( xform.m, ElementsAre( make_float4( 1.0f / 10.0f, zero, zero, zero ),  //
                                       make_float4( zero, 1.0f / 20.0f, zero, zero ),  //
                                       make_float4( zero, zero, 1.0f / 40.0f, zero ),  //
                                       make_float4( zero, zero, zero, 1.0f ) ) );
}

TEST( TestTransform4, inverseTranslate )
{
    Transform4 xform{ inverse( translate( 10.0f, 20.0f, 40.0f ) ) };

    EXPECT_THAT( xform.m, ElementsAre( make_float4( 1.0f, zero, zero, -10.0f ),  //
                                       make_float4( zero, 1.0f, zero, -20.0f ),  //
                                       make_float4( zero, zero, 1.0f, -40.0f ),  //
                                       make_float4( zero, zero, zero, 1.0f ) ) );
}

TEST( TestTransform4, point2Multiply )
{
    const float2     src{ 1.0f, 2.0f };
    const Transform4 xform1{ translate( 2.0f, 3.0f, 4.0f ) };
    const Transform4 xform2{ scale( 2.0f, 4.0f, 10.0f ) };

    const float4 dest1{ xform1 * src };
    const float4 dest2{ xform2 * src };

    EXPECT_EQ( make_float4( 3.0f, 5.0f, 4.0f, 1.0f ), dest1 );
    EXPECT_EQ( make_float4( 2.0f, 8.0f, zero, 1.0f ), dest2 );
}

TEST( TestTransform4, point3Multiply )
{
    const float3     src{ 1.0f, 2.0f, 3.0f };
    const Transform4 xform1{ translate( 2.0f, 3.0f, 4.0f ) };
    const Transform4 xform2{ scale( 2.0f, 4.0f, 10.0f ) };

    const float4 dest1{ xform1 * src };
    const float4 dest2{ xform2 * src };

    EXPECT_EQ( make_float4( 3.0f, 5.0f, 7.0f, 1.0f ), dest1 );
    EXPECT_EQ( make_float4( 2.0f, 8.0f, 30.0f, 1.0f ), dest2 );
}

TEST( TestTransform4, point4Multiply )
{
    const float4     src{ 1.0f, 2.0f, 3.0f, 1.0f };
    const Transform4 xform1{ translate( 2.0f, 3.0f, 4.0f ) };
    const Transform4 xform2{ scale( 2.0f, 4.0f, 10.0f ) };

    const float4 dest1{ xform1 * src };
    const float4 dest2{ xform2 * src };

    EXPECT_EQ( make_float4( 3.0f, 5.0f, 7.0f, 1.0f ), dest1 );
    EXPECT_EQ( make_float4( 2.0f, 8.0f, 30.0f, 1.0f ), dest2 );
}

TEST( TestTransform4, matrixMultiply )
{
    const Transform4 move{ translate( 2.0f, 3.0f, 4.0f ) };
    const Transform4 resize{ scale( 2.0f, 4.0f, 10.0f ) };

    const Transform4 dest = move * resize;

    EXPECT_THAT( dest.m, ElementsAre( make_float4( 2.0f, zero, zero, 2.0f ),      //
                                      make_float4( zero, 4.0f, zero, 3.0f ),      //
                                      make_float4( zero, zero, 10.0f, 4.0f ),     //
                                      make_float4( zero, zero, zero, 1.0f ) ) );  //
}
