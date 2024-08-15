// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
