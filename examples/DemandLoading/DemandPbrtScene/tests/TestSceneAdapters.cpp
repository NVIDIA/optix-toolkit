// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// gtest has to be included before any pbrt junk
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <SceneAdapters.h>

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <algorithm>
#include <iterator>

using namespace demandPbrtScene;
using namespace testing;

using Vector3 = pbrt::Vector3f;

constexpr float zero{};

TEST( TestToInstanceTransform, translate )
{
    const pbrt::Transform translate( Translate( Vector3( 1.0f, 2.0f, 3.0f ) ) );
    float                 instanceTransform[12]{};

    toInstanceTransform( instanceTransform, translate );

    EXPECT_THAT( instanceTransform, ElementsAre(                 //
                                        1.0f, zero, zero, 1.0f,  //
                                        zero, 1.0f, zero, 2.0f,  //
                                        zero, zero, 1.0f, 3.0f ) );
}

TEST( TestToInstanceTransform, scale )
{
    const pbrt::Transform scale( pbrt::Scale( 2.0f, 3.0f, 4.0f ) );
    float                 instanceTransform[12]{};

    toInstanceTransform( instanceTransform, scale );

    EXPECT_THAT( instanceTransform, ElementsAre(                 //
                                        2.0f, zero, zero, zero,  //
                                        zero, 3.0f, zero, zero,  //
                                        zero, zero, 4.0f, zero ) );
}

TEST( TestToFloat4Transform, translate )
{
    const pbrt::Transform translate( Translate( Vector3( 1.0f, 2.0f, 3.0f ) ) );
    float4                transform[4]{};

    toFloat4Transform( transform, translate );

    EXPECT_EQ( make_float4( 1.0f, zero, zero, 1.0f ), transform[0] );
    EXPECT_EQ( make_float4( zero, 1.0f, zero, 2.0f ), transform[1] );
    EXPECT_EQ( make_float4( zero, zero, 1.0f, 3.0f ), transform[2] );
    EXPECT_EQ( make_float4( zero, zero, zero, 1.0f ), transform[3] );
}

TEST( TestToFloat4Transform, scale )
{
    const pbrt::Transform scale( pbrt::Scale( 2.0f, 3.0f, 4.0f ) );
    float4                transform[4]{};

    toFloat4Transform( transform, scale );

    EXPECT_EQ( make_float4( 2.0f, zero, zero, zero ), transform[0] );
    EXPECT_EQ( make_float4( zero, 3.0f, zero, zero ), transform[1] );
    EXPECT_EQ( make_float4( zero, zero, 4.0f, zero ), transform[2] );
    EXPECT_EQ( make_float4( zero, zero, zero, 1.0f ), transform[3] );
}
