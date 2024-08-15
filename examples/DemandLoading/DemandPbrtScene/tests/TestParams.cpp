// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <Params.h>

#include "ParamsPrinters.h"

#include <gtest/gtest.h>

using namespace demandPbrtScene;

TEST( TestDirectionalLightEquality, same )
{
    const DirectionalLight lhs{ make_float3( 1.0f, 2.0f, 3.0f ), make_float3( 4.0f, 5.0f, 6.0f ) };
    const DirectionalLight rhs{ make_float3( 1.0f, 2.0f, 3.0f ), make_float3( 4.0f, 5.0f, 6.0f ) };

    EXPECT_EQ( lhs, rhs );
}

TEST( TestDirectionalLightEquality, directionsDiffer )
{
    const DirectionalLight lhs{ make_float3( 0.0f, 1.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ) };
    const DirectionalLight rhs{ make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 1.0f, 1.0f, 1.0f ) };

    EXPECT_NE( lhs, rhs );
}

TEST( TestDirectionalLightEquality, colorsDiffer )
{
    const DirectionalLight lhs{ make_float3( 0.0f, 1.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ) };
    const DirectionalLight rhs{ make_float3( 0.0f, 1.0f, 1.0f ), make_float3( 1.0f, 0.0f, 1.0f ) };

    EXPECT_NE( lhs, rhs );
}

TEST( TestInfiniteLightEquality, same )
{
    const InfiniteLight lhs{ make_float3( 1.0f, 2.0f, 3.0f ) };
    const InfiniteLight rhs{ make_float3( 1.0f, 2.0f, 3.0f ) };

    EXPECT_EQ( lhs, rhs );
}

TEST( TestInfiniteLightEquality, colorsDiffer )
{
    const InfiniteLight lhs{ make_float3( 0.0f, 1.0f, 0.0f ) };
    const InfiniteLight rhs{ make_float3( 0.0f, 1.0f, 1.0f ) };

    EXPECT_NE( lhs, rhs );
}

TEST( TestInfiniteLightEquality, textureIdsDiffer )
{
    const InfiniteLight lhs{ make_float3( 1.0f, 2.0f, 3.0f ), 1234U };
    const InfiniteLight rhs{ make_float3( 1.0f, 2.0f, 3.0f ), 5678U };

    EXPECT_NE( lhs, rhs );
}

class TestParamPrinters : public ::testing::Test
{
  protected:
    std::ostringstream m_str;
};

TEST_F( TestParamPrinters, infiniteLight )
{
    const InfiniteLight val{ make_float3( 1.0f, 2.0f, 3.0f ), make_float3( 4.0f, 5.0f, 6.0f ), 1234U };

    m_str << val;

    EXPECT_EQ( "InfiniteLight{ color: (1, 2, 3), scale: (4, 5, 6), textureId: 1234 }", m_str.str() );
}
