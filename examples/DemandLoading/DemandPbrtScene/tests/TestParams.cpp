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
