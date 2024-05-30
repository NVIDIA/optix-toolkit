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
