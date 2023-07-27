//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <gtest/gtest.h>

TEST( OperatorTest, float2Equal )
{
    const float2 lhs{1.0f, 2.0f};
    const float2 rhs{1.0f, 2.0f};

    ASSERT_EQ( lhs, rhs );
}

TEST( OperatorTest, float2NotEqual )
{
    const float2 lhs{1.0f, 2.0f};
    const float2 rhs{2.0f, 1.0f};

    ASSERT_NE( lhs, rhs );
}

TEST( OperatorTest, float3Equal )
{
    const float3 lhs{1.0f, 2.0f, 3.0f};
    const float3 rhs{1.0f, 2.0f, 3.0f};

    ASSERT_EQ( lhs, rhs );
}

TEST( OperatorTest, float3NotEqual )
{
    const float3 lhs{1.0f, 2.0f, 3.0f};
    const float3 rhs{3.0f, 2.0f, 1.0f};

    ASSERT_NE( lhs, rhs );
}
