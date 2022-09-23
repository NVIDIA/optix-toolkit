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

#include "Memory/EventPool.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class TestEventPool : public testing::Test
{
public:
    void SetUp() override
    {
        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );
    }

    unsigned int m_deviceIndex = 0;
};

TEST_F( TestEventPool, TestEmpty )
{
    EventPool pool( m_deviceIndex );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_EQ( 0U, pool.capacity() );
}

TEST_F( TestEventPool, TestUnusedCapacity )
{
    EventPool pool( m_deviceIndex, 1 );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );
}

TEST_F( TestEventPool, TestWithinCapacity )
{
    EventPool pool( m_deviceIndex, 1 );
    CUevent   event = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );

    pool.free( event );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );

    pool.allocate();
    EXPECT_EQ( 1U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );
}

TEST_F( TestEventPool, TestGrowth )
{
    EventPool pool( m_deviceIndex );
    CUevent   event = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );
    EXPECT_GE( 1U, pool.capacity() );

    pool.free( event );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_GE( 1U, pool.capacity() );

    pool.allocate();
    pool.allocate();
    EXPECT_EQ( 2U, pool.size() );
    EXPECT_GE( 2U, pool.capacity() );
}
