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

#include "Util/PerContextData.h"

#include <gtest/gtest.h>

#include <cuda.h>

using namespace demandLoading;


class TestPerContextData : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA and create context.
        cuInit( 0 );
        DEMAND_CUDA_CHECK( cuDeviceGet( &m_device, m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuCtxCreate( &m_context, 0, m_device ) );
    }

    void TearDown() override { DEMAND_CUDA_CHECK( cuCtxDestroy( m_context ) ); }

  protected:
    unsigned int m_deviceIndex = 0;
    CUdevice     m_device;
    CUcontext    m_context;
};

TEST_F( TestPerContextData, TestEmpty )
{
    PerContextData<int> map;
}

TEST_F( TestPerContextData, TestFindNone )
{
    PerContextData<int> map;
    int*                data = map.find();
    EXPECT_EQ( nullptr, data );
}

TEST_F( TestPerContextData, TestFind )
{
    PerContextData<int> map;
    map.insert( 1 );
    int* data = map.find();
    ASSERT_NE( nullptr, data );
    EXPECT_EQ( 1, *data );
}

TEST_F( TestPerContextData, TestDestroy )
{
    std::shared_ptr<int> data = std::make_shared<int>();
    EXPECT_EQ( 1, data.use_count() );
    {
        std::shared_ptr<int>                 copy = data;
        PerContextData<std::shared_ptr<int>> map;
        map.insert( std::move( copy ) );
        EXPECT_EQ( 2, data.use_count() );
    }
    EXPECT_EQ( 1, data.use_count() );
}

TEST_F( TestPerContextData, DISABLED_TestDestroyOnInsert )
{
    std::shared_ptr<int> data = std::make_shared<int>();
    EXPECT_EQ( 1, data.use_count() );
    {
        std::shared_ptr<int>                 copy = data;
        PerContextData<std::shared_ptr<int>> map;
        map.insert( std::move( copy ) );
        EXPECT_EQ( 2, data.use_count() );

        map.insert( std::shared_ptr<int>() );
        EXPECT_EQ( 1, data.use_count() );
    }
}

TEST_F( TestPerContextData, TestMultipleContexts )
{
    CUcontext newContext;
    {
        // Associate a value with the current CUDA context.
        PerContextData<int> map;
        map.insert( 1 );

        // Create a new CUDA context.
        DEMAND_CUDA_CHECK( cuCtxCreate( &newContext, 0, m_device ) );

        // Associate a value with the new context.
        map.insert( 2 );
        EXPECT_EQ( 2, *map.find() );

        // Restore previous CUDA context and check associated data.
        DEMAND_CUDA_CHECK( cuCtxSetCurrent( m_context ) );
        EXPECT_EQ( 1, *map.find() );
    }

    // Cleanup
    DEMAND_CUDA_CHECK( cuCtxDestroy( newContext ) );
}
