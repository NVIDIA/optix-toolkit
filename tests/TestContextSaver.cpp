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

#include "Util/ContextSaver.h"

#include <gtest/gtest.h>

#include <cuda.h>

using namespace demandLoading;


class TestContextSaver : public testing::Test
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

    void expectCurrentContext( CUcontext expected )
    {
        CUcontext current;
        DEMAND_CUDA_CHECK( cuCtxGetCurrent( &current ) );
        EXPECT_EQ( expected, current );
    }

  protected:
    unsigned int m_deviceIndex = 0;
    CUdevice     m_device;
    CUcontext    m_context;
};

TEST_F( TestContextSaver, TestNop )
{
    expectCurrentContext( m_context );
    {
        ContextSaver saver;
    }
    expectCurrentContext( m_context );
}

TEST_F( TestContextSaver, TestSave )
{
    expectCurrentContext( m_context );
    {
        ContextSaver saver;
        CUcontext newContext;
        DEMAND_CUDA_CHECK( cuCtxCreate( &newContext, 0, m_device ) );
        expectCurrentContext( newContext );
    }
    expectCurrentContext( m_context );
}


TEST_F( TestContextSaver, TestNestedSave )
{
    expectCurrentContext( m_context );
    {
        ContextSaver saver;
        CUcontext    newContext;
        DEMAND_CUDA_CHECK( cuCtxCreate( &newContext, 0, m_device ) );

        {
            ContextSaver saver;
            CUcontext    nestedContext;
            DEMAND_CUDA_CHECK( cuCtxCreate( &nestedContext, 0, m_device ) );
            expectCurrentContext( nestedContext );
        }

        expectCurrentContext( newContext );
    }
    expectCurrentContext( m_context );
}
