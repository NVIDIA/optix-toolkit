// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Util/ContextSaver.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <gtest/gtest.h>

#include <cuda.h>

using namespace demandLoading;

namespace {

CUresult ctxCreate( CUcontext* ctx, unsigned int flags, CUdevice dev )
{
#if CUDA_VERSION >= 13000
    CUctxCreateParams params{};
    return cuCtxCreate( ctx, &params, flags, dev );
#else
    return cuCtxCreate( ctx, flags, dev );
#endif
}

}  // anonymous namespace


class TestContextSaver : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA and create context.
        cuInit( 0 );
        OTK_ERROR_CHECK( cuDeviceGet( &m_device, m_deviceIndex ) );
        OTK_ERROR_CHECK( ctxCreate( &m_context, 0, m_device ) );
    }

    void TearDown() override { OTK_ERROR_CHECK( cuCtxDestroy( m_context ) ); }

    void expectCurrentContext( CUcontext expected )
    {
        CUcontext current;
        OTK_ERROR_CHECK( cuCtxGetCurrent( &current ) );
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
        OTK_ERROR_CHECK( ctxCreate( &newContext, 0, m_device ) );
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
        OTK_ERROR_CHECK( ctxCreate( &newContext, 0, m_device ) );

        {
            ContextSaver saver2;
            CUcontext    nestedContext;
            OTK_ERROR_CHECK( ctxCreate( &nestedContext, 0, m_device ) );
            expectCurrentContext( nestedContext );
        }

        expectCurrentContext( newContext );
    }
    expectCurrentContext( m_context );
}
