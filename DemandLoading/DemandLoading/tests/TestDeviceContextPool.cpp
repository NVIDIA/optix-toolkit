// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Memory/DeviceContextPool.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

using namespace demandLoading;

class TestDeviceContextPool : public testing::Test
{
  public:
    const unsigned int m_deviceIndex = 0;
    Options            m_options{};

    TestDeviceContextPool()
    {
        m_options.numPages          = 1025;
        m_options.maxRequestedPages = 65;
        m_options.maxFilledPages    = 63;
        m_options.maxStalePages     = 33;
        m_options.maxEvictablePages = 31;
        m_options.maxEvictablePages = 17;
        m_options.useLruTable       = true;
    }

    void SetUp() { cudaFree( nullptr ); }
};

TEST_F( TestDeviceContextPool, Test )
{
    DeviceContextPool pool( m_options );

    DeviceContext* c1 = pool.allocate();
    DeviceContext* c2 = pool.allocate();
    EXPECT_NE( c1, c2 );

    pool.free( c1 );
    DeviceContext* c1a = pool.allocate();
    EXPECT_EQ( c1, c1a );

    pool.free( c2 );
    DeviceContext* c2a = pool.allocate();
    EXPECT_EQ( c2, c2a );
}
