// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

using namespace otk;

class TestError : public testing::Test
{
  public:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
    }
};

TEST_F( TestError, TestStreamCheck )
{
    CUstream stream{};
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
}

TEST_F( TestError, TestAssertMsg )
{
    OTK_ASSERT_MSG( true, "" );
}

TEST_F( TestError, TestCuErrorCheck )
{
    OTK_ERROR_CHECK( cuMemFree( CUdeviceptr{} ) );
}
