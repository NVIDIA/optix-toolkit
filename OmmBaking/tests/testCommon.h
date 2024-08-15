// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>

#include "cuOmmBakingErrorCheck.h"
#include "Util/Image.h"
#include "Util/OptiXOmmArray.h"

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include <gtest/gtest.h>

class TestCommon : public testing::Test
{
  protected:
    OptixDeviceContext    optixContext = {};

    void SetUp() override;

    void TearDown() override;

    cuOmmBaking::Result saveImageToFile( std::string imageNamePrefix, const std::vector<uchar3>& image, uint32_t width, uint32_t height );

    void compareImage();

private:
    std::string m_imageNamePrefix;
};
