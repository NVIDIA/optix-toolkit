// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <optix.h>
#include <optix_function_table_definition.h>

#include "SourceDir.h"  // generated from SourceDir.h.in
#include <OptiXToolkit/NeuralTextures/NeuralTextureSource.h>

using namespace neuralTextures;

class TestNeuralTextureSource : public testing::Test
{
  protected:
    void SetUp() override
    {
        // Initialize CUDA context for tests
        cudaFree(0);
    }
};


TEST_F( TestNeuralTextureSource, Loading )
{
    std::string fileName = getSourceDir() + "/Textures/colors.ntc";
    NeuralTextureSource image( fileName.c_str() );
    imageSource::TextureInfo info = image.getInfo();
    image.open( &info );
    NtcTextureSet textureSet = image.getTextureSet();

    EXPECT_EQ( info.width, 105 );
    EXPECT_EQ( info.height, 106 );
    EXPECT_EQ( textureSet.latentFeatures, 8 );
    EXPECT_EQ( textureSet.constants.imageWidth, 422 );
    EXPECT_EQ( textureSet.constants.imageHeight, 425 );
}
