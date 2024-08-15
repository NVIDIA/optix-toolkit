// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "ImageSourceTestConfig.h"

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/RateLimitedImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <gtest/gtest.h>

using namespace imageSource;

TEST(TestRateLimitedImageSource, TestReadTile)
{
    // Create a shared atomic int, which represents the time remaining.
    std::shared_ptr<std::atomic<int64_t>> timeRemaining( new std::atomic<int64_t> );
    *timeRemaining = 0; // microseconds

    // Create EXRReader and RateLimitedImageSource.
    std::shared_ptr<ImageSource> exrReader( createImageSource( getSourceDir() + "/Textures/TiledMipMappedFloat.exr" ) );
    std::shared_ptr<ImageSource> imageSource( new RateLimitedImageSource( exrReader, timeRemaining ) );

    // Open the texture file.  Note that open() calls proceed regardless of the time limit.
    TextureInfo textureInfo = {};
    ASSERT_NO_THROW( imageSource->open( &textureInfo ) );
    const unsigned int mipLevel = 0;
    const unsigned int width    = exrReader->getTileWidth();
    const unsigned int height   = exrReader->getTileHeight();

    // Allocate tile buffer.
    ASSERT_TRUE( textureInfo.format == CU_AD_FORMAT_FLOAT && textureInfo.numChannels == 4 );
    std::vector<float4> texels( width * height );
    char* data = reinterpret_cast<char*>( texels.data() );

    // When the time remaining is zero, readTile() should exit early and return false.
    *timeRemaining = 0; // microseconds
    EXPECT_FALSE( imageSource->readTile( data, mipLevel, { 1, 1, width, height }, nullptr ) );

    // When the time remaining is positive, readTile() should proceed and return true.
    *timeRemaining = 1; // microseconds
    EXPECT_TRUE( imageSource->readTile( data, mipLevel, { 1, 1, width, height }, nullptr ) );
}

