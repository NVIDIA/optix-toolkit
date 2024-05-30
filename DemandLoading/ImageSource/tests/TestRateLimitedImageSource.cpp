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

