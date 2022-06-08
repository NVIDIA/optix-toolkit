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

#include <ImageSource/CheckerBoardImage.h>

#include <gtest/gtest.h>

using namespace imageSource;

class TestCheckerBoardImage : public testing::Test
{
};

TEST_F( TestCheckerBoardImage, ReadTile )
{
    CheckerBoardImage image( 128, 128, /*squaresPerSide*/ 16, /*useMipMaps*/ true );

    TextureInfo info;
    image.open( &info );

    const unsigned int mipLevel = 2;

    unsigned int        tileWidth  = 32;
    unsigned int        tileHeight = 32;
    std::vector<float4> buffer( tileWidth * tileHeight );

    float4* texels = buffer.data();
    ASSERT_NO_THROW( image.readTile( reinterpret_cast<char*>( texels ), mipLevel, 1, 1, tileWidth, tileHeight ) );

    // For now, we print the pixels for visual inspection.
    for( unsigned int y = 0; y < tileHeight; ++y )
    {
        for( unsigned int x = 0; x < tileWidth; ++x )
        {
            float4 texel = texels[y * tileWidth + x];
            printf( "%i%i%i ", static_cast<int>( texel.x ), static_cast<int>( texel.y ), static_cast<int>( texel.z ) );
        }
        printf( "\n" );
    }
}


TEST_F( TestCheckerBoardImage, ReadMipLevel )
{
    CheckerBoardImage image( 128, 128, /*squaresPerSide*/ 4, /*useMipMaps*/ true );

    TextureInfo info;
    image.open( &info );

    const unsigned int mipLevel    = 3;
    unsigned int       levelWidth  = 16;
    unsigned int       levelHeight = 16;

    std::vector<float4> buffer( levelWidth * levelHeight );
    float4*             texels = buffer.data();
    ASSERT_NO_THROW( image.readMipLevel( reinterpret_cast<char*>( texels ), mipLevel, levelWidth, levelHeight ) );

    // For now, we print the pixels for visual inspection.
    for( unsigned int y = 0; y < levelHeight; ++y )
    {
        for( unsigned int x = 0; x < levelWidth; ++x )
        {
            float4 texel = texels[y * levelWidth + x];
            printf( "%i%i%i ", static_cast<int>( texel.x ), static_cast<int>( texel.y ), static_cast<int>( texel.z ) );
        }
        printf( "\n" );
    }
}
