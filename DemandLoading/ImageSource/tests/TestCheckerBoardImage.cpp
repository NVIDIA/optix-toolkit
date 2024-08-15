// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>

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
    ASSERT_NO_THROW( image.readTile( reinterpret_cast<char*>( texels ), mipLevel, { 1, 1, tileWidth, tileHeight }, nullptr ) );

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
    ASSERT_NO_THROW( image.readMipLevel( reinterpret_cast<char*>( texels ), mipLevel, levelWidth, levelHeight, nullptr ) );

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
