// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>
#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>

#include <gtest/gtest.h>

using namespace demandLoading;

class TestTileIndexing : public testing::Test
{
  public:
    TextureSampler sampler{};
    int            m_nextVirtualPage = 0;

    TestTileIndexing()
    {
        static unsigned int texWidth  = 512;
        static unsigned int tileWidth = 64;

        sampler.desc.numMipLevels     = static_cast<int>( 1 + std::log2f( static_cast<float>( texWidth ) ) );
        sampler.desc.logTileWidth     = static_cast<unsigned int>( log2f( static_cast<float>( tileWidth ) ) );
        sampler.desc.logTileHeight    = static_cast<unsigned int>( log2f( static_cast<float>( tileWidth ) ) );
        sampler.desc.isSparseTexture  = 1;
        sampler.desc.wrapMode0        = static_cast<unsigned int>( CU_TR_ADDRESS_MODE_CLAMP );
        sampler.desc.wrapMode1        = static_cast<unsigned int>( CU_TR_ADDRESS_MODE_CLAMP );
        sampler.desc.maxAnisotropy    = 16;
        sampler.desc.mipmapFilterMode = static_cast<unsigned int>( CU_TR_FILTER_MODE_LINEAR );

        sampler.width             = texWidth;
        sampler.height            = texWidth;
        sampler.mipTailFirstLevel = sampler.desc.numMipLevels - static_cast<int>( 1 + sampler.desc.logTileWidth );

        sampler.startPage = 0;

        // Calculate number of tiles.
        demandLoading::TextureSampler::MipLevelSizes* mls = sampler.mipLevelSizes;
        memset( mls, 0, MAX_TILE_LEVELS * sizeof( demandLoading::TextureSampler::MipLevelSizes ) );

        for( int mipLevel = static_cast<int>( sampler.mipTailFirstLevel ); mipLevel >= 0; --mipLevel )
        {
            unsigned int levelWidthInTiles = getLevelDimInTiles( sampler.width, mipLevel + 1, 1 << sampler.desc.logTileWidth );
            unsigned int levelHeightInTiles = getLevelDimInTiles( sampler.height, mipLevel + 1, 1 << sampler.desc.logTileHeight );

            if( mipLevel < static_cast<int>( sampler.mipTailFirstLevel ) )
                mls[mipLevel].mipLevelStart =
                    mls[mipLevel + 1].mipLevelStart + calculateNumTilesInLevel( levelWidthInTiles, levelHeightInTiles );
            else
                mls[mipLevel].mipLevelStart = 0;

            mls[mipLevel].levelWidthInTiles = static_cast<unsigned short>(
                getLevelDimInTiles( sampler.width, static_cast<unsigned int>( mipLevel ), 1 << sampler.desc.logTileWidth ) );
            mls[mipLevel].levelHeightInTiles = static_cast<unsigned short>(
                getLevelDimInTiles( sampler.height, static_cast<unsigned int>( mipLevel ), 1 << sampler.desc.logTileHeight ) );
        }

        unsigned int levelWidthInTiles  = getLevelDimInTiles( sampler.width, 0, 1 << sampler.desc.logTileWidth );
        unsigned int levelHeightInTiles = getLevelDimInTiles( sampler.height, 0, 1 << sampler.desc.logTileHeight );
        sampler.numPages = mls[0].mipLevelStart + calculateNumTilesInLevel( levelWidthInTiles, levelHeightInTiles );
    }
};

TEST_F( TestTileIndexing, calculateLevelDim )
{
    EXPECT_EQ( 32u, calculateLevelDim( 0u, 32u ) );
    EXPECT_EQ( 128u, calculateLevelDim( 1u, 256u ) );
    EXPECT_EQ( 75u, calculateLevelDim( 2u, 301u ) );
    EXPECT_EQ( 1u, calculateLevelDim( 10u, 32u ) );
    EXPECT_EQ( 99u, calculateLevelDim( 1u, 199u ) );
}

TEST_F( TestTileIndexing, wrapTexCoord )
{
    const float firstFloatLessThanOne = 0.999999940395355224609375f;

    EXPECT_FLOAT_EQ( 0.0f, wrapTexCoord( -0.1f, CU_TR_ADDRESS_MODE_CLAMP ) );
    EXPECT_FLOAT_EQ( 0.4f, wrapTexCoord( 0.4f, CU_TR_ADDRESS_MODE_CLAMP ) );
    EXPECT_FLOAT_EQ( firstFloatLessThanOne, wrapTexCoord( 1.2f, CU_TR_ADDRESS_MODE_CLAMP ) );

    EXPECT_FLOAT_EQ( 0.0f, wrapTexCoord( -0.1f, CU_TR_ADDRESS_MODE_BORDER ) );
    EXPECT_FLOAT_EQ( 0.4f, wrapTexCoord( 0.4f, CU_TR_ADDRESS_MODE_BORDER ) );
    EXPECT_FLOAT_EQ( firstFloatLessThanOne, wrapTexCoord( 1.2f, CU_TR_ADDRESS_MODE_BORDER ) );

    EXPECT_FLOAT_EQ( 0.9f, wrapTexCoord( -0.1f, CU_TR_ADDRESS_MODE_WRAP ) );
    EXPECT_FLOAT_EQ( 0.6f, wrapTexCoord( 0.6f, CU_TR_ADDRESS_MODE_WRAP ) );
    EXPECT_FLOAT_EQ( 0.0f, wrapTexCoord( 1.0f, CU_TR_ADDRESS_MODE_WRAP ) );
}

TEST_F( TestTileIndexing, calculateNumTilesInLevel )
{
    EXPECT_EQ( 1u, calculateNumTilesInLevel( 1, 1 ) );
    EXPECT_EQ( 81u, calculateNumTilesInLevel( 9, 9 ) );
    EXPECT_EQ( 128u, calculateNumTilesInLevel( 16, 8 ) );
    EXPECT_EQ( 64u, calculateNumTilesInLevel( 4, 16 ) );
}

TEST_F( TestTileIndexing, unpackTileIndex )
{
    unsigned int mipLevel = 0;
    unsigned int x = 0, y = 0;

    unpackTileIndex( sampler, 0, mipLevel, x, y );
    EXPECT_EQ( 3u, mipLevel );
    EXPECT_EQ( 0u, x );
    EXPECT_EQ( 0u, y );

    unpackTileIndex( sampler, 30, mipLevel, x, y );
    EXPECT_EQ( 0u, mipLevel );
    EXPECT_EQ( 1u, x );
    EXPECT_EQ( 1u, y );

    unpackTileIndex( sampler, 4, mipLevel, x, y );
    EXPECT_EQ( 2u, mipLevel );
    EXPECT_EQ( 1u, x );
    EXPECT_EQ( 1u, y );
}
