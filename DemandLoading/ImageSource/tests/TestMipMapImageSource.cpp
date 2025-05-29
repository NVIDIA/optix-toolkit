// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/MipMapImageSource.h>

#include "MockImageSource.h"

#include <OptiXToolkit/Memory/BitCast.h>

#include <gtest/gtest.h>

#include <vector_functions.h>

#include <algorithm>

using namespace testing;
using namespace imageSource;

namespace {

enum
{
    EXPECTED_NUM_MIP_LEVELS = 9
};

using MipMapImageSourcePtr = std::shared_ptr<imageSource::MipMapImageSource>;

class TestMipMapImageSource : public Test
{
  public:
    ~TestMipMapImageSource() override = default;

  protected:
    void SetUp() override;

    ExpectationSet expectCreate();
    void           create();
    ExpectationSet expectOpen();

    otk::testing::MockImageSourcePtr m_baseImage{ std::make_shared<otk::testing::MockImageSource>() };
    imageSource::TextureInfo         m_baseInfo{};
    MipMapImageSourcePtr             m_mipMapImage;
    CUstream                         m_stream{ otk::bit_cast<CUstream>( 0xdeadbeefULL ) };
};

void TestMipMapImageSource::SetUp()
{
    m_baseInfo.width        = 256;
    m_baseInfo.height       = 256;
    m_baseInfo.format       = CU_AD_FORMAT_UNSIGNED_INT8;
    m_baseInfo.numChannels  = 3;
    m_baseInfo.numMipLevels = 1;
    m_baseInfo.isValid      = true;
    m_baseInfo.isTiled      = false;
}

ExpectationSet TestMipMapImageSource::expectCreate()
{
    ExpectationSet create;
    create += EXPECT_CALL( *m_baseImage, isOpen() ).WillRepeatedly( Return( false ) );
    return create;
}

void TestMipMapImageSource::create()
{
    m_mipMapImage = std::make_shared<imageSource::MipMapImageSource>( m_baseImage );
}

ExpectationSet TestMipMapImageSource::expectOpen()
{
    ExpectationSet create{ expectCreate() };
    ExpectationSet open;
    open += EXPECT_CALL( *m_baseImage, open( IsNull() ) ).Times( 1 ).After( create );
    open += EXPECT_CALL( *m_baseImage, getInfo() ).After( create ).WillOnce( ReturnRef( m_baseInfo ) );
    return open;
}

}  // namespace

TEST_F( TestMipMapImageSource, create )
{
    expectCreate();

    create();

    EXPECT_TRUE( m_mipMapImage );
}

TEST_F( TestMipMapImageSource, closeResetsInfo )
{
    ExpectationSet open{ expectOpen() };
    EXPECT_CALL( *m_baseImage, close() ).After( open );
    create();
    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );

    m_mipMapImage->close();
    const imageSource::TextureInfo closedInfo = m_mipMapImage->getInfo();

    EXPECT_TRUE( info.isValid );
    EXPECT_NE( closedInfo, m_baseInfo );
    EXPECT_FALSE( closedInfo.isValid );
    EXPECT_NE( closedInfo, info );
}

TEST_F( TestMipMapImageSource, openReturnsMipMapInfo )
{
    imageSource::TextureInfo mipInfo{ m_baseInfo };
    mipInfo.numMipLevels = EXPECTED_NUM_MIP_LEVELS;
    expectOpen();
    create();

    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );
    const imageSource::TextureInfo result = m_mipMapImage->getInfo();

    EXPECT_EQ( mipInfo, info );
    EXPECT_EQ( mipInfo, result );
}

TEST_F( TestMipMapImageSource, openGetsMipMapInfoOnNullPtr )
{
    imageSource::TextureInfo mipInfo{ m_baseInfo };
    mipInfo.numMipLevels = EXPECTED_NUM_MIP_LEVELS;
    expectOpen();
    create();

    m_mipMapImage->open( nullptr );
    const imageSource::TextureInfo result = m_mipMapImage->getInfo();

    EXPECT_EQ( mipInfo, result );
}

TEST_F( TestMipMapImageSource, getInfoWithoutOpenIsInvalid )
{
    expectCreate();
    create();

    const imageSource::TextureInfo result = m_mipMapImage->getInfo();

    ASSERT_FALSE( result.isValid );
}

TEST_F( TestMipMapImageSource, readTileMipLevelZeroSourcesDataFromReadMipLevelZero )
{
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes{ getBitsPerPixel( m_baseInfo ) / BITS_PER_BYTE };
    const auto fillMipLevel = [=]( char* dest, unsigned int /*mipLevel*/, unsigned int expectedWidth,
                                   unsigned int expectedHeight, CUstream /*stream*/ ) {
        for( unsigned int y = 0; y < expectedHeight; ++y )
        {
            for( unsigned int x = 0; x < expectedWidth; ++x )
            {
                const char val = static_cast<char>( ( x + y ) % 256 );
                std::fill( &dest[0], &dest[pixelSizeInBytes], val );
                dest += pixelSizeInBytes;
            }
        }
    };
    ExpectationSet open{ expectOpen() };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, mipLevelWidth, mipLevelHeight, m_stream ) )
        .After( open )
        .WillOnce( DoAll( fillMipLevel, Return( true ) ) );
    create();
    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );
    ASSERT_TRUE( info.isValid );

    const unsigned int tileX{ 0 };
    const unsigned int tileY{ 0 };
    const unsigned int tileWidth{ 64 };
    const unsigned int tileHeight{ 64 };
    std::vector<char>  dest;
    dest.resize( tileWidth * tileHeight * 4 );
    ASSERT_TRUE( m_mipMapImage->readTile( dest.data(), 0, { tileX, tileY, tileWidth, tileHeight }, m_stream ) );

    for( unsigned int y = 0; y < 64; ++y )
    {
        for( unsigned int x = 0; x < 64; ++x )
        {
            for( unsigned int c = 0; c < pixelSizeInBytes; ++c )
            {
                EXPECT_EQ( static_cast<char>( ( x + y ) % 256 ), dest[y * tileWidth * pixelSizeInBytes + x * pixelSizeInBytes + c] );
            }
        }
    }
}

TEST_F( TestMipMapImageSource, readTileMipLevelOneSourcesDataFromMipLevelZero )
{
    ExpectationSet     open{ expectOpen() };
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes{ getBitsPerPixel( m_baseInfo ) / BITS_PER_BYTE };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, mipLevelWidth, mipLevelHeight, m_stream ) ).After( open ).WillOnce( Return( true ) );
    create();
    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );
    ASSERT_TRUE( info.isValid );

    const unsigned int tileX{ 0 };
    const unsigned int tileY{ 0 };
    const unsigned int tileWidth{ 64 };
    const unsigned int tileHeight{ 64 };
    std::vector<char>  dest;
    dest.resize( tileWidth * tileHeight * pixelSizeInBytes );
    ASSERT_TRUE( m_mipMapImage->readTile( dest.data(), 1, { tileX, tileY, tileWidth, tileHeight }, m_stream ) );
}

TEST_F( TestMipMapImageSource, readMipLevelOneSourcesDataFromMipLevelZero )
{
    ExpectationSet     open{ expectOpen() };
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes{ getBitsPerPixel( m_baseInfo ) / BITS_PER_BYTE };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, mipLevelWidth, mipLevelHeight, m_stream ) ).After( open ).WillOnce( Return( true ) );
    create();
    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );
    ASSERT_TRUE( info.isValid );

    const unsigned int expectedWidth{ 64 };
    const unsigned int expectedHeight{ 64 };
    std::vector<char>  dest;
    dest.resize( expectedWidth * expectedHeight * pixelSizeInBytes );
    ASSERT_TRUE( m_mipMapImage->readMipLevel( dest.data(), 1, expectedWidth, expectedHeight, m_stream ) );
}

TEST_F( TestMipMapImageSource, readMipTailReadsMipLevelZero )
{
    ExpectationSet open{ expectOpen() };
    m_baseInfo.width        = 16;
    m_baseInfo.height       = 16;
    m_baseInfo.numMipLevels = 1;
    const unsigned int mipTailFirstLevel{ 0 };
    const unsigned int numMipLevels{ 5 };
    std::vector<uint2> mipLevelDims;
    unsigned int       size = 16;

    for( unsigned int i = 0; i < numMipLevels; ++i )
    {
        mipLevelDims.push_back( make_uint2( size, size ) );
        size /= 2;
    }
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, m_baseInfo.width, m_baseInfo.height, m_stream ) )
        .After( open )
        .WillOnce( Return( true ) );
    create();
    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );
    ASSERT_TRUE( info.isValid );

    std::vector<char>        dest;
    imageSource::TextureInfo mippedInfo{ m_baseInfo };
    mippedInfo.numMipLevels = numMipLevels;
    dest.resize( getTextureSizeInBytes( mippedInfo ) );
    EXPECT_TRUE( m_mipMapImage->readMipTail( dest.data(), mipTailFirstLevel, numMipLevels, mipLevelDims.data(), m_stream ) );
}

TEST_F( TestMipMapImageSource, tracksTileReadCount )
{
    ExpectationSet open{ expectOpen() };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, m_baseInfo.width, m_baseInfo.height, _ ) ).After( open ).WillOnce( Return( true ) );
    create();
    m_mipMapImage->open( nullptr );
    const unsigned int tileX{ 0 };
    const unsigned int tileY{ 0 };
    const unsigned int tileWidth{ 64 };
    const unsigned int tileHeight{ 64 };
    std::vector<char>  dest;
    dest.resize( tileWidth * tileHeight * 4 );
    m_mipMapImage->readTile( dest.data(), 0, { tileX, tileY, tileWidth, tileHeight }, m_stream );
    m_mipMapImage->readTile( dest.data(), 0, { tileX + tileX, tileY, tileWidth, tileHeight }, m_stream );

    EXPECT_EQ( 2ULL, m_mipMapImage->getNumTilesRead() );
}

namespace {

class TestMipMapImageSourcePassThrough : public TestMipMapImageSource
{
  public:
    ~TestMipMapImageSourcePassThrough() override = default;

  protected:
    void SetUp() override
    {
        TestMipMapImageSource::SetUp();
        m_baseInfo.numMipLevels = 9;
        m_open                  = expectOpen();
        create();
        m_mipMapImage->open( nullptr );
    }

    ExpectationSet m_open;
};

}  // namespace

TEST_F( TestMipMapImageSourcePassThrough, close )
{
    EXPECT_CALL( *m_baseImage, close() ).After( m_open );

    m_mipMapImage->close();
}

TEST_F( TestMipMapImageSourcePassThrough, getInfo )
{
    EXPECT_CALL( *m_baseImage, getInfo() ).After( m_open ).WillOnce( ReturnRef( m_baseInfo ) );

    const imageSource::TextureInfo info = m_mipMapImage->getInfo();

    EXPECT_TRUE( info.isValid );
    EXPECT_LT( 1, info.numMipLevels );
}

TEST_F( TestMipMapImageSourcePassThrough, readTile )
{
    char buffer{};
    EXPECT_CALL( *m_baseImage, readTile( &buffer, 1, imageSource::Tile{ 2, 3, 16, 16 }, m_stream ) ).After( m_open ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_mipMapImage->readTile( &buffer, 1, { 2, 3, 16, 16 }, m_stream ) );
}

TEST_F( TestMipMapImageSourcePassThrough, readMipLevel )
{
    char buffer{};
    EXPECT_CALL( *m_baseImage, readMipLevel( &buffer, 2, 16, 32, m_stream ) ).After( m_open ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_mipMapImage->readMipLevel( &buffer, 2, 16, 32, m_stream ) );
}

TEST_F( TestMipMapImageSourcePassThrough, readMipTail )
{
    char        buffer{};
    const uint2 dims{};
    EXPECT_CALL( *m_baseImage, readMipTail( &buffer, 1, 2, &dims, m_stream ) ).After( m_open ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_mipMapImage->readMipTail( &buffer, 1, 2, &dims, m_stream ) );
}

TEST_F( TestMipMapImageSourcePassThrough, getNumTilesRead )
{
    EXPECT_CALL( *m_baseImage, getNumTilesRead() ).After( m_open ).WillOnce( Return( 13 ) );

    EXPECT_EQ( 13, m_mipMapImage->getNumTilesRead() );
}
