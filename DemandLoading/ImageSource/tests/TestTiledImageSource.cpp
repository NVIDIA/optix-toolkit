// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/TiledImageSource.h>

#include "MockImageSource.h"

#include <OptiXToolkit/Memory/BitCast.h>

#include <vector_functions.h>

#include <gmock/gmock.h>

#include <algorithm>

using namespace testing;
using namespace imageSource;

namespace {

using TiledImageSourcePtr = std::shared_ptr<imageSource::TiledImageSource>;

class TestTiledImageSource : public Test
{
  public:
    ~TestTiledImageSource() override = default;

  protected:
    void SetUp() override;

    ExpectationSet expectCreate();
    void           create();
    ExpectationSet expectOpen();
    unsigned int   getPixelSizeInBytes() const
    {
        return getBitsPerPixel( m_baseInfo ) / BITS_PER_BYTE;
    }
    void expectLevelZeroFilledAfter( const ExpectationSet& before );

    otk::testing::MockImageSourcePtr m_baseImage{ std::make_shared<otk::testing::MockImageSource>() };
    imageSource::TextureInfo         m_baseInfo{};
    TiledImageSourcePtr              m_tiledImage;
    CUstream                         m_stream{ otk::bit_cast<CUstream>( 0xdeadbeefULL ) };
};

void TestTiledImageSource::SetUp()
{
    m_baseInfo.width        = 1920;
    m_baseInfo.height       = 1080;
    m_baseInfo.format       = CU_AD_FORMAT_UNSIGNED_INT8;
    m_baseInfo.numChannels  = 3;
    m_baseInfo.numMipLevels = 1;
    m_baseInfo.isValid      = true;
    m_baseInfo.isTiled      = false;
}

ExpectationSet TestTiledImageSource::expectCreate()
{
    ExpectationSet expect;
    expect += EXPECT_CALL( *m_baseImage, isOpen() ).WillOnce( Return( false ) );
    return expect;
}

void TestTiledImageSource::create()
{
    m_tiledImage = std::make_shared<imageSource::TiledImageSource>( m_baseImage );
}

ExpectationSet TestTiledImageSource::expectOpen()
{
    ExpectationSet create{ expectCreate() };
    ExpectationSet open;
    open += EXPECT_CALL( *m_baseImage, isOpen() ).Times( 1 ).After( create ).WillRepeatedly( Return( false ) );
    open += EXPECT_CALL( *m_baseImage, open( IsNull() ) ).After( create );
    open += EXPECT_CALL( *m_baseImage, getInfo() ).After( create ).WillOnce( ReturnRef( m_baseInfo ) );
    return open;
}

void TestTiledImageSource::expectLevelZeroFilledAfter( const ExpectationSet& before )
{
    const unsigned int mipLevel{ 0 };
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes = getPixelSizeInBytes();
    const auto         fillMipLevel     = [=]( char* dest, unsigned int /*mipLevel*/, unsigned int expectedWidth,
                                   unsigned int expectedHeight, CUstream /*stream*/ ) {
        int val{};
        for( unsigned int y = 0; y < expectedHeight; ++y )
        {
            for( unsigned int x = 0; x < expectedWidth; ++x )
            {
                std::fill( &dest[0], &dest[pixelSizeInBytes], static_cast<char>( val & 0xFF ) );
                ++val;
                dest += pixelSizeInBytes;
            }
        }
    };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), mipLevel, mipLevelWidth, mipLevelHeight, m_stream ) )
        .After( before )
        .WillOnce( DoAll( fillMipLevel, Return( true ) ) );
}

}  // namespace

TEST_F( TestTiledImageSource, create )
{
    expectCreate();

    create();

    EXPECT_TRUE( m_tiledImage );
}

TEST_F( TestTiledImageSource, closeResetsInfo )
{
    ExpectationSet open{ expectOpen() };
    EXPECT_CALL( *m_baseImage, close() ).After( open );
    imageSource::TextureInfo info{};
    create();
    m_tiledImage->open( &info );

    m_tiledImage->close();
    const imageSource::TextureInfo closedInfo = m_tiledImage->getInfo();

    EXPECT_TRUE( info.isValid );
    EXPECT_NE( closedInfo, m_baseInfo );
    EXPECT_FALSE( closedInfo.isValid );
    EXPECT_NE( closedInfo, info );
}

TEST_F( TestTiledImageSource, openReturnsTiledInfo )
{
    imageSource::TextureInfo tiledInfo{ m_baseInfo };
    tiledInfo.isTiled = true;
    expectOpen();
    create();

    imageSource::TextureInfo info{};
    m_tiledImage->open( &info );
    const imageSource::TextureInfo result = m_tiledImage->getInfo();

    EXPECT_EQ( tiledInfo, info );
    EXPECT_EQ( tiledInfo, result );
}

TEST_F( TestTiledImageSource, openGetsTiledInfoOnNullPtr )
{
    imageSource::TextureInfo tiledInfo{ m_baseInfo };
    tiledInfo.isTiled = true;
    expectOpen();
    create();

    m_tiledImage->open( nullptr );
    const imageSource::TextureInfo result = m_tiledImage->getInfo();

    EXPECT_EQ( tiledInfo, result );
}

TEST_F( TestTiledImageSource, getInfoWithoutOpenIsInvalid )
{
    expectCreate();
    create();

    const imageSource::TextureInfo result = m_tiledImage->getInfo();

    ASSERT_FALSE( result.isValid );
}

inline int charAsInt( char c )
{
    return static_cast<unsigned char>( c );
}

MATCHER_P4( hasPixelRowValueSequence, y, rowWidth, pixelSizeInBytes, start, "" )
{
    const std::vector<char> &dest = arg;
    unsigned int val = start;
    bool result{true};
    for( unsigned int x = 0; x < rowWidth; ++x )
    {
        for( unsigned int c = 0; c < pixelSizeInBytes; ++c )
        {
            const char expected = static_cast<char>( val & 0xFF );
            const char actual = dest[y * rowWidth * pixelSizeInBytes + x * pixelSizeInBytes + c];
            if ( expected != actual )
            {
                if (!result)
                {
                    *result_listener << "; ";
                }
                *result_listener << "[" << x << "," << y << "][" << c << "]: expected " << charAsInt(expected) << ", got " << charAsInt(actual);
                result = false;
            }
        }
        ++val;
    }
    if (result)
    {
        *result_listener << "matches pattern";
    }
    return result;
}

MATCHER_P3( hasPixelRowZeroSequence, y, rowWidth, pixelSizeInBytes, "")
{
    const std::vector<char> &dest = arg;
    bool result{true};
    for( unsigned int x = 0; x < rowWidth; ++x )
    {
        for( unsigned int c = 0; c < pixelSizeInBytes; ++c )
        {
            const char actual = dest[y * rowWidth * pixelSizeInBytes + x * pixelSizeInBytes + c];
            if ( actual != 0 )
            {
                if (!result)
                {
                    *result_listener << "; ";
                }
                *result_listener << "[" << x << "," << y << "][" << c << "]: expected zero, got " << charAsInt(actual);
                result = false;
            }
        }
    }
    if (result)
    {
        *result_listener << "matches zeros";
    }
    return result;
}

TEST_F( TestTiledImageSource, readFullTileSourcesDataFromReadMipLevel )
{
    ExpectationSet before = expectOpen();
    create();
    expectLevelZeroFilledAfter( before );
    imageSource::TextureInfo info{};
    m_tiledImage->open( &info );
    ASSERT_TRUE( info.isValid );
    const unsigned int      tileSize{ 64 };
    const imageSource::Tile tile{ 0, 0, tileSize, tileSize };
    std::vector<char>       dest;
    dest.resize( tile.width * tile.height * 4 );

    ASSERT_TRUE( m_tiledImage->readTile( dest.data(), 0, tile, m_stream ) );

    const unsigned int pixelSizeInBytes = getPixelSizeInBytes();
    for( unsigned int y = 0; y < tile.height; ++y )
    {
        unsigned int val = y * m_baseInfo.width;
        EXPECT_THAT( dest, hasPixelRowValueSequence( y, tile.width, pixelSizeInBytes, val ) );
    }
}

TEST_F( TestTiledImageSource, readPartialTileAtWidthBoundary )
{
    const unsigned int tileSize{ 32 };
    m_baseInfo.width      = 2 * tileSize + tileSize / 2;  // 2.5 tiles wide
    m_baseInfo.height     = 2 * tileSize;                 // 2 tiles high
    ExpectationSet before = expectOpen();
    create();
    expectLevelZeroFilledAfter( before );
    imageSource::TextureInfo info{};
    m_tiledImage->open( &info );
    ASSERT_TRUE( info.isValid );
    const imageSource::Tile tile{ 2, 0, tileSize, tileSize };
    std::vector<char>       dest;
    dest.resize( tile.width * tile.height * 4 );

    ASSERT_TRUE( m_tiledImage->readTile( dest.data(), 0, tile, m_stream ) );

    const unsigned int pixelSizeInBytes = getPixelSizeInBytes();
    for( unsigned int y = 0; y < tile.height; ++y )
    {
        unsigned int val = y * m_baseInfo.width + tile.x * tile.width;
        for( unsigned int x = 0; x < tile.width; ++x )
        {
            if( x < tile.width / 2 )
            {
                for( unsigned int c = 0; c < pixelSizeInBytes; ++c )
                {
                    EXPECT_EQ( static_cast<char>( val & 0xFF ), dest[y * tile.width * pixelSizeInBytes + x * pixelSizeInBytes + c] )
                        << "[" << x << ", " << y << "] channel " << c;
                }
            }
            else
            {
                for( unsigned int c = 0; c < pixelSizeInBytes; ++c )
                {
                    EXPECT_EQ( 0, dest[y * tile.width * pixelSizeInBytes + x * pixelSizeInBytes + c] )
                        << "[" << x << ", " << y << "] channel " << c;
                }
            }
            ++val;
        }
    }
}

TEST_F( TestTiledImageSource, readPartialTileAtHeightBoundary )
{
    const unsigned int tileSize{ 32 };
    m_baseInfo.width      = 2 * tileSize;                 // 2 tiles high
    m_baseInfo.height     = 2 * tileSize + tileSize / 2;  // 2.5 tiles wide
    ExpectationSet before = expectOpen();
    create();
    expectLevelZeroFilledAfter( before );
    imageSource::TextureInfo info{};
    m_tiledImage->open( &info );
    ASSERT_TRUE( info.isValid );
    const imageSource::Tile tile{ 0, 2, tileSize, tileSize };
    std::vector<char>       dest;
    dest.resize( tile.width * tile.height * 4 );

    ASSERT_TRUE( m_tiledImage->readTile( dest.data(), 0, tile, m_stream ) );

    const unsigned int pixelSizeInBytes = getPixelSizeInBytes();
    for( unsigned int y = 0; y < tile.height; ++y )
    {
        unsigned int val = ( y + tile.y * tile.height ) * m_baseInfo.width;
        if( y < tile.height / 2 )
        {
            EXPECT_THAT( dest, hasPixelRowValueSequence( y, tile.width, pixelSizeInBytes, val ) );
        }
        else
        {
            EXPECT_THAT( dest, hasPixelRowZeroSequence( y, tile.width, pixelSizeInBytes ) );
        }
    }
}

TEST_F( TestTiledImageSource, readMipTailReadsMipLevels )
{
    ExpectationSet open{ expectOpen() };
    create();
    imageSource::TextureInfo baseMipInfo{ m_baseInfo };
    baseMipInfo.width        = 16;
    baseMipInfo.height       = 16;
    baseMipInfo.numMipLevels = 5;
    const unsigned int mipTailFirstLevel{ 0 };
    const unsigned int numMipLevels{ 5 };
    std::vector<uint2> mipLevelDims;
    unsigned int       size = 1;

    for( unsigned int i = 0; i < numMipLevels; ++i )
    {
        mipLevelDims.push_back( make_uint2( size, size ) );
        EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), i, size, size, m_stream ) ).WillOnce( Return( true ) );
        size *= 2;
    }
    m_tiledImage->open( nullptr );

    std::vector<char> dest;
    dest.resize( getTextureSizeInBytes( baseMipInfo ) );
    EXPECT_TRUE( m_tiledImage->readMipTail( dest.data(), mipTailFirstLevel, numMipLevels, mipLevelDims.data(), m_stream ) );
}

TEST_F( TestTiledImageSource, tracksTileReadCount )
{
    ExpectationSet open{ expectOpen() };
    create();
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, m_baseInfo.width, m_baseInfo.height, _ ) ).WillOnce( Return( true ) );
    m_tiledImage->open( nullptr );
    const imageSource::Tile tile{ 0, 0, 64, 64 };
    const imageSource::Tile tile2{ 1, 0, 64, 64 };
    std::vector<char>       dest;
    dest.resize( tile.width * tile.height * 4 );
    m_tiledImage->readTile( dest.data(), 0, tile, m_stream );
    m_tiledImage->readTile( dest.data(), 0, tile2, m_stream );

    EXPECT_EQ( 2ULL, m_tiledImage->getNumTilesRead() );
}

namespace {

class TestTiledImageSourcePassThrough : public TestTiledImageSource
{
  public:
    ~TestTiledImageSourcePassThrough() override = default;

  protected:
    void SetUp() override
    {
        TestTiledImageSource::SetUp();
        m_open             = expectOpen();
        m_baseInfo.isTiled = true;
        create();
        m_tiledImage->open( nullptr );
    }

    ExpectationSet m_open;
};

}  // namespace

TEST_F( TestTiledImageSourcePassThrough, close )
{
    EXPECT_CALL( *m_baseImage, close() ).After( m_open );

    m_tiledImage->close();
}

TEST_F( TestTiledImageSourcePassThrough, getInfo )
{
    EXPECT_CALL( *m_baseImage, getInfo() ).After( m_open ).WillOnce( ReturnRef( m_baseInfo ) );

    const imageSource::TextureInfo info = m_tiledImage->getInfo();

    EXPECT_TRUE( info.isValid );
    EXPECT_TRUE( info.isTiled );
}

TEST_F( TestTiledImageSourcePassThrough, readTile )
{
    EXPECT_CALL( *m_baseImage, readTile( NotNull(), 1, imageSource::Tile{ 2, 3, 16, 16 }, m_stream ) ).After( m_open ).WillOnce( Return( true ) );

    char buffer{};
    EXPECT_TRUE( m_tiledImage->readTile( &buffer, 1, { 2, 3, 16, 16 }, m_stream ) );
}

TEST_F( TestTiledImageSourcePassThrough, readMipTail )
{
    char        buffer{};
    const uint2 dims{};
    EXPECT_CALL( *m_baseImage, readMipTail( &buffer, 1, 2, &dims, m_stream ) ).After( m_open ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_tiledImage->readMipTail( &buffer, 1, 2, &dims, m_stream ) );
}

TEST_F( TestTiledImageSourcePassThrough, getNumTilesRead )
{
    EXPECT_CALL( *m_baseImage, getNumTilesRead() ).After( m_open ).WillOnce( Return( 13 ) );

    EXPECT_EQ( 13, m_tiledImage->getNumTilesRead() );
}
