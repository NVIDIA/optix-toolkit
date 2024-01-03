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

#include <OptiXToolkit/ImageSource/MipMapImageSource.h>

#include "MockImageSource.h"

#include <OptiXToolkit/Memory/BitCast.h>

#include <gtest/gtest.h>

#include <vector_functions.h>

#include <algorithm>

using namespace testing;

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
    m_mipMapImage           = std::make_shared<imageSource::MipMapImageSource>( m_baseImage );
}

}  // namespace

TEST_F( TestMipMapImageSource, create )
{
    EXPECT_TRUE( m_mipMapImage );
}

TEST_F( TestMipMapImageSource, closeResetsInfo )
{
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, close() );

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
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, getInfo() ).Times( 0 );

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
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, getInfo() ).Times( 0 );

    m_mipMapImage->open( nullptr );
    const imageSource::TextureInfo result = m_mipMapImage->getInfo();

    EXPECT_EQ( mipInfo, result );
}

TEST_F( TestMipMapImageSource, getInfoWithoutOpenIsInvalid )
{
    const imageSource::TextureInfo result = m_mipMapImage->getInfo();

    ASSERT_FALSE( result.isValid );
}

TEST_F( TestMipMapImageSource, readTileMipLevelZeroSourcesDataFromReadMipLevelZero )
{
    EXPECT_CALL( *m_baseImage, open( _ ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes{ imageSource::getBytesPerChannel( m_baseInfo.format ) * m_baseInfo.numChannels };
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
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, mipLevelWidth, mipLevelHeight, m_stream ) )
        .WillOnce( DoAll( fillMipLevel, Return( true ) ) );
    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );
    ASSERT_TRUE( info.isValid );

    const unsigned int tileX{ 0 };
    const unsigned int tileY{ 0 };
    const unsigned int tileWidth{ 64 };
    const unsigned int tileHeight{ 64 };
    std::vector<char>  dest;
    dest.resize( tileWidth * tileHeight * 4 );
    ASSERT_TRUE( m_mipMapImage->readTile( dest.data(), 0, tileX, tileY, tileWidth, tileHeight, m_stream ) );

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
    EXPECT_CALL( *m_baseImage, open( _ ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes{ imageSource::getBytesPerChannel( m_baseInfo.format ) * m_baseInfo.numChannels };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, mipLevelWidth, mipLevelHeight, m_stream ) ).WillOnce( Return( true ) );
    imageSource::TextureInfo info{};
    m_mipMapImage->open( &info );
    ASSERT_TRUE( info.isValid );

    const unsigned int tileX{ 0 };
    const unsigned int tileY{ 0 };
    const unsigned int tileWidth{ 64 };
    const unsigned int tileHeight{ 64 };
    std::vector<char>  dest;
    dest.resize( tileWidth * tileHeight * pixelSizeInBytes );
    ASSERT_TRUE( m_mipMapImage->readTile( dest.data(), 1, tileX, tileY, tileWidth, tileHeight, m_stream ) );
}

TEST_F( TestMipMapImageSource, readMipLevelOneSourcesDataFromMipLevelZero )
{
    EXPECT_CALL( *m_baseImage, open( _ ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes{ imageSource::getBytesPerChannel( m_baseInfo.format ) * m_baseInfo.numChannels };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, mipLevelWidth, mipLevelHeight, m_stream ) ).WillOnce( Return( true ) );
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
    imageSource::TextureInfo baseInfo{ m_baseInfo };
    baseInfo.width        = 16;
    baseInfo.height       = 16;
    baseInfo.numMipLevels = 1;
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( baseInfo ) );
    const unsigned int mipTailFirstLevel{ 0 };
    const unsigned int numMipLevels{ 5 };
    std::vector<uint2> mipLevelDims;
    unsigned int       size = 16;
    const unsigned int pixelSizeInBytes{ imageSource::getBytesPerChannel( baseInfo.format ) * baseInfo.numChannels };
    for( unsigned int i = 0; i < numMipLevels; ++i )
    {
        mipLevelDims.push_back( make_uint2( size, size ) );
        size /= 2;
    }
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, baseInfo.width, baseInfo.height, m_stream ) ).WillOnce( Return( true ) );

    m_mipMapImage->open( nullptr );
    std::vector<char> dest;
    imageSource::TextureInfo mippedInfo{ baseInfo };
    mippedInfo.numMipLevels = numMipLevels;
    dest.resize( getTextureSizeInBytes( mippedInfo ) );
    EXPECT_TRUE( m_mipMapImage->readMipTail( dest.data(), mipTailFirstLevel, numMipLevels, mipLevelDims.data(),
                                             pixelSizeInBytes, m_stream ) );
}

TEST_F( TestMipMapImageSource, tracksTileReadCount )
{
    EXPECT_CALL( *m_baseImage, open( _ ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, m_baseInfo.width, m_baseInfo.height, _ ) ).WillOnce( Return( true ) );
    m_mipMapImage->open( nullptr );
    const unsigned int tileX{ 0 };
    const unsigned int tileY{ 0 };
    const unsigned int tileWidth{ 64 };
    const unsigned int tileHeight{ 64 };
    std::vector<char>  dest;
    dest.resize( tileWidth * tileHeight * 4 );
    m_mipMapImage->readTile( dest.data(), 0, tileX, tileY, tileWidth, tileHeight, m_stream );
    m_mipMapImage->readTile( dest.data(), 0, tileX + tileX, tileY, tileWidth, tileHeight, m_stream );

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
        EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
        m_mipMapImage->open( nullptr );
    }
};

}  // namespace

TEST_F( TestMipMapImageSourcePassThrough, open )
{
}

TEST_F( TestMipMapImageSourcePassThrough, close )
{
    EXPECT_CALL( *m_baseImage, close() );

    m_mipMapImage->close();
}

TEST_F( TestMipMapImageSourcePassThrough, getInfo )
{
    EXPECT_CALL( *m_baseImage, getInfo() ).WillOnce( ReturnRef( m_baseInfo ) );

    const imageSource::TextureInfo info = m_mipMapImage->getInfo();

    EXPECT_TRUE( info.isValid );
    EXPECT_LT( 1, info.numMipLevels );
}

TEST_F( TestMipMapImageSourcePassThrough, readTile )
{
    char buffer{};
    EXPECT_CALL( *m_baseImage, readTile( &buffer, 1, 2, 3, 16, 16, m_stream ) ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_mipMapImage->readTile( &buffer, 1, 2, 3, 16, 16, m_stream ) );
}

TEST_F( TestMipMapImageSourcePassThrough, readMipLevel )
{
    char buffer{};
    EXPECT_CALL( *m_baseImage, readMipLevel( &buffer, 2, 16, 32, m_stream ) ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_mipMapImage->readMipLevel( &buffer, 2, 16, 32, m_stream ) );
}

TEST_F( TestMipMapImageSourcePassThrough, readMipTail )
{
    char        buffer{};
    const uint2 dims{};
    EXPECT_CALL( *m_baseImage, readMipTail( &buffer, 1, 2, &dims, 4, m_stream ) ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_mipMapImage->readMipTail( &buffer, 1, 2, &dims, 4, m_stream ) );
}

TEST_F( TestMipMapImageSourcePassThrough, getNumTilesRead )
{
    EXPECT_CALL( *m_baseImage, getNumTilesRead() ).WillOnce( Return( 13 ) );

    EXPECT_EQ( 13, m_mipMapImage->getNumTilesRead() );
}
