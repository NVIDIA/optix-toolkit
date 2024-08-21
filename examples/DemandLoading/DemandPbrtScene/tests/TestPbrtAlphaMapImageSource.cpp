// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/PbrtAlphaMapImageSource.h>

#include "MockImageSource.h"

#include <OptiXToolkit/Memory/BitCast.h>

#include <gmock/gmock.h>

#include <vector_functions.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace testing;

namespace {

using PbrtAlphaMapImageSourcePtr = std::shared_ptr<demandPbrtScene::PbrtAlphaMapImageSource>;

class TestPbrtAlphaMapImageSource : public Test
{
  public:
    ~TestPbrtAlphaMapImageSource() override = default;

  protected:
    void SetUp() override;

    ExpectationSet expectCreate();
    void           create();
    ExpectationSet expectOpen();

    otk::testing::MockImageSourcePtr m_baseImage{ std::make_shared<otk::testing::MockImageSource>() };
    imageSource::TextureInfo         m_baseInfo{};
    PbrtAlphaMapImageSourcePtr       m_alphaImage;
};

void TestPbrtAlphaMapImageSource::SetUp()
{
    m_baseInfo.width        = 256;
    m_baseInfo.height       = 256;
    m_baseInfo.format       = CU_AD_FORMAT_FLOAT;
    m_baseInfo.numChannels  = 3;
    m_baseInfo.numMipLevels = 1;
    m_baseInfo.isValid      = true;
    m_baseInfo.isTiled      = true;
}

ExpectationSet TestPbrtAlphaMapImageSource::expectCreate()
{
    ExpectationSet create;
    create += EXPECT_CALL( *m_baseImage, isOpen() ).WillRepeatedly( Return( false ) );
    return create;
}

void TestPbrtAlphaMapImageSource::create()
{
    m_alphaImage = std::make_shared<demandPbrtScene::PbrtAlphaMapImageSource>( m_baseImage );
}

ExpectationSet TestPbrtAlphaMapImageSource::expectOpen()
{
    ExpectationSet first{ expectCreate() };
    ExpectationSet open;
    open += EXPECT_CALL( *m_baseImage, open( IsNull() ) ).After( first );
    open += EXPECT_CALL( *m_baseImage, getInfo() ).After( first ).WillOnce( ReturnRef( m_baseInfo ) );
    return open;
}

}  // namespace

TEST_F( TestPbrtAlphaMapImageSource, create )
{
    expectCreate();

    create();

    EXPECT_TRUE( m_alphaImage );
}

TEST_F( TestPbrtAlphaMapImageSource, reportsSingleChannelUInt8 )
{
    imageSource::TextureInfo alphaInfo{};
    expectOpen();
    create();

    m_alphaImage->open( &alphaInfo );

    EXPECT_EQ( m_baseInfo.width, alphaInfo.width );
    EXPECT_EQ( m_baseInfo.height, alphaInfo.height );
    EXPECT_EQ( CU_AD_FORMAT_UNSIGNED_INT8, alphaInfo.format );
    EXPECT_EQ( 1, alphaInfo.numChannels );
    EXPECT_EQ( 1, alphaInfo.numMipLevels );
    EXPECT_TRUE( alphaInfo.isValid );
    EXPECT_TRUE( alphaInfo.isTiled );
}

namespace {

class TestPbrtAlphaMapImageSourceReadTile : public TestPbrtAlphaMapImageSource
{
  public:
    bool copySourcePixels( char* dest, unsigned int /*mipLevel*/, const imageSource::Tile& /*tile*/, CUstream /*stream*/ )
    {
        std::memcpy( dest, m_sourcePixels.data(), m_sourcePixels.size() * sizeof( float ) );
        return true;
    }

  protected:
    void SetUp() override;

    imageSource::Tile         m_tile{ 10, 11, 16, 16 };
    unsigned int              m_mipLevel{ 1 };
    CUstream                  m_stream{ otk::bit_cast<CUstream>( 0xdeadbeefULL ) };
    std::vector<float>        m_sourcePixels;
    std::uint8_t              m_initialAlpha{ 32U };
    std::vector<std::uint8_t> m_alphaPixels;
    ExpectationSet            m_opened;
};

void TestPbrtAlphaMapImageSourceReadTile::SetUp()
{
    TestPbrtAlphaMapImageSource::SetUp();
    m_opened = expectOpen();
    create();
    m_alphaImage->open( nullptr );
    m_sourcePixels.resize( m_tile.width * m_tile.height * m_baseInfo.numChannels );
    m_alphaPixels.resize( m_tile.width * m_tile.height );
    std::fill( m_alphaPixels.begin(), m_alphaPixels.end(), m_initialAlpha );
}

}  // namespace

TEST_F( TestPbrtAlphaMapImageSourceReadTile, convertsPixelsNonZero )
{
    std::fill( m_sourcePixels.begin(), m_sourcePixels.end(), 0.3141592427f );
    EXPECT_CALL( *m_baseImage, readTile( NotNull(), m_mipLevel, m_tile, m_stream ) )
        .After( m_opened )
        .WillOnce( DoAll( Invoke( this, &TestPbrtAlphaMapImageSourceReadTile::copySourcePixels ), Return( true ) ) );

    EXPECT_TRUE( m_alphaImage->readTile( reinterpret_cast<char*>( m_alphaPixels.data() ), m_mipLevel, m_tile, m_stream ) );

    EXPECT_EQ( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), m_initialAlpha ) );
    EXPECT_EQ( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 0U ) );
    EXPECT_NE( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 255U ) );
}

TEST_F( TestPbrtAlphaMapImageSourceReadTile, convertsPixelsZero )
{
    std::fill( m_sourcePixels.begin(), m_sourcePixels.end(), 0.0f );
    EXPECT_CALL( *m_baseImage, readTile( NotNull(), m_mipLevel, m_tile, m_stream ) )
        .After( m_opened )
        .WillOnce( DoAll( Invoke( this, &TestPbrtAlphaMapImageSourceReadTile::copySourcePixels ), Return( true ) ) );

    EXPECT_TRUE( m_alphaImage->readTile( reinterpret_cast<char*>( m_alphaPixels.data() ), m_mipLevel, m_tile, m_stream ) );

    EXPECT_EQ( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), m_initialAlpha ) );
    EXPECT_NE( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 0U ) );
    EXPECT_EQ( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 255U ) );
}

TEST_F( TestPbrtAlphaMapImageSourceReadTile, convertsPixelsMixed )
{
    int count{ 0 };
    std::generate( m_sourcePixels.begin(), m_sourcePixels.end(), [&] {
        // {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, ...
        const float val = count % ( m_baseInfo.numChannels * 2 ) == 0 ? 1.0f : 0.0f;
        ++count;
        return val;
    } );
    EXPECT_CALL( *m_baseImage, readTile( NotNull(), m_mipLevel, m_tile, m_stream ) )
        .After( m_opened )
        .WillOnce( DoAll( Invoke( this, &TestPbrtAlphaMapImageSourceReadTile::copySourcePixels ), Return( true ) ) );

    EXPECT_TRUE( m_alphaImage->readTile( reinterpret_cast<char*>( m_alphaPixels.data() ), m_mipLevel, m_tile, m_stream ) );

    EXPECT_EQ( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), m_initialAlpha ) );
    EXPECT_NE( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 0U ) );
    EXPECT_NE( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 255U ) );
}

namespace {

class TestPbrtAlphaMapImageSourceReadMipLevel : public TestPbrtAlphaMapImageSource
{
  public:
    bool copySourcePixels( char* dest, unsigned int /*mipLevel*/, unsigned int /*width*/, unsigned int /*height*/, CUstream /*m_stream*/ )
    {
        std::memcpy( dest, m_sourcePixels.data(), m_sourcePixels.size() * sizeof( float ) );
        return true;
    }

  protected:
    void SetUp() override;

    const unsigned int        m_mipLevel{ 1 };
    CUstream                  m_stream{ otk::bit_cast<CUstream>( 0xdeadbeefULL ) };
    std::vector<float>        m_sourcePixels;
    std::uint8_t              m_initialAlpha{ 32U };
    std::vector<std::uint8_t> m_alphaPixels;
    ExpectationSet            m_opened;
};

void TestPbrtAlphaMapImageSourceReadMipLevel::SetUp()
{
    TestPbrtAlphaMapImageSource::SetUp();
    m_opened = expectOpen();
    create();
    m_alphaImage->open( nullptr );
    m_sourcePixels.resize( m_baseInfo.width * m_baseInfo.height * m_baseInfo.numChannels );
    m_alphaPixels.resize( m_baseInfo.width * m_baseInfo.height );
    std::fill( m_alphaPixels.begin(), m_alphaPixels.end(), m_initialAlpha );
}

}  // namespace

TEST_F( TestPbrtAlphaMapImageSourceReadMipLevel, convertsMipLevel )
{
    int count{ 0 };
    std::generate( m_sourcePixels.begin(), m_sourcePixels.end(), [&] {
        // {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, ...
        const float val = count % ( m_baseInfo.numChannels * 2 ) == 0 ? 1.0f : 0.0f;
        ++count;
        return val;
    } );
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), m_mipLevel, m_baseInfo.width, m_baseInfo.height, m_stream ) )
        .After( m_opened )
        .WillOnce( DoAll( Invoke( this, &TestPbrtAlphaMapImageSourceReadMipLevel::copySourcePixels ), Return( true ) ) );

    EXPECT_TRUE( m_alphaImage->readMipLevel( reinterpret_cast<char*>( m_alphaPixels.data() ), m_mipLevel,
                                             m_baseInfo.width, m_baseInfo.height, m_stream ) );

    EXPECT_EQ( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), m_initialAlpha ) );
    EXPECT_NE( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 0U ) );
    EXPECT_NE( m_alphaPixels.end(), std::find( m_alphaPixels.begin(), m_alphaPixels.end(), 255U ) );
}
