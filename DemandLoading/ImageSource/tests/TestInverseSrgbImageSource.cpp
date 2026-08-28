// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/InverseSrgbImageSource.h>

#include <OptiXToolkit/ImageSource/Testing/MockImageSource.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector_functions.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace imageSource;
using namespace testing;

namespace {

float inverseSrgb( float value )
{
    return value <= 0.04045f ? value / 12.92f : std::pow( ( value + 0.055f ) / 1.055f, 2.4f );
}

class TestInverseSrgbImageSource : public Test
{
  protected:
    void SetUp() override
    {
        m_baseInfo.width        = 2;
        m_baseInfo.height       = 1;
        m_baseInfo.format       = CU_AD_FORMAT_UNSIGNED_INT8;
        m_baseInfo.numChannels  = 4;
        m_baseInfo.numMipLevels = 1;
        m_baseInfo.isValid      = true;
        m_baseInfo.isTiled      = true;

        EXPECT_CALL( *m_baseImage, isOpen() ).WillRepeatedly( Return( false ) );
        m_image = std::make_shared<InverseSrgbImageSource>( m_baseImage );
    }

    void expectOpen()
    {
        EXPECT_CALL( *m_baseImage, open( IsNull() ) );
        EXPECT_CALL( *m_baseImage, getFillType() ).WillOnce( Return( CU_MEMORYTYPE_HOST ) );
        EXPECT_CALL( *m_baseImage, getInfo() ).WillOnce( ReturnRef( m_baseInfo ) );
    }

    void open()
    {
        expectOpen();
        m_image->open( nullptr );
    }

    std::shared_ptr<otk::testing::MockImageSource> m_baseImage{ std::make_shared<otk::testing::MockImageSource>() };
    TextureInfo                                    m_baseInfo{};
    std::shared_ptr<InverseSrgbImageSource>        m_image;
};

}  // namespace

TEST_F( TestInverseSrgbImageSource, reportsFloatPixelsAndPreservesImageShape )
{
    expectOpen();

    TextureInfo info{};
    m_image->open( &info );

    EXPECT_EQ( CU_AD_FORMAT_FLOAT, info.format );
    EXPECT_EQ( m_baseInfo.width, info.width );
    EXPECT_EQ( m_baseInfo.height, info.height );
    EXPECT_EQ( m_baseInfo.numChannels, info.numChannels );
    EXPECT_EQ( m_baseInfo.numMipLevels, info.numMipLevels );
    EXPECT_EQ( m_baseInfo.isTiled, info.isTiled );
    EXPECT_EQ( info, m_image->getInfo() );
}

TEST_F( TestInverseSrgbImageSource, convertsTileRgbToLinearAndPreservesAlpha )
{
    open();
    const std::vector<std::uint8_t> source{ 0U, 10U, 128U, 64U, 255U, 192U, 32U, 255U };
    const Tile                      tile{ 0U, 0U, 2U, 1U };
    EXPECT_CALL( *m_baseImage, readTile( NotNull(), 0U, tile, CUstream{} ) )
        .WillOnce( DoAll( SetArrayArgument<0>( source.begin(), source.end() ), Return( true ) ) );
    std::vector<float> result( source.size() );

    ASSERT_TRUE( m_image->readTile( reinterpret_cast<char*>( result.data() ), 0U, tile, CUstream{} ) );

    for( size_t i = 0; i < source.size(); ++i )
    {
        const float normalized{ static_cast<float>( source[i] ) / 255.0f };
        const float expected{ i % 4U == 3U ? normalized : inverseSrgb( normalized ) };
        EXPECT_FLOAT_EQ( expected, result[i] );
    }
}

TEST_F( TestInverseSrgbImageSource, preservesSecondChannelOfTwoChannelImage )
{
    m_baseInfo.numChannels = 2;
    open();
    const std::vector<std::uint8_t> source{ 128U, 64U, 32U, 255U };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0U, 2U, 1U, CUstream{} ) )
        .WillOnce( DoAll( SetArrayArgument<0>( source.begin(), source.end() ), Return( true ) ) );
    std::vector<float> result( source.size() );

    ASSERT_TRUE( m_image->readMipLevel( reinterpret_cast<char*>( result.data() ), 0U, 2U, 1U, CUstream{} ) );

    EXPECT_FLOAT_EQ( inverseSrgb( 128.0f / 255.0f ), result[0] );
    EXPECT_FLOAT_EQ( 64.0f / 255.0f, result[1] );
    EXPECT_FLOAT_EQ( inverseSrgb( 32.0f / 255.0f ), result[2] );
    EXPECT_FLOAT_EQ( 1.0f, result[3] );
}

TEST_F( TestInverseSrgbImageSource, convertsBaseColor )
{
    open();
    EXPECT_CALL( *m_baseImage, readBaseColor( _ ) )
        .WillOnce( DoAll( SetArgReferee<0>( make_float4( 128.0f, 64.0f, 32.0f, 255.0f ) ), Return( true ) ) );
    float4 result{};

    ASSERT_TRUE( m_image->readBaseColor( result ) );

    EXPECT_FLOAT_EQ( inverseSrgb( 128.0f / 255.0f ), result.x );
    EXPECT_FLOAT_EQ( inverseSrgb( 64.0f / 255.0f ), result.y );
    EXPECT_FLOAT_EQ( inverseSrgb( 32.0f / 255.0f ), result.z );
    EXPECT_FLOAT_EQ( 1.0f, result.w );
}
