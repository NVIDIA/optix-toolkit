// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// gtest has to come before pbrt stuff
#include <gtest/gtest.h>

#include <DemandPbrtScene/PbrtCheckerboardImageSource.h>

#include <vector>

using namespace demandPbrtScene;

namespace {

PbrtCheckerboardDefinition checkerboardDefinition()
{
    return { float4{ 1.0f, 0.0f, 0.0f, 1.0f },
             float4{ 0.0f, 1.0f, 0.0f, 1.0f },
             2.0f,
             2.0f,
             0.0f,
             0.0f };
}

void expectFloat4Eq( const float4& expected, const float4& actual )
{
    EXPECT_FLOAT_EQ( expected.x, actual.x );
    EXPECT_FLOAT_EQ( expected.y, actual.y );
    EXPECT_FLOAT_EQ( expected.z, actual.z );
    EXPECT_FLOAT_EQ( expected.w, actual.w );
}

}  // namespace

TEST( TestPbrtCheckerboardImageSource, keyRoundTrip )
{
    const PbrtCheckerboardDefinition definition{ checkerboardDefinition() };
    const std::string                key{ makePbrtCheckerboardTextureKey( definition ) };

    EXPECT_TRUE( isPbrtCheckerboardTextureKey( key ) );
    EXPECT_EQ( definition, parsePbrtCheckerboardTextureKey( key ) );
}

TEST( TestPbrtCheckerboardImageSource, differentScaleProducesDifferentKey )
{
    PbrtCheckerboardDefinition first{ checkerboardDefinition() };
    PbrtCheckerboardDefinition second{ first };
    second.uscale = 4.0f;

    EXPECT_NE( makePbrtCheckerboardTextureKey( first ), makePbrtCheckerboardTextureKey( second ) );
}

TEST( TestPbrtCheckerboardImageSource, openReportsExplicitVirtualTextureShape )
{
    PbrtCheckerboardImageSource image{ checkerboardDefinition() };
    imageSource::TextureInfo    info{};

    image.open( &info );

    EXPECT_EQ( PBRT_CHECKERBOARD_TEXTURE_SIZE, info.width );
    EXPECT_EQ( PBRT_CHECKERBOARD_TEXTURE_SIZE, info.height );
    EXPECT_EQ( CU_AD_FORMAT_FLOAT, info.format );
    EXPECT_EQ( 4U, info.numChannels );
    EXPECT_EQ( imageSource::calculateNumMipLevels( PBRT_CHECKERBOARD_TEXTURE_SIZE, PBRT_CHECKERBOARD_TEXTURE_SIZE ),
               info.numMipLevels );
    EXPECT_TRUE( info.isValid );
    EXPECT_TRUE( info.isTiled );
}

TEST( TestPbrtCheckerboardImageSource, readMipLevelReturnsCheckerValues )
{
    PbrtCheckerboardImageSource image{ checkerboardDefinition() };
    std::vector<float4>         pixels( 16 );

    ASSERT_TRUE( image.readMipLevel( reinterpret_cast<char*>( pixels.data() ), 0, 4, 4, nullptr ) );

    expectFloat4Eq( float4{ 1.0f, 0.0f, 0.0f, 1.0f }, pixels[0] );
    expectFloat4Eq( float4{ 0.0f, 1.0f, 0.0f, 1.0f }, pixels[2] );
    expectFloat4Eq( float4{ 0.0f, 1.0f, 0.0f, 1.0f }, pixels[8] );
    expectFloat4Eq( float4{ 1.0f, 0.0f, 0.0f, 1.0f }, pixels[10] );
}

TEST( TestPbrtCheckerboardImageSource, readMipLevelUsesSamePatternAtMipLevels )
{
    PbrtCheckerboardImageSource image{ checkerboardDefinition() };
    std::vector<float4>         pixels( 16 );

    ASSERT_TRUE( image.readMipLevel( reinterpret_cast<char*>( pixels.data() ), 3, 4, 4, nullptr ) );

    expectFloat4Eq( float4{ 1.0f, 0.0f, 0.0f, 1.0f }, pixels[0] );
    expectFloat4Eq( float4{ 0.0f, 1.0f, 0.0f, 1.0f }, pixels[2] );
    expectFloat4Eq( float4{ 0.0f, 1.0f, 0.0f, 1.0f }, pixels[8] );
    expectFloat4Eq( float4{ 1.0f, 0.0f, 0.0f, 1.0f }, pixels[10] );
}
