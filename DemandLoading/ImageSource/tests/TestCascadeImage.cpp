// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/CascadeImage.h>
#include <OptiXToolkit/DemandLoading/TextureCascade.h>

#include "MockImageSource.h"

#include <OptiXToolkit/Memory/BitCast.h>

#include <gtest/gtest.h>

#include <memory>

using namespace testing;
using namespace imageSource;

namespace {

class TestCascadeImage : public Test
{
  public:
    ~TestCascadeImage() override = default;

  protected:
    void SetUp() override;

    otk::testing::MockImageSourcePtr m_backingImage{ std::make_shared<otk::testing::MockImageSource>() };
    TextureInfo                      m_backingInfo{};
    std::shared_ptr<CascadeImage>    m_cascadeImage;
    CUstream                         m_stream{ otk::bit_cast<CUstream>( 0xdeadbeefULL ) };
};

void TestCascadeImage::SetUp()
{
    m_backingInfo.width        = 1024;
    m_backingInfo.height       = 1024;
    m_backingInfo.format       = CU_AD_FORMAT_UNSIGNED_INT8;
    m_backingInfo.numChannels  = 4;
    m_backingInfo.numMipLevels = 11;
    m_backingInfo.isValid      = true;
    m_backingInfo.isTiled      = true;
}

}  // namespace

TEST_F( TestCascadeImage, open )
{
    const unsigned int minDim = 64;
    EXPECT_CALL( *m_backingImage, open( NotNull() ) )
        .WillOnce( SetArgPointee<0>( m_backingInfo ) );
    m_cascadeImage = std::make_shared<CascadeImage>( m_backingImage, minDim );

    TextureInfo info{};
    m_cascadeImage->open( &info );

    EXPECT_TRUE( info.isValid );
    EXPECT_EQ( 64u, info.width );
    EXPECT_EQ( 64u, info.height );
    EXPECT_EQ( 7u, info.numMipLevels );
    EXPECT_EQ( CU_AD_FORMAT_UNSIGNED_INT8, info.format );
    EXPECT_EQ( 4u, info.numChannels );
}

TEST_F( TestCascadeImage, hasCascade )
{
    const unsigned int minDim = 64;
    EXPECT_CALL( *m_backingImage, open( NotNull() ) )
        .WillOnce( SetArgPointee<0>( m_backingInfo ) );
    EXPECT_CALL( *m_backingImage, getInfo() ).WillRepeatedly( ReturnRef( m_backingInfo ) );
    m_cascadeImage = std::make_shared<CascadeImage>( m_backingImage, minDim );
    m_cascadeImage->open( nullptr );

    EXPECT_TRUE( m_cascadeImage->hasCascade() );
}

TEST_F( TestCascadeImage, noCascade )
{
    const unsigned int minDim = 1024;
    EXPECT_CALL( *m_backingImage, open( NotNull() ) )
        .WillOnce( SetArgPointee<0>( m_backingInfo ) );
    EXPECT_CALL( *m_backingImage, getInfo() ).WillRepeatedly( ReturnRef( m_backingInfo ) );
    m_cascadeImage = std::make_shared<CascadeImage>( m_backingImage, minDim );
    m_cascadeImage->open( nullptr );

    EXPECT_FALSE( m_cascadeImage->hasCascade() );
}

TEST_F( TestCascadeImage, readTile )
{
    const unsigned int minDim = 64;
    EXPECT_CALL( *m_backingImage, open( NotNull() ) )
        .WillOnce( SetArgPointee<0>( m_backingInfo ) );
    m_cascadeImage = std::make_shared<CascadeImage>( m_backingImage, minDim );
    m_cascadeImage->open( nullptr );

    // readTile at mip level 0 should delegate to backing image at mip level 4 (backingMipLevel)
    char buffer{};
    EXPECT_CALL( *m_backingImage, readTile( &buffer, 4, Tile{ 0, 0, 64, 64 }, m_stream ) )
        .WillOnce( Return( true ) );

    EXPECT_TRUE( m_cascadeImage->readTile( &buffer, 0, { 0, 0, 64, 64 }, m_stream ) );
}

TEST_F( TestCascadeImage, readMipLevel )
{
    const unsigned int minDim = 64;
    EXPECT_CALL( *m_backingImage, open( NotNull() ) )
        .WillOnce( SetArgPointee<0>( m_backingInfo ) );
    m_cascadeImage = std::make_shared<CascadeImage>( m_backingImage, minDim );
    m_cascadeImage->open( nullptr );

    // readMipLevel at mip level 1 should delegate to backing image at mip level 5
    char buffer{};
    EXPECT_CALL( *m_backingImage, readMipLevel( &buffer, 5, 32, 32, m_stream ) )
        .WillOnce( Return( true ) );

    EXPECT_TRUE( m_cascadeImage->readMipLevel( &buffer, 1, 32, 32, m_stream ) );
}

TEST_F( TestCascadeImage, readBaseColor )
{
    const unsigned int minDim = 64;
    EXPECT_CALL( *m_backingImage, open( NotNull() ) )
        .WillOnce( SetArgPointee<0>( m_backingInfo ) );
    m_cascadeImage = std::make_shared<CascadeImage>( m_backingImage, minDim );
    m_cascadeImage->open( nullptr );

    float4 baseColor{};
    EXPECT_CALL( *m_backingImage, readBaseColor( _ ) ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_cascadeImage->readBaseColor( baseColor ) );
}

// Tests for getCascadeLevel free function from TextureCascade.h

TEST( TestGetCascadeLevel, smallTexture )
{
    // 64x64 fits in CASCADE_BASE (64), so level 0
    EXPECT_EQ( 0u, demandLoading::getCascadeLevel( 64, 64 ) );
}

TEST( TestGetCascadeLevel, largeTexture )
{
    // 128x128 needs CASCADE_BASE << 1 = 128, so level 1
    EXPECT_EQ( 1u, demandLoading::getCascadeLevel( 128, 128 ) );
}

TEST( TestGetCascadeLevel, maxTexture )
{
    // Very large texture clamps to NUM_CASCADES - 1
    EXPECT_EQ( NUM_CASCADES - 1, demandLoading::getCascadeLevel( 100000, 100000 ) );
}

TEST( TestGetCascadeLevel, asymmetric )
{
    // 64x128: max dimension is 128, needs level 1
    EXPECT_EQ( 1u, demandLoading::getCascadeLevel( 64, 128 ) );
}
