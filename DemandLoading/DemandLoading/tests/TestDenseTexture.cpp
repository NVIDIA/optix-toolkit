// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Textures/DenseTexture.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <gtest/gtest.h>

#include <cuda.h>

using namespace demandLoading;
using namespace imageSource;

class TestDenseTexture : public testing::Test
{
  protected:
    unsigned int      m_deviceIndex = 0;
    TextureDescriptor m_desc;
    TextureInfo       m_info;

  public:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );

        m_desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
        m_desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
        m_desc.filterMode       = CU_TR_FILTER_MODE_POINT;
        m_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        m_desc.maxAnisotropy    = 16;

        m_info.width        = 32;
        m_info.height       = 32;
        m_info.format       = CU_AD_FORMAT_FLOAT;
        m_info.numChannels  = 4;
        m_info.numMipLevels = 6;
    }

    void TearDown() override {}
};

TEST_F( TestDenseTexture, TestInit )
{
    DenseTexture texture;
    EXPECT_FALSE( texture.isInitialized() );

    texture.init( m_desc, m_info, nullptr );

    EXPECT_TRUE( texture.isInitialized() );

    for( unsigned int mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        EXPECT_EQ( m_info.width >> mipLevel, texture.getMipLevelDims( mipLevel ).x );
    }
}

// Note: the FillTile and FillMipTail methods are covered by TestDemandTexture.
