//
//  Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include "Textures/SparseTexture.h"
#include "Util/Exception.h"

#include <gtest/gtest.h>

#include <cuda.h>

using namespace demandLoading;
using namespace imageSource;

class TestSparseTexture : public testing::Test
{
  protected:
    unsigned int      m_deviceIndex = 0;
    TextureDescriptor m_desc;
    TextureInfo       m_info;

  public:
    void SetUp() override
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

        m_desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
        m_desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
        m_desc.filterMode       = CU_TR_FILTER_MODE_POINT;
        m_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        m_desc.maxAnisotropy    = 16;

        m_info.width        = 256;
        m_info.height       = 256;
        m_info.format       = CU_AD_FORMAT_FLOAT;
        m_info.numChannels  = 4;
        m_info.numMipLevels = 9;
    }

    void TearDown() override {}
};

TEST_F( TestSparseTexture, TestInit )
{
    SparseTexture texture( m_deviceIndex );
    EXPECT_FALSE( texture.isInitialized() );

    texture.init( m_desc, m_info );

    EXPECT_TRUE( texture.isInitialized() );
    EXPECT_EQ( 64U, texture.getTileWidth() );
    EXPECT_EQ( 64U, texture.getTileHeight() );
    EXPECT_EQ( 3U, texture.getMipTailFirstLevel() );
    EXPECT_LE( texture.getMipTailSize(), 65536U );

    for( unsigned int mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        EXPECT_EQ( m_info.width >> mipLevel, texture.getMipLevelDims( mipLevel ).x );
    }
}

// Note: the FillTile and FillMipTail methods are covered by TestDemandTexture.
