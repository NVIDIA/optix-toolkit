//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
