//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <DemandTextureCache.h>

#include <Dependencies.h>
#include <ImageSourceFactory.h>

#include <OptiXToolkit/DemandGeometry/Mocks/MockDemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <gmock/gmock.h>

#include "MockImageSource.h"

using namespace testing;
using namespace demandPbrtScene;
using Stats = DemandTextureCacheStatistics;

class MockDemandTexture : public StrictMock<demandLoading::DemandTexture>
{
  public:
    ~MockDemandTexture() override = default;

    MOCK_METHOD( unsigned, getId, (), ( const, override ) );
};

class MockImageSourceFactory : public StrictMock<ImageSourceFactory>
{
  public:
    ~MockImageSourceFactory() override = default;

    MOCK_METHOD( ImageSourcePtr, createDiffuseImageFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( ImageSourcePtr, createAlphaImageFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( ImageSourcePtr, createSkyboxImageFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( ImageSourceFactoryStatistics, getStatistics, (), ( const override ) );
};

static demandLoading::TextureDescriptor expectedTextureDesc()
{
    demandLoading::TextureDescriptor textureDesc{};
    textureDesc.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
    textureDesc.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
    textureDesc.filterMode       = CU_TR_FILTER_MODE_POINT;
    textureDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    textureDesc.maxAnisotropy    = 1;
    return textureDesc;
}

class TestDemandTextureCache : public Test
{
  public:
    ~TestDemandTextureCache() override = default;

  protected:
    void expectDiffuseTextureCreated()
    {
        EXPECT_CALL( *m_imageSourceFactory, createDiffuseImageFromFile( m_path ) ).WillOnce( Return( m_diffuseImage ) );
        EXPECT_CALL( *m_demandLoader, createTexture( static_cast<ImageSourcePtr>( m_diffuseImage ), expectedTextureDesc() ) )
            .WillOnce( ReturnRef( m_diffuseTexture ) );
        EXPECT_CALL( m_diffuseTexture, getId() ).WillOnce( Return( m_diffuseTextureId ) );
    }
    void expectAlphaTextureCreated()
    {
        EXPECT_CALL( *m_imageSourceFactory, createAlphaImageFromFile( m_path ) ).WillOnce( Return( m_alphaImage ) );
        EXPECT_CALL( *m_demandLoader, createTexture( static_cast<ImageSourcePtr>( m_alphaImage ), expectedTextureDesc() ) )
            .WillOnce( ReturnRef( m_alphaTexture ) );
        EXPECT_CALL( m_alphaTexture, getId() ).WillOnce( Return( m_alphaTextureId ) );
    }
    void expectSkyboxTextureCreated()
    {
        EXPECT_CALL( *m_imageSourceFactory, createSkyboxImageFromFile( m_path ) ).WillOnce( Return( m_skyboxImage ) );
        EXPECT_CALL( *m_demandLoader, createTexture( static_cast<ImageSourcePtr>( m_skyboxImage ), expectedTextureDesc() ) )
            .WillOnce( ReturnRef( m_skyboxTexture ) );
        EXPECT_CALL( m_skyboxTexture, getId() ).WillOnce( Return( m_skyboxTextureId ) );
    }

    std::shared_ptr<otk::testing::MockDemandLoader> m_demandLoader{ std::make_shared<otk::testing::MockDemandLoader>() };
    std::shared_ptr<MockImageSourceFactory>        m_imageSourceFactory{ std::make_shared<MockImageSourceFactory>() };
    std::shared_ptr<otk::testing::MockImageSource> m_diffuseImage{ std::make_shared<otk::testing::MockImageSource>() };
    std::shared_ptr<otk::testing::MockImageSource> m_alphaImage{ std::make_shared<otk::testing::MockImageSource>() };
    std::shared_ptr<otk::testing::MockImageSource> m_skyboxImage{ std::make_shared<otk::testing::MockImageSource>() };
    std::string                                    m_path{ "mock.png" };
    MockDemandTexture                              m_diffuseTexture;
    MockDemandTexture                              m_alphaTexture;
    MockDemandTexture                              m_skyboxTexture;
    std::shared_ptr<DemandTextureCache> m_cache{ createDemandTextureCache( m_demandLoader, m_imageSourceFactory ) };
    const uint_t                        m_diffuseTextureId{ 5678 };
    const uint_t                        m_alphaTextureId{ 1234 };
    const uint_t                        m_skyboxTextureId{ 9012 };
};

TEST_F( TestDemandTextureCache, createDiffuseTexture )
{
    expectDiffuseTextureCreated();

    const uint_t result = m_cache->createDiffuseTextureFromFile( m_path );
    const Stats  stats  = m_cache->getStatistics();

    EXPECT_EQ( m_diffuseTextureId, result );
    EXPECT_EQ( 1, stats.numDiffuseTexturesCreated );
}

TEST_F( TestDemandTextureCache, createAlphaTexture )
{
    expectAlphaTextureCreated();

    const uint_t result = m_cache->createAlphaTextureFromFile( m_path );
    const Stats  stats  = m_cache->getStatistics();

    EXPECT_EQ( m_alphaTextureId, result );
    EXPECT_EQ( 1, stats.numAlphaTexturesCreated );
}

TEST_F( TestDemandTextureCache, createSkyboxTexture )
{
    expectSkyboxTextureCreated();

    const uint_t result = m_cache->createSkyboxTextureFromFile( m_path );
    const Stats  stats  = m_cache->getStatistics();

    EXPECT_EQ( m_skyboxTextureId, result );
    EXPECT_EQ( 1, stats.numSkyboxTexturesCreated );
}

TEST_F( TestDemandTextureCache, cachesDiffuseTexture )
{
    expectDiffuseTextureCreated();

    const uint_t result       = m_cache->createDiffuseTextureFromFile( m_path );
    const uint_t cachedResult = m_cache->createDiffuseTextureFromFile( m_path );
    const Stats  stats        = m_cache->getStatistics();

    EXPECT_EQ( m_diffuseTextureId, result );
    EXPECT_EQ( m_diffuseTextureId, cachedResult );
    EXPECT_EQ( 1, stats.numDiffuseTexturesCreated );
}

TEST_F( TestDemandTextureCache, cachesAlphaTexture )
{
    expectAlphaTextureCreated();

    const uint_t result       = m_cache->createAlphaTextureFromFile( m_path );
    const uint_t cachedResult = m_cache->createAlphaTextureFromFile( m_path );
    const Stats  stats        = m_cache->getStatistics();

    EXPECT_EQ( m_alphaTextureId, result );
    EXPECT_EQ( m_alphaTextureId, cachedResult );
    EXPECT_EQ( 1, stats.numAlphaTexturesCreated );
}

TEST_F( TestDemandTextureCache, cachesSkyboxTexture )
{
    expectSkyboxTextureCreated();

    const uint_t result       = m_cache->createSkyboxTextureFromFile( m_path );
    const uint_t cachedResult = m_cache->createSkyboxTextureFromFile( m_path );
    const Stats  stats        = m_cache->getStatistics();

    EXPECT_EQ( m_skyboxTextureId, result );
    EXPECT_EQ( m_skyboxTextureId, cachedResult );
    EXPECT_EQ( 1, stats.numSkyboxTexturesCreated );
}

TEST_F( TestDemandTextureCache, differentTexturesForSamePath )
{
    expectDiffuseTextureCreated();
    expectAlphaTextureCreated();
    expectSkyboxTextureCreated();

    const uint_t diffuse = m_cache->createDiffuseTextureFromFile( m_path );
    const uint_t alpha   = m_cache->createAlphaTextureFromFile( m_path );
    const uint_t skybox  = m_cache->createSkyboxTextureFromFile( m_path );
    const Stats  stats   = m_cache->getStatistics();

    EXPECT_EQ( m_diffuseTextureId, diffuse );
    EXPECT_EQ( m_alphaTextureId, alpha );
    EXPECT_EQ( m_skyboxTextureId, skybox );
    EXPECT_NE( diffuse, alpha );
    EXPECT_NE( diffuse, skybox );
    EXPECT_EQ( 1, stats.numDiffuseTexturesCreated );
    EXPECT_EQ( 1, stats.numAlphaTexturesCreated );
    EXPECT_EQ( 1, stats.numSkyboxTexturesCreated );
}
