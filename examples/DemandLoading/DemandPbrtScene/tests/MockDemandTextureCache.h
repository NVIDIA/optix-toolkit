#pragma once

#include <DemandTextureCache.h>

#include <gmock/gmock.h>

#include <memory>

namespace demandPbrtScene {

namespace testing {

class MockDemandTextureCache : public ::testing::StrictMock<DemandTextureCache>
{
  public:
    ~MockDemandTextureCache() override = default;

    MOCK_METHOD( uint_t, createDiffuseTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasDiffuseTextureForFile, ( const std::string& path ), ( const override ) );
    MOCK_METHOD( uint_t, createAlphaTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasAlphaTextureForFile, (const std::string&), ( const, override ) );
    MOCK_METHOD( uint_t, createSkyboxTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasSkyboxTextureForFile, (const std::string&), ( const, override ) );
    MOCK_METHOD( DemandTextureCacheStatistics, getStatistics, (), ( const override ) );
};

using MockDemandTextureCachePtr = std::shared_ptr<MockDemandTextureCache>;

inline MockDemandTextureCachePtr createMockDemandTextureCache()
{
    return std::make_shared<MockDemandTextureCache>();
}

}  // namespace testing
}  // namespace demandPbrtScene
