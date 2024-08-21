// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/DemandTextureCache.h"

#include "DemandPbrtScene/ImageSourceFactory.h"

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <map>
#include <memory>

namespace demandPbrtScene {

namespace {

using uint_t = unsigned int;
using ImageSourcePtr = std::shared_ptr<imageSource::ImageSource>;

demandLoading::TextureDescriptor textureDescription()
{
    demandLoading::TextureDescriptor textureDesc{};
    textureDesc.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
    textureDesc.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
    textureDesc.filterMode       = CU_TR_FILTER_MODE_POINT;
    textureDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    textureDesc.maxAnisotropy    = 1;
    return textureDesc;
}

class DemandTextureCacheImpl : public DemandTextureCache
{
  public:
    DemandTextureCacheImpl( DemandLoaderPtr demandLoader, ImageSourceFactoryPtr imageSourceFactory )
        : m_demandLoader( std::move( demandLoader ) )
        , m_imageSourceFactory( std::move( imageSourceFactory ) )
    {
    }

    uint_t createDiffuseTextureFromFile( const std::string& path ) override
    {
        auto it = m_diffuseCache.find(path);
        if (it != m_diffuseCache.end())
        {
            return it->second;
        }
        const ImageSourcePtr imageSource = m_imageSourceFactory->createDiffuseImageFromFile( path );
        const demandLoading::DemandTexture& texture = m_demandLoader->createTexture( imageSource, textureDescription() );
        ++m_stats.numDiffuseTexturesCreated;
        const uint_t id = texture.getId();
        m_diffuseCache[path] = id;
        return id;
    }
    bool hasDiffuseTextureForFile( const std::string& path ) const override
    {
        return m_diffuseCache.find( path ) != m_diffuseCache.end();
    }

    uint_t createAlphaTextureFromFile( const std::string& path ) override
    {
        auto it = m_alphaCache.find(path);
        if (it != m_alphaCache.end())
        {
            return it->second;
        }
        const ImageSourcePtr imageSource = m_imageSourceFactory->createAlphaImageFromFile( path );
        const demandLoading::DemandTexture& texture = m_demandLoader->createTexture( imageSource, textureDescription() );
        ++m_stats.numAlphaTexturesCreated;
        const uint_t id = texture.getId();
        m_alphaCache[path] = id;
        return id;
    }
    bool hasAlphaTextureForFile( const std::string& path ) const override
    {
        return m_alphaCache.find( path ) != m_alphaCache.end();
    }

    uint_t createSkyboxTextureFromFile( const std::string& path ) override
    {
        auto it = m_skyboxCache.find(path);
        if (it != m_skyboxCache.end())
        {
            return it->second;
        }
        const ImageSourcePtr imageSource = m_imageSourceFactory->createSkyboxImageFromFile( path );
        const demandLoading::DemandTexture& texture = m_demandLoader->createTexture( imageSource, textureDescription() );
        ++m_stats.numSkyboxTexturesCreated;
        const uint_t id = texture.getId();
        m_skyboxCache[path] = id;
        return id;
    }
    bool hasSkyboxTextureForFile( const std::string& path ) const override
    {
        return m_skyboxCache.find( path ) != m_skyboxCache.end();
    }

    DemandTextureCacheStatistics getStatistics() const override { return m_stats; }

private:
    DemandTextureCacheStatistics  m_stats{};
    DemandLoaderPtr               m_demandLoader;
    ImageSourceFactoryPtr         m_imageSourceFactory;
    std::map<std::string, uint_t> m_diffuseCache;
    std::map<std::string, uint_t> m_alphaCache;
    std::map<std::string, uint_t> m_skyboxCache;
};


}  // namespace

DemandTextureCachePtr createDemandTextureCache( DemandLoaderPtr demandLoader, ImageSourceFactoryPtr imageSourceFactory )
{
    return std::make_shared<DemandTextureCacheImpl>( std::move( demandLoader ), std::move( imageSourceFactory ) );
}

}  // namespace demandPbrtScene
