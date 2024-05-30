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

#include "DemandTextureCache.h"

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>

#include "ImageSourceFactory.h"

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
