// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandTextureCacheStatistics.h"

#include <memory>
#include <string>

namespace demandLoading {
class DemandLoader;
}

namespace demandPbrtScene {

using uint_t = unsigned int;

class ImageSourceFactory;

class DemandTextureCache
{
  public:
    virtual ~DemandTextureCache() = default;

    virtual uint_t createDiffuseTextureFromFile( const std::string& path )   = 0;
    virtual bool   hasDiffuseTextureForFile( const std::string& path ) const = 0;

    virtual uint_t createAlphaTextureFromFile( const std::string& path )   = 0;
    virtual bool   hasAlphaTextureForFile( const std::string& path ) const = 0;

    virtual uint_t createSkyboxTextureFromFile( const std::string& path )   = 0;
    virtual bool   hasSkyboxTextureForFile( const std::string& path ) const = 0;

    virtual DemandTextureCacheStatistics getStatistics() const = 0;
};

using DemandLoaderPtr       = std::shared_ptr<demandLoading::DemandLoader>;
using DemandTextureCachePtr = std::shared_ptr<DemandTextureCache>;
using ImageSourceFactoryPtr = std::shared_ptr<ImageSourceFactory>;

DemandTextureCachePtr createDemandTextureCache( DemandLoaderPtr demandLoader, ImageSourceFactoryPtr imageSourceFactory );

}  // namespace demandPbrtScene
