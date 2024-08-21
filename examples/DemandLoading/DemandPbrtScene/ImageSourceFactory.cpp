// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/ImageSourceFactory.h"

#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/PbrtAlphaMapImageSource.h"

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/ImageSourceCache.h>
#include <OptiXToolkit/ImageSource/TiledImageSource.h>

#include <fstream>
#include <iostream>
#include <map>

namespace demandPbrtScene {

using ImageSourcePtr = std::shared_ptr<imageSource::ImageSource>;

namespace {
class ImageSourceFactoryImpl : public ImageSourceFactory
{
  public:
    ImageSourceFactoryImpl( const Options& options )
        : m_options( options )
    {
    }
    ~ImageSourceFactoryImpl() override = default;

    ImageSourcePtr createDiffuseImageFromFile( const std::string& path ) override;

    ImageSourcePtr createAlphaImageFromFile( const std::string& path ) override;

    ImageSourcePtr createSkyboxImageFromFile( const std::string& path ) override;

    ImageSourceFactoryStatistics getStatistics() const override;

private:
    const Options&                m_options;
    imageSource::ImageSourceCache m_fileCache;
    imageSource::ImageSourceCache m_diffuseCache;
    imageSource::ImageSourceCache m_alphaCache;
    imageSource::ImageSourceCache m_skyboxCache;
};

bool fileExists( const std::string& path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

std::string replaceExtension( const std::string& path, const std::string& extension )
{
    const size_t dot = path.find_last_of( '.' );
    return path.substr( 0, dot ) + extension;
}

std::shared_ptr<imageSource::ImageSource> ImageSourceFactoryImpl::createDiffuseImageFromFile( const std::string& path )
{
    // Prefer EXR file if available.
    std::string        exrPath( replaceExtension( path, ".exr" ) );
    const std::string& filePath = fileExists( exrPath ) ? exrPath : path;
    ImageSourcePtr     image    = m_diffuseCache.find( filePath );
    if( image )
    {
        return image;
    }

    if( m_options.verboseTextureCreation )
    {
        std::cout << "Creating diffuse map from " << filePath << '\n';
    }

    image = createTiledImageSource( m_fileCache.get( filePath ) );
    m_diffuseCache.set( path, image );
    return image;
}

std::shared_ptr<imageSource::ImageSource> ImageSourceFactoryImpl::createAlphaImageFromFile( const std::string& path )
{
    ImageSourcePtr image = m_alphaCache.find( path );
    if( image )
    {
        return image;
    }

    if( m_options.verboseTextureCreation )
    {
        std::cout << "Creating alpha map from " << path << '\n';
    }

    image = std::make_shared<PbrtAlphaMapImageSource>( createTiledImageSource( m_fileCache.get( path ) ) );
    m_alphaCache.set( path, image );
    return image;
}

ImageSourcePtr ImageSourceFactoryImpl::createSkyboxImageFromFile( const std::string& path )
{
    ImageSourcePtr image = m_skyboxCache.find( path );
    if( image )
    {
        return image;
    }

    if( m_options.verboseTextureCreation )
    {
        std::cout << "Creating skybox map from " << path << '\n';
    }

    image = createTiledImageSource( m_fileCache.get( path ) );
    m_skyboxCache.set( path, image );
    return image;
}

ImageSourceFactoryStatistics ImageSourceFactoryImpl::getStatistics() const
{
    ImageSourceFactoryStatistics result{};
    result.fileSources = m_fileCache.getStatistics();
    result.alphaSources = m_alphaCache.getStatistics();
    result.diffuseSources = m_diffuseCache.getStatistics();
    result.skyboxSources = m_skyboxCache.getStatistics();
    return result;
}

}  // namespace

std::shared_ptr<ImageSourceFactory> createImageSourceFactory( const Options& options )
{
    return std::make_shared<ImageSourceFactoryImpl>( options );
}

}  // namespace demandPbrtScene
