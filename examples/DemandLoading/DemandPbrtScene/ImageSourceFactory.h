// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "ImageSourceFactoryStatistics.h"

#include <OptiXToolkit/ImageSource/ImageSourceCache.h>

#include <memory>
#include <string>

namespace imageSource {
class ImageSource;
}  // namespace imageSource

namespace demandPbrtScene {

struct Options;

class ImageSourceFactory
{
  public:
    virtual ~ImageSourceFactory() = default;

    virtual std::shared_ptr<imageSource::ImageSource> createDiffuseImageFromFile( const std::string& path ) = 0;

    virtual std::shared_ptr<imageSource::ImageSource> createAlphaImageFromFile( const std::string& path ) = 0;

    virtual std::shared_ptr<imageSource::ImageSource> createSkyboxImageFromFile( const std::string& path ) = 0;

    virtual ImageSourceFactoryStatistics getStatistics() const = 0;
};

std::shared_ptr<ImageSourceFactory> createImageSourceFactory( const Options& options );

}  // namespace demandPbrtScene
