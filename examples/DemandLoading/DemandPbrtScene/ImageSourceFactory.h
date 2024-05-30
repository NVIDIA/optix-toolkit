//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
