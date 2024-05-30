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

#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/ImageSource/WrappedImageSource.h>

#include <mutex>
#include <utility>
#include <vector>

namespace demandPbrtScene {

/// Adapt a base image to a single channel uint8_t texture with values
/// zero or 255.  Any non-zero RGB value in the base image results in
/// the value of 255.  (Any alpha channel, if present, is ignored.)
class PbrtAlphaMapImageSource : public imageSource::WrappedImageSource
{
  public:
    PbrtAlphaMapImageSource( std::shared_ptr<ImageSource> baseImage );

    void open( imageSource::TextureInfo* info ) override;

    bool readTile( char* buffer, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream ) override;

    bool readMipLevel( char* buffer, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override;

    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      unsigned int pixelSizeInBytes,
                      CUstream     stream ) override;

  private:
    void getBaseInfo();
    void convertBasePixels( char* buffer, unsigned int width, unsigned int height );

    std::mutex               m_dataMutex;
    imageSource::TextureInfo m_baseInfo{};
    unsigned int             m_basePixelStride{};
    imageSource::TextureInfo m_alphaInfo{};
    std::vector<char>        m_basePixels;
};

}  // namespace demandPbrtScene
