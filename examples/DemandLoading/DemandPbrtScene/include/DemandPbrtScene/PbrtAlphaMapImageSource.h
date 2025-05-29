// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
