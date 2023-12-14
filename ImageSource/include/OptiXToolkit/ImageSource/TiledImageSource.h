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

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace imageSource {

class TiledImageSource : public WrappedImageSource
{
  public:
    explicit TiledImageSource( std::shared_ptr<ImageSource> baseImage );
    ~TiledImageSource() override = default;

    void open( TextureInfo* info ) override;

    void close() override;

    const TextureInfo& getInfo() const override;

    bool readTile( char*        dest,
                   unsigned int mipLevel,
                   unsigned int tileX,
                   unsigned int tileY,
                   unsigned int tileWidth,
                   unsigned int tileHeight,
                   CUstream     stream ) override;

    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      unsigned int pixelSizeInBytes,
                      CUstream     stream ) override;

    unsigned long long getNumTilesRead() const override;

  private:
    mutable std::mutex m_dataMutex;
    bool               m_baseIsTiled{};
    TextureInfo        m_tiledInfo{};
    unsigned long long m_numTilesRead{};
    std::vector<char>  m_buffer;
    std::vector<char*> m_mipLevels;
};

/// A simple convenience function to reliably get a tiled image source.
/// NOTE: This function performs an eager open on the underlying image
/// source if it isn't already open, which may not be desirable.
inline std::shared_ptr<ImageSource> createTiledImageSource( std::shared_ptr<ImageSource> baseImage )
{
    if( !baseImage )
        return {};

    if( !baseImage->isOpen() )
        baseImage->open( nullptr );

    if( baseImage->getInfo().isTiled )
        return baseImage;

    return std::make_shared<TiledImageSource>( std::move( baseImage ) );
}

}  // namespace imageSource
