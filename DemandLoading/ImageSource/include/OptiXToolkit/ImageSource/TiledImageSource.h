// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/ImageSource/WrappedImageSource.h>

#include <vector_types.h>

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

    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      CUstream     stream ) override;

    unsigned long long getNumTilesRead() const override;

  private:
    void getBaseInfo();

    mutable std::mutex m_dataMutex;
    bool               m_baseIsTiled{};
    TextureInfo        m_tiledInfo{};
    unsigned long long m_numTilesRead{};
    std::vector<char>  m_buffer;
    std::vector<char*> m_mipLevels;
    std::vector<uint2> m_mipDimensions;
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
