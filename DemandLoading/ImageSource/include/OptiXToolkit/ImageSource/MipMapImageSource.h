// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/ImageSource/WrappedImageSource.h>

#include <mutex>
#include <utility>
#include <vector>

namespace imageSource {

class MipMapImageSource : public WrappedImageSource
{
  public:
    MipMapImageSource( std::shared_ptr<ImageSource> baseImage );

    void open( TextureInfo* info ) override;

    void close() override;

    const TextureInfo& getInfo() const override;

    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override;

    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      CUstream     stream ) override;

    unsigned long long getNumTilesRead() const override;

  private:
    void getBaseInfo();

    // Must be called while the mutex is locked.
    const char* getMipLevelBuffer( unsigned int mipLevel, CUstream stream );

    mutable std::mutex m_dataMutex;
    unsigned int       m_numTilesRead{};
    TextureInfo        m_mipMapInfo{};
    bool               m_mipMappedBase{};
    unsigned int       m_pixelStrideInBytes{};
    std::vector<char>  m_buffer;
    std::vector<char*> m_mipLevels;
};

inline std::shared_ptr<ImageSource> createMipMapImageSource( std::shared_ptr<ImageSource> baseImage )
{
    if( !baseImage )
        return {};

    if( !baseImage->isOpen() )
        baseImage->open( nullptr );

    if( baseImage->getInfo().numMipLevels > 1 )
        return baseImage;

    return std::make_shared<MipMapImageSource>( std::move( baseImage ) );
}

}  // namespace imageSource
