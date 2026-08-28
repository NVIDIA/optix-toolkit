// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/ImageSource/WrappedImageSource.h>

#include <memory>

namespace imageSource {

/// Converts sRGB image pixels to linear floating-point values.
///
/// The first three channels are treated as color channels.  For two-channel
/// images, only the first channel is treated as color; the second channel is
/// preserved as alpha.  A fourth channel is likewise preserved as alpha.
class InverseSrgbImageSource : public WrappedImageSource
{
  public:
    explicit InverseSrgbImageSource( std::shared_ptr<ImageSource> imageSource );

    void open( TextureInfo* info ) override;

    void close() override;

    const TextureInfo& getInfo() const override;

    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override;

    bool readMipTail( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, const uint2* mipLevelDims, CUstream stream ) override;

    bool readBaseColor( float4& dest ) override;

  private:
    void getBaseInfo();
    void convertPixels( const char* source, float* dest, size_t numPixels ) const;

    TextureInfo m_baseInfo{};
    TextureInfo m_info{};
};

std::shared_ptr<ImageSource> createInverseSrgbImageSource( std::shared_ptr<ImageSource> imageSource );

}  // namespace imageSource
