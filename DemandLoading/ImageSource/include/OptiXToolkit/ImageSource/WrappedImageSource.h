// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <memory>
#include <utility>

namespace imageSource {

struct TextureInfo;

/// WrappedImageSource delegates each method call to a given ImageSource.
class WrappedImageSource : public ImageSource
{
  public:
    /// Wrap the given ImageSource, delegating method calls to it.
    explicit WrappedImageSource( std::shared_ptr<ImageSource> imageSource )
        : m_imageSource( std::move( imageSource ) )
    {
    }

    /// The destructor is virtual to ensure that instances of derived classes are properly destroyed.
    ~WrappedImageSource() override = default;

    /// Delegates to the wrapped ImageSource.
    void open( TextureInfo* info ) override { m_imageSource->open( info ); }

    /// Delegates to the wrapped ImageSource.
    void close() override { m_imageSource->close(); }

    /// Delegates to the wrapped ImageSource.
    bool isOpen() const override { return m_imageSource->isOpen(); }

    /// Delegates to the wrapped ImageSource.
    const TextureInfo& getInfo() const override { return m_imageSource->getInfo(); }

    /// Delegates to the wrapped ImageSource.
    CUmemorytype getFillType() const override { return m_imageSource->getFillType(); }

    /// Delegates to the wrapped ImageSource.
    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override
    {
        return m_imageSource->readTile( dest, mipLevel, tile, stream);
    }

    /// Delegates to the wrapped ImageSource.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override
    {
        return m_imageSource->readMipLevel( dest, mipLevel, expectedWidth, expectedHeight, stream );
    }

    /// Delegates to the wrapped ImageSource.
    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      CUstream     stream ) override
    {
        return m_imageSource->readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
    }

    /// Delegates to the wrapped ImageSource.
    bool readBaseColor( float4& dest ) override { return m_imageSource->readBaseColor( dest ); }

    /// Delegates to the wrapped ImageSource.
    unsigned int getTileWidth() const override { return m_imageSource->getTileWidth(); }

    /// Delegates to the wrapped ImageSource.
    unsigned int getTileHeight() const override { return m_imageSource->getTileHeight(); }

    /// Delegates to the wrapped ImageSource.
    unsigned long long getNumTilesRead() const override { return m_imageSource->getNumTilesRead(); }

    /// Delegates to the wrapped ImageSource.
    unsigned long long getNumBytesRead() const override { return m_imageSource->getNumBytesRead(); }

    /// Delegates to the wrapped ImageSource.
    double getTotalReadTime() const override { return m_imageSource->getTotalReadTime(); }

    /// Delegates to the wrapped ImageSource.
    bool hasCascade() const override { return m_imageSource->hasCascade(); }

  private:
    std::shared_ptr<ImageSource> m_imageSource;
};

}  // namespace imageSource
