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
                      unsigned int pixelSizeInBytes,
                      CUstream     stream ) override
    {
        return m_imageSource->readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, pixelSizeInBytes, stream );
    }

    /// Delegates to the wrapped ImageSource.
    bool readBaseColor( float4& dest ) override { return m_imageSource->readBaseColor( dest ); }

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
