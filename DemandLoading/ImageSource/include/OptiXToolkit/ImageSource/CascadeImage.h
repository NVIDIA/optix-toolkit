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
#include <OptiXToolkit/ImageSource/TextureInfo.h>

namespace imageSource {

/// An image adapter used for cascading image sizes
class CascadeImage : public ImageSourceBase
{
  public:
    /// Create a CascadeImage for the given backing image such that width and height
    /// are greater than minDim if possible.
    CascadeImage( std::shared_ptr<ImageSource> backingImage, unsigned int minDim );

    /// The destructor is virtual.
    ~CascadeImage() override = default;

    /// The open method simply initializes the given image info struct.
    void open( TextureInfo* info ) override;

    /// The close operation is a no-op.
    void close() override {}

    /// Check if image is currently open.
    bool isOpen() const override { return m_isOpen; }

    /// Get the image info.  Valid only after calling open().
    const TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return m_backingImage->getFillType(); }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    /// Throws an exception on error.
    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override
    {
        return m_backingImage ? m_backingImage->readTile( dest, mipLevel + m_backingMipLevel, tile, stream) : false;
    }

    /// Read the specified mipLevel. Throws an exception on error.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override
    {
        if( !m_backingImage )
            return false;
        return m_backingImage->readMipLevel( dest, mipLevel + m_backingMipLevel, expectedWidth, expectedHeight, stream );
    }

    /// Read the mip tail into a single buffer
    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      unsigned int pixelSizeInBytes,
                      CUstream     stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override
    {
        return m_backingImage ? m_backingImage->readBaseColor( dest ) : false;
    }

    // Get the backing image
    std::shared_ptr<ImageSource> getBackingImage() { return m_backingImage; }

    // Set the backing image
    void setBackingImage( std::shared_ptr<ImageSource> image ) { m_backingImage = image; }

    // Return whether the image has a cascade
    bool hasCascade() const override { return m_info.width < m_backingImage->getInfo().width; }

  private:
    std::shared_ptr<ImageSource> m_backingImage;
    unsigned int                 m_backingMipLevel;
    TextureInfo                  m_info;
    unsigned int                 m_minDim;
    bool                         m_isOpen;
};

}  // namespace imageSource
