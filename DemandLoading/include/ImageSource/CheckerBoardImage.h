//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>

#include <vector_types.h>

#include <vector>

namespace imageSource {

/// If OpenEXR is not available, this test image is used.  It generates a
/// procedural pattern, rather than loading image data from disk.
class CheckerBoardImage : public MipTailImageSource
{
  public:
    /// Create a test image with the specified dimensions.
    CheckerBoardImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps = true, bool tiled = true );

    /// The destructor is virtual.
    ~CheckerBoardImage() override {}

    /// The open method simply initializes the given image info struct.
    void open( TextureInfo* info ) override;

    /// The close operation is a no-op.
    void close() override {}

    /// Check if image is currently open.
    bool isOpen() const override { return true; }

    /// Get the image info.  Valid only after calling open().
    const TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    virtual CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    /// Throws an exception on error.
    void readTile( char*        dest,
                   unsigned int mipLevel,
                   unsigned int tileX,
                   unsigned int tileY,
                   unsigned int tileWidth,
                   unsigned int tileHeight,
                   CUstream     stream = 0 ) override;

    /// Read the specified mipLevel. Throws an exception on error.
    void readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream = 0 ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override { return false; }

  private:
    bool isOddChecker( float x, float y, unsigned int squaresPerSide );

    unsigned int        m_squaresPerSide;
    TextureInfo         m_info;
    std::vector<float4> m_mipLevelColors;
};

}  // namespace imageSource
