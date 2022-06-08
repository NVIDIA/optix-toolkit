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

#include <ImageSource/DeviceMandelbrotParams.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace imageSource {

/// This image makes a basic mandelbrot set using a CUDA kernel on the GPU
class DeviceMandelbrotImage : public MipTailImageSource
{
  public:
    /// Create a test image with the specified dimensions.
    DeviceMandelbrotImage( unsigned int               width,
                           unsigned int               height,
                           double                     xmin,
                           double                     ymin,
                           double                     xmax,
                           double                     ymax,
                           int                        maxIterations,
                           const std::vector<float4>& colors );

    /// The destructor is virtual.
    ~DeviceMandelbrotImage() override {}

    /// The open method initializes the given image info struct.
    void open( TextureInfo* info ) override;

    /// The close operation.
    void close() override {}

    /// Check if image is currently open.
    bool isOpen() const override { return true; }

    /// Get the image info.  Valid only after calling open().
    const TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_DEVICE; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    void readTile( char*        dest,
                   unsigned int mipLevel,
                   unsigned int tileX,
                   unsigned int tileY,
                   unsigned int tileWidth,
                   unsigned int tileHeight,
                   CUstream     stream ) override;

    /// Read the specified mipLevel.  Returns true for success.
    void readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream ) override;

    /// Read the mip tail into a single buffer
    void readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      unsigned int pixelSizeInBytes,
                      CUstream     stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override;

  private:
    TextureInfo  m_info;
    unsigned int m_max_iterations;
    MandelbrotParams m_params;
};

}  // namespace imageSource
