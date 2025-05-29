// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <OptiXToolkit/ImageSources/DeviceMandelbrotParams.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace imageSources {

/// This image makes a basic mandelbrot set using a CUDA kernel on the GPU
class DeviceMandelbrotImage : public imageSource::ImageSourceBase
{
  public:
    /// Create a mandelbrot image with the specified dimensions.
    DeviceMandelbrotImage( unsigned int width, unsigned int height, double xmin, double ymin, double xmax, double ymax );

    /// The destructor is virtual.
    ~DeviceMandelbrotImage() override = default;

    void setColors( const std::vector<float4>& colors, int maxIterations );

    /// The open method initializes the given image info struct.
    void open( imageSource::TextureInfo* info ) override;

    /// The close operation.
    void close() override {}

    /// Check if image is currently open.
    bool isOpen() const override { return true; }

    /// Get the image info.  Valid only after calling open().
    const imageSource::TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_DEVICE; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    bool readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel.  Returns true for success.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream ) override;

    /// Read the mip tail into a single buffer
    bool readMipTail( char* dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override;

  private:
    imageSource::TextureInfo  m_info;
    MandelbrotParams m_params;
};

}  // namespace imageSources
