// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <vector_types.h>

#include <vector>

namespace imageSource {

/// If OpenEXR is not available, this test image is used.  It generates a
/// procedural pattern, rather than loading image data from disk.
class CheckerBoardImage : public ImageSourceBase
{
  public:
    /// Create a test image with the specified dimensions.
    CheckerBoardImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps = true, bool tiled = true );

    /// The destructor is virtual.
    ~CheckerBoardImage() override = default;

    /// The open method simply initializes the given image info struct.
    void open( TextureInfo* info ) override;

    /// The close operation is a no-op.
    void close() override {}

    /// Check if image is currently open.
    bool isOpen() const override { return true; }

    /// Get the image info.  Valid only after calling open().
    const TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    /// Throws an exception on error.
    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel. Throws an exception on error.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& /*dest*/ ) override { return false; }

  private:
    bool isOddChecker( float x, float y, unsigned int squaresPerSide );

    unsigned int        m_squaresPerSide;
    TextureInfo         m_info;
    std::vector<float4> m_mipLevelColors;
};

}  // namespace imageSource
