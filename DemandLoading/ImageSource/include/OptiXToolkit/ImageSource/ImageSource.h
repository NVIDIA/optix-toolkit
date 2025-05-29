// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file ImageSource.h
/// Interface for a mipmapped image.

#include <cuda.h>
#include <vector_types.h>

#include <cmath>
#include <memory>
#include <string>

namespace imageSource {

struct TextureInfo;

struct PixelPosition
{
    unsigned int x;
    unsigned int y;
};

struct Tile
{
    unsigned int x;
    unsigned int y;
    unsigned int width;
    unsigned int height;
};

inline PixelPosition pixelPosition( const Tile& tile )
{
    return { tile.x * tile.width, tile.y * tile.height };
}

/// Interface for a mipmapped image.
///
/// Any method may be called from multiple threads; the implementation must be threadsafe.
class ImageSource
{
  public:
    /// The destructor is virtual to ensure that instances of derived classes are properly destroyed.
    virtual ~ImageSource() = default;

    /// Open the image and read header info, including dimensions and format.  Throws an exception on error.
    virtual void open( TextureInfo* info ) = 0;

    /// Close the image.
    virtual void close() = 0;

    /// Check if image is currently open.
    virtual bool isOpen() const = 0;

    /// Get the image info.  Valid only after calling open().
    /// The caller should check the isValid struct member to determine
    /// if it contains valid information or not.
    virtual const TextureInfo& getInfo() const = 0;

    /// Return the mode in which the image fills part of itself
    virtual CUmemorytype getFillType() const = 0;

    /// Read the specified tile or mip level, returning the data in dest.
    /// dest must be large enough to hold the tile.  Pixels outside
    /// the bounds of the mip level will be filled in with black.
    /// Throws an exception on error.
    /// Returns true if the request was satisfied and data was copied into dest.
    virtual bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) = 0;

    /// Read the specified mipLevel. Throws an exception on error.
    /// Returns true if the request was satisfied and data was copied into dest.
    virtual bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) = 0;

    /// Read the mip tail into the given buffer, starting with the specified level.  An array
    /// containing the expected dimensions of all the miplevels is provided (starting from miplevel
    /// zero), along with the pixel size.
    /// Throws an exception on error.
    /// Returns true if the request was satisfied and data was copied into dest.
    virtual bool readMipTail( char*        dest,
                              unsigned int mipTailFirstLevel,
                              unsigned int numMipLevels,
                              const uint2* mipLevelDims,
                              CUstream     stream ) = 0;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    virtual bool readBaseColor( float4& dest ) = 0;

    /// Get tile width (used only for testing).
    virtual unsigned int getTileWidth() const = 0;

    /// Get tile height (used only for testing).
    virtual unsigned int getTileHeight() const = 0;

    /// Returns the number of tiles that have been read.
    virtual unsigned long long getNumTilesRead() const = 0;

    /// Returns the number of bytes that have been read.  This number may be zero if the reader does
    /// not load tiles from disk, e.g. for procedural textures.
    virtual unsigned long long getNumBytesRead() const = 0;

    /// Returns the time in seconds spent reading image data (tiles or mip levels).  This number may
    /// be zero if the reader does not load tiles from disk, e.g. for procedural textures.
    virtual double getTotalReadTime() const = 0;

    /// Return true if the image has a cascade (larger size) that could be switched to.
    virtual bool hasCascade() const = 0;

    /// Return a hash of the image, using a small mip level.
    unsigned long long getHash( CUstream stream );
};

/// Base class for ImageSource with default implementation of readMipTail, etc.
class ImageSourceBase : public ImageSource
{
  public:
    ~ImageSourceBase() override = default;

    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      CUstream     stream ) override;

    unsigned int getTileWidth() const override { return 0u; }

    unsigned int getTileHeight() const override { return 0u; }

    unsigned long long getNumTilesRead() const override { return 0u; }

    unsigned long long getNumBytesRead() const override { return 0u; }

    double getTotalReadTime() const override { return 0.0; }

    bool hasCascade() const override { return false; }
};

/// @private
inline unsigned int calculateNumMipLevels( unsigned int width, unsigned int height )
{
    unsigned int dim = ( width > height ) ? width : height;
    return 1 + static_cast<unsigned int>( std::log2f( static_cast<float>( dim ) ) );
}

std::shared_ptr<ImageSource> createImageSource( const std::string& filename, const std::string& directory = "" );

}  // namespace imageSource
