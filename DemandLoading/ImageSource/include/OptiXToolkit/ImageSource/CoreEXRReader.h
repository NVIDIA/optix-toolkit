// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// OpenEXRCore forward declaration.
using exr_context_t = struct _priv_exr_context_t*;

namespace imageSource {

/// OpenEXR Core image reader. Uses OpenEXR 3.0. This is preferred because
/// it allows concurrent reading of tiles in the same EXR file.
class CoreEXRReader : public ImageSourceBase
{
  public:
    /// The constructor copies the given filename.  The file is not opened until open() is called.
    explicit CoreEXRReader( const std::string& filename, bool readBaseColor = true );

    /// Destructor
    ~CoreEXRReader() override;

    /// Open the image and read header info, including dimensions and format.  Throws an exception on error.
    void open( TextureInfo* info ) override;

    /// Close the image.
    void close() override;

    /// Check if image is currently open.
    bool isOpen() const override { return static_cast<bool>( m_exrCtx ); }

    /// Get the image info.  Valid only after calling open().
    /// The caller should check the isValid struct member to determine
    /// if it contains valid information or not.
    const TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    /// Throws an exception on error.
    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel. Throws an exception on error.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight,
                       CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as an array of floats. Returns true on success.
    bool readBaseColor( float4& dest ) override;

    /// Get tile width (used only for testing).
    unsigned int getTileWidth() const override { return m_tileWidth; }

    /// Get tile height (used only for testing).
    unsigned int getTileHeight() const override { return m_tileHeight; }

    /// Returns the number of tiles that have been read.
    unsigned long long getNumTilesRead() const override { return m_numTilesRead; }

    /// Returns the number of bytes that have been read.
    unsigned long long getNumBytesRead() const override { return m_numBytesRead; }

    /// Returns the time in seconds spent reading image tiles.
    double getTotalReadTime() const override { return m_totalReadTime; }

  private:
    std::string        m_filename;
    exr_context_t      m_exrCtx = nullptr;
    bool               m_isScanline = false;
    TextureInfo        m_info{};
    unsigned int       m_tileWidth{};
    unsigned int       m_tileHeight{};
    float4             m_baseColor{};
    bool               m_readBaseColor    = false;
    bool               m_baseColorWasRead = false;
    std::mutex         m_initMutex;
    std::mutex         m_statsMutex;
    unsigned long long m_numTilesRead  = 0;
    unsigned long long m_numBytesRead  = 0;
    double             m_totalReadTime = 0.0;

    int m_tileWidths[20]{};
    int m_tileHeights[20]{};
    int m_levelWidths[20]{};
    int m_levelHeights[20]{};

    unsigned int m_pixelType;
    unsigned int m_roundMode;
    unsigned int m_levelMode;

    // We are only supporting one-part files for now
    static constexpr int m_partIndex = 0;

    void readActualTile( char* dest, int rowPitch, int mipLevel, int tileX, int tileY );
    void readScanlineData( char* dest );
};

}  // namespace demandLoading
