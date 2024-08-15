// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file EXRReader.h
/// OpenEXR image reader.

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations of OpenEXR classes.
namespace OTK_IMF_NAMESPACE
{
    class FrameBuffer;
    class InputFile;
    class TiledInputFile;
}

namespace imageSource {

/// OpenEXR image reader. Uses the OpenEXR 2.x tile reading API.
/// CoreEXRReader is preferred, since it allows concurrent tile reads in the same texture.
class EXRReader : public ImageSourceBase
{
  public:
    /// The constructor copies the given filename.  The file is not opened until open() is called.
    explicit EXRReader( const std::string& filename, bool readBaseColor = true );

    /// Destructor
    ~EXRReader() override;

    /// Open the image and read header info, including dimensions and format.  Throws an exception on error.
    void open( TextureInfo* info ) override;

    /// Close the image.
    void close() override;

    /// Check if image is currently open.
    bool isOpen() const override { return static_cast<bool>( m_inputFile ) || static_cast<bool>( m_tiledInputFile ); }

    /// Get the image info.  Valid only after calling open().
    /// The caller should check the isValid struct member to determine
    /// if it contains valid information or not.
    const TextureInfo& getInfo() const override { return m_info;  }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    /// Throws an exception on error.
    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel.  Throws an exception on error.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as an array of floats. Returns true on success.
    bool readBaseColor( float4& dest ) override;

    /// Get tile width (used only for testing).
    unsigned int getTileWidth() const override { return m_tileWidth; }

    /// Get tile height (used only for testing).
    unsigned int getTileHeight() const override { return m_tileHeight; }

    /// Returns the number of tiles that have been read.
    unsigned long long getNumTilesRead() const override
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_numTilesRead;
    }

    /// Returns the number of bytes that have been read.
    unsigned long long getNumBytesRead() const override
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_numBytesRead;
    }

    /// Returns the time in seconds spent reading image tiles.
    double getTotalReadTime() const override
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_totalReadTime;
    }

    /// Serialize the image filename (etc.) to the give stream.
    void serialize( std::ostream& stream ) const;

    /// Deserialize an EXRReader.  Called from ImageSource::deserialize.
    static std::shared_ptr<ImageSource> deserialize( std::istream& stream );

  private:
    std::string m_firstChannelName{"R"};
    std::string m_filename;

    std::unique_ptr<OTK_IMF_NAMESPACE::TiledInputFile> m_tiledInputFile;
    std::unique_ptr<OTK_IMF_NAMESPACE::InputFile>      m_inputFile;

    TextureInfo        m_info{};
    unsigned int       m_pixelType;
    unsigned int       m_tileWidth{};
    unsigned int       m_tileHeight{};
    float4             m_baseColor{};
    bool               m_readBaseColor    = false;
    bool               m_baseColorWasRead = false;
    mutable std::mutex m_mutex;
    unsigned long long m_numTilesRead  = 0;
    unsigned long long m_numBytesRead  = 0;
    double             m_totalReadTime = 0.0;

    void setupFrameBuffer( OTK_IMF_NAMESPACE::FrameBuffer& frameBuffer, char* base, size_t xStride, size_t yStride );
    void readActualTile( char* dest, unsigned int rowPitch, unsigned int mipLevel, unsigned int tileX, unsigned int tileY );
    void readScanlineData( char* dest );
};

}  // namespace imageSource
