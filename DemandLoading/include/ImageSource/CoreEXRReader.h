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

#include <openexr.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace imageSource {

/// OpenEXR Core image reader.
class CoreEXRReader : public MipTailImageSource
{
  public:
    /// The constructor copies the given filename.  The file is not opened until open() is called.
    explicit CoreEXRReader( const char* filename, bool readBaseColor = true )
        : m_filename( filename )
        , m_readBaseColor( readBaseColor )
    {
    }

    /// Destructor
    ~CoreEXRReader() override { close(); }

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
    void readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream = 0 ) override;

    /// Read the base color of the image (1x1 mip level) as an array of floats. Returns true on success.
    virtual bool readBaseColor( float4& dest ) override;

    /// Get tile width (used only for testing).
    unsigned int getTileWidth() const { return m_tileWidth; }

    /// Get tile height (used only for testing).
    unsigned int getTileHeight() const { return m_tileHeight; }

    /// Returns the number of tiles that have been read.
    unsigned long long getNumTilesRead() const override { return m_numTilesRead; };

    /// Returns the number of bytes that have been read.
    unsigned long long getNumBytesRead() const override { return m_numBytesRead; };

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
    std::mutex         m_mutex;
    unsigned long long m_numTilesRead  = 0;
    unsigned long long m_numBytesRead  = 0;
    double             m_totalReadTime = 0.0;

    int m_tileWidths[20]   = { 0 };
    int m_tileHeights[20]  = { 0 };
    int m_levelWidths[20]  = { 0 };
    int m_levelHeights[20] = { 0 };

    exr_pixel_type_t      m_pixelType = EXR_PIXEL_LAST_TYPE;
    exr_tile_round_mode_t m_roundMode;
    exr_tile_level_mode_t m_levelMode;

    // We are only supporting one-part files for now
    static constexpr int m_partIndex = 0;

    void readActualTile( char* dest, int rowPitch, int mipLevel, int tileX, int tileY );
    void readScanlineData( char* dest );
};

}  // namespace demandLoading
