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

/// \file EXRReader.h
/// OpenEXR image reader.

#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>

#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfTiledInputFile.h>

#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace imageSource {

/// OpenEXR image reader.
class EXRReader : public MipTailImageSource
{
  public:
    /// The constructor copies the given filename.  The file is not opened until open() is called.
    explicit EXRReader( const char* filename, bool readBaseColor = true )
        : m_filename( filename )
        , m_readBaseColor( readBaseColor )
    {
    }

    /// Destructor
    ~EXRReader() override { close(); }

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

    /// Read the specified mipLevel.  Throws an exception on error.
    void readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream = 0 ) override;

    /// Read the base color of the image (1x1 mip level) as an array of floats. Returns true on success.
    virtual bool readBaseColor( float4& dest ) override;

    /// Get tile width (used only for testing).
    unsigned int getTileWidth() const { return m_tileWidth; }

    /// Get tile height (used only for testing).
    unsigned int getTileHeight() const { return m_tileHeight; }

    /// Returns the number of tiles that have been read.
    unsigned long long getNumTilesRead() const override
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_numTilesRead;
    };

    /// Returns the number of bytes that have been read.
    unsigned long long getNumBytesRead() const override
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_numBytesRead;
    };

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
    std::string                          m_firstChannelName = { "R" };
    std::string                          m_filename;
    std::unique_ptr<Imf::TiledInputFile> m_tiledInputFile;
    std::unique_ptr<Imf::InputFile>      m_inputFile;

    TextureInfo                          m_info{};
    Imf::PixelType                       m_pixelType = Imf::NUM_PIXELTYPES;
    unsigned int                         m_tileWidth{};
    unsigned int                         m_tileHeight{};
    float4                               m_baseColor{};
    bool                                 m_readBaseColor = false;
    bool                                 m_baseColorWasRead = false;
    mutable std::mutex                   m_mutex;
    unsigned long long                   m_numTilesRead  = 0;
    unsigned long long                   m_numBytesRead  = 0;
    double                               m_totalReadTime = 0.0;

    void setupFrameBuffer( Imf::FrameBuffer& frameBuffer, char* base, size_t xStride, size_t yStride );
    void readActualTile( char* dest, unsigned int rowPitch, unsigned int mipLevel, unsigned int tileX, unsigned int tileY );
    void readScanlineData( char* dest );
};

}  // namespace imageSource
