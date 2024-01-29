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

#include <OptiXToolkit/ImageSource/ImageHelpers.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace imageSources {

/// This image generates a procedural pattern in many different formats.
template <class TYPE>
class MultiCheckerImage : public imageSource::ImageSourceBase
{
  public:
    /// Create a test image with the specified dimensions.
    MultiCheckerImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps = true );

    /// The destructor is virtual.
    ~MultiCheckerImage() override {}

    /// The open method simply initializes the given image info struct.
    void open( imageSource::TextureInfo* info ) override;

    /// The close operation is a no-op.
    void close() override {}

    /// Check if image is currently open.
    bool isOpen() const override { return true; }

    /// Get the image info.  Valid only after calling open().
    const imageSource::TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    bool readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel.  Returns true for success.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override;

  private:
    bool isOddChecker( float x, float y, unsigned int squaresPerSide );

    unsigned int      m_squaresPerSide;
    imageSource::TextureInfo       m_info;
    std::vector<TYPE> m_mipLevelColors;
};


template <class TYPE>
MultiCheckerImage<TYPE>::MultiCheckerImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps )
    : m_squaresPerSide( squaresPerSide )
{
    TYPE c;
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = imageSource::getFormat( c );
    m_info.numChannels  = imageSource::getNumChannels( c );
    m_info.numMipLevels = useMipmaps ? imageSource::calculateNumMipLevels( width, height ) : 1;
    m_info.isValid      = true;
    m_info.isTiled      = true;

    // Use a different color per miplevel.
    std::vector<float4> colors{
        {1.0f, 0.0, 0.0, 0},    // red
        {1.0f, 0.5f, 0.0, 0},   // orange
        {1.0f, 1.0f, 0.0, 0},   // yellow
        {0.0, 1.0f, 0.0, 0},    // green
        {0.0, 0.0, 1.0f, 0},    // blue
        {0.5f, 0.0, 0.0, 0},    // dark red
        {0.5f, 0.25f, 0.0, 0},  // dark orange
        {0.5f, 0.5f, 0.0, 0},   // dark yellow
        {0.0, 0.5f, 0.0, 0},    // dark green
        {0.0, 0.0, 0.5f, 0},    // dark blue
    };
    for( float4& color : colors )
    {
        imageSource::convertType( color, c );
        m_mipLevelColors.push_back( c );
    }
}

template <class TYPE>
void MultiCheckerImage<TYPE>::open( imageSource::TextureInfo* info )
{
    if( info != nullptr )
        *info = m_info;
}

template <class TYPE>
inline bool MultiCheckerImage<TYPE>::isOddChecker( float x, float y, unsigned int squaresPerSide )
{
    int cx = static_cast<int>( x * squaresPerSide );
    int cy = static_cast<int>( y * squaresPerSide );
    return ( ( cx + cy ) & 1 ) != 0;
}

template <class TYPE>
bool MultiCheckerImage<TYPE>::readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream /*stream*/ )
{
    if( mipLevel >= m_info.numMipLevels )                                                                                               
    {                                                                           
        std::stringstream ss;
        ss << "Attempt to read from non-existent mip-level." << ": " << __FILE__ << " (" << __LINE__ << "): mipLevel >= m_info.numMipLevels";
        throw std::runtime_error( ss.str().c_str() );
    }

    TYPE black;
    imageSource::convertType( float4{0.0f, 0.0f, 0.0f, 0.0f}, black );
    const TYPE color = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];

    unsigned int levelWidth     = std::max( 1u, m_info.width >> mipLevel );
    unsigned int levelHeight    = std::max( 1u, m_info.height >> mipLevel );
    unsigned int squaresPerSide = std::min( levelWidth, m_squaresPerSide );

    const imageSource::PixelPosition start = pixelPosition( tile );
    const unsigned int rowPitch = tile.width * m_info.numChannels * imageSource::getBytesPerChannel( m_info.format );

    for( unsigned int destY = 0; destY < tile.height; ++destY )
    {
        TYPE* row = reinterpret_cast<TYPE*>( dest + destY * rowPitch );
        for( unsigned int destX = 0; destX < tile.width; ++destX )
        {
            float tx   = static_cast<float>( destX + start.x ) / static_cast<float>( levelWidth );
            float ty   = static_cast<float>( destY + start.y ) / static_cast<float>( levelHeight );
            bool  odd  = isOddChecker( tx, ty, squaresPerSide );
            row[destX] = odd ? black : color;
        }
    }

    return true;
}

template <class TYPE>
bool MultiCheckerImage<TYPE>::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream /*stream*/ )
{
    if( mipLevel >= m_info.numMipLevels )
    {
        std::stringstream ss;
        ss << "Attempt to read from non-existent mip-level." << ": " << __FILE__ << " (" << __LINE__ << "): mipLevel >= m_info.numMipLevels";
        throw std::runtime_error(ss.str().c_str());
    }

    TYPE black;
    imageSource::convertType( float4{0.0f, 0.0f, 0.0f, 0.0f}, black );
    const TYPE color  = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];
    TYPE*      pixels = reinterpret_cast<TYPE*>( dest );

    unsigned int squaresPerSide = std::min( width, m_squaresPerSide );

    for( unsigned int y = 0; y < height; ++y )
    {
        TYPE* row = pixels + y * width;
        for( unsigned int x = 0; x < width; ++x )
        {
            float tx  = static_cast<float>( x ) / static_cast<float>( width );
            float ty  = static_cast<float>( y ) / static_cast<float>( height );
            bool  odd = isOddChecker( tx, ty, squaresPerSide );
            row[x]    = odd ? black : color;
        }
    }

    return true;
}

template <class TYPE>
bool MultiCheckerImage<TYPE>::readBaseColor( float4& dest )
{
    dest = float4{1.0f, 1.0f, 0.0f, 0.0f};
    return ( m_info.numMipLevels > 1 ) || ( m_info.width == 1 && m_info.height == 1 );
}

}  // namespace imageSources
