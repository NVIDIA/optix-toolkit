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

#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
#include <ImageSource/EXRReader.h>
#endif

#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>
#include <vector>

#include <sstream>
#include <stdexcept>

#ifndef ubyte
typedef unsigned char ubyte;
#endif

#ifndef ubyte4
struct ubyte4
{
    ubyte x, y, z, w;
};
#endif
#ifndef ubyte2 
struct ubyte2 
{
    ubyte x, y;
};
#endif

#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
struct half4
{
    half x, y, z, w;
};

struct half2
{
    half x, y;
};
#endif

namespace imageSource {

// clang-format off
unsigned int getNumChannels( float4& x ) { return 4; }
unsigned int getNumChannels( float2& x ) { return 2; }
unsigned int getNumChannels( float&  x ) { return 1; }
unsigned int getNumChannels( ubyte4& x ) { return 4; }
unsigned int getNumChannels( ubyte2& x ) { return 2; }
unsigned int getNumChannels( ubyte&  x ) { return 1; }
unsigned int getNumChannels( unsigned int& x ) { return 1; }

CUarray_format_enum getFormat( float4& x ) { return CU_AD_FORMAT_FLOAT; }
CUarray_format_enum getFormat( float2& x ) { return CU_AD_FORMAT_FLOAT; }
CUarray_format_enum getFormat( float&  x ) { return CU_AD_FORMAT_FLOAT; }
CUarray_format_enum getFormat( ubyte4& x ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
CUarray_format_enum getFormat( ubyte2& x ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
CUarray_format_enum getFormat( ubyte&  x ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
CUarray_format_enum getFormat( unsigned int& x ) { return CU_AD_FORMAT_UNSIGNED_INT32; }

void convertType( float4 a, float4& b ) { b = a; }
void convertType( float4 a, float2& b ) { b = {a.x, (a.y+a.z)}; }
void convertType( float4 a, float& b  ) { b = (a.x + a.y + a.z) / 3.0f; }
void convertType( float4 a, ubyte4& b ) { b = {ubyte(a.x*255.0f), ubyte(a.y*255.0f), ubyte(a.z*255.0f), ubyte(a.w*255.0f)}; }
void convertType( float4 a, ubyte2& b ) { b = {ubyte(a.x*255.0f), ubyte(a.y*255.0f)}; }
void convertType( float4 a, ubyte& b ) { b = ubyte(255.0f * (a.x + a.y + a.z) / 3.0f); }
void convertType( float4 a, unsigned int& b ) { b = (a.x+a.y+a.z > 0.1f) ? (1<<30) : 0; }

#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
unsigned int getNumChannels( half4& x ) { return 4; }
unsigned int getNumChannels( half2& x ) { return 2; }
unsigned int getNumChannels( half&  x ) { return 1; }
CUarray_format_enum getFormat( half4& x ) { return CU_AD_FORMAT_HALF; }
CUarray_format_enum getFormat( half2& x ) { return CU_AD_FORMAT_HALF; }
CUarray_format_enum getFormat( half&  x ) { return CU_AD_FORMAT_HALF; }
void convertType( float4 a, half4& b ) { b = {half(a.x), half(a.y), half(a.z), half(a.w)}; }
void convertType( float4 a, half2& b ) { b = {half(a.x), half(a.y)}; }
void convertType( float4 a, half& b ) { b = half((a.x + a.y + a.z) / 3.0f); }
#endif
// clang-format on

/// This image generates a procedural pattern in many different formats.
template <class TYPE>
class MultiCheckerImage : public MipTailImageSource
{
  public:
    /// Create a test image with the specified dimensions.
    MultiCheckerImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps = true );

    /// The destructor is virtual.
    ~MultiCheckerImage() override {}

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
    void readTile( char*        dest,
                   unsigned int mipLevel,
                   unsigned int tileX,
                   unsigned int tileY,
                   unsigned int tileWidth,
                   unsigned int tileHeight,
                   CUstream     stream = 0 ) override;

    /// Read the specified mipLevel.  Returns true for success.
    void readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream = 0 ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override;

  private:
    bool isOddChecker( float x, float y, unsigned int squaresPerSide );

    unsigned int      m_squaresPerSide;
    TextureInfo       m_info;
    std::vector<TYPE> m_mipLevelColors;
};


template <class TYPE>
MultiCheckerImage<TYPE>::MultiCheckerImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps )
    : m_squaresPerSide( squaresPerSide )
{
    TYPE c;
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = getFormat( c );
    m_info.numChannels  = getNumChannels( c );
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
        convertType( color, c );
        m_mipLevelColors.push_back( c );
    }
}

template <class TYPE>
void MultiCheckerImage<TYPE>::open( TextureInfo* info )
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
void MultiCheckerImage<TYPE>::readTile( char*        dest,
                                        unsigned int mipLevel,
                                        unsigned int tileX,
                                        unsigned int tileY,
                                        unsigned int tileWidth,
                                        unsigned int tileHeight,
                                        CUstream     stream )
{
    if( mipLevel >= m_info.numMipLevels )                                                                                               
    {                                                                           
        std::stringstream ss;
        ss << "Attempt to read from non-existent mip-level." << ": " << __FILE__ << " (" << __LINE__ << "): mipLevel >= m_info.numMipLevels";
        throw std::runtime_error( ss.str().c_str() );
    }

    TYPE black;
    convertType( float4{0.0f, 0.0f, 0.0f, 0.0f}, black );
    const TYPE color = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];

    unsigned int levelWidth     = std::max( 1u, m_info.width >> mipLevel );
    unsigned int levelHeight    = std::max( 1u, m_info.height >> mipLevel );
    unsigned int squaresPerSide = std::min( levelWidth, m_squaresPerSide );

    const unsigned int startX   = tileX * tileWidth;
    const unsigned int startY   = tileY * tileHeight;
    const unsigned int rowPitch = tileWidth * m_info.numChannels * getBytesPerChannel( m_info.format );

    for( unsigned int destY = 0; destY < tileHeight; ++destY )
    {
        TYPE* row = reinterpret_cast<TYPE*>( dest + destY * rowPitch );
        for( unsigned int destX = 0; destX < tileWidth; ++destX )
        {
            float tx   = static_cast<float>( destX + startX ) / static_cast<float>( levelWidth );
            float ty   = static_cast<float>( destY + startY ) / static_cast<float>( levelHeight );
            bool  odd  = isOddChecker( tx, ty, squaresPerSide );
            row[destX] = odd ? black : color;
        }
    }
}

template <class TYPE>
void MultiCheckerImage<TYPE>::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream )
{
    if( mipLevel >= m_info.numMipLevels )
    {
        std::stringstream ss;
        ss << "Attempt to read from non-existent mip-level." << ": " << __FILE__ << " (" << __LINE__ << "): mipLevel >= m_info.numMipLevels";
        throw std::runtime_error(ss.str().c_str());
    }

    TYPE black;
    convertType( float4{0.0f, 0.0f, 0.0f, 0.0f}, black );
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
}

template <class TYPE>
bool MultiCheckerImage<TYPE>::readBaseColor( float4& dest )
{
    dest = float4{1.0f, 1.0f, 0.0f, 0.0f};
    return ( m_info.numMipLevels > 1 ) || ( m_info.width == 1 && m_info.height == 1 );
}


}  // namespace imageSource
