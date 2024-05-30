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

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <vector_types.h>

#include <set>
#include <vector>

namespace imageSource {

/// A brush for the canvas image
class CanvasBrush
{
  public:
    void set( int width, int height, float4 color );

    int                 m_width;
    int                 m_height;
    float4              m_color;
    std::vector<float4> m_pixels;
};

/// An image for texture painting
class CanvasImage : public ImageSourceBase
{
  public:
    /// Create a canvas image with the specified dimensions.
    CanvasImage( unsigned int width, unsigned int height );

    /// The destructor is virtual.
    ~CanvasImage() override {}

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
    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel.  Returns true for success.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& /*dest*/ ) override { return false; }

    void clearImage( float4 color ) { std::fill( m_pixels.begin(), m_pixels.end(), color ); }
    void drawBrush( CanvasBrush& brush, int xcenter, int ycenter );
    void drawStroke( CanvasBrush& brush, int x0, int y0, int x1, int y1 );
    void setDirtyTilesRegion( int x0, int y0, int x1, int y1 );
    void clearDirtyTiles() { m_dirtyTiles.clear(); }
    std::set<int>& getDirtyTiles() { return m_dirtyTiles; }

    static int packTileId( int x, int y, int mipLevel ) { return (mipLevel << 24) + ( y << 12 ) + x; }
    static int3 unpackTileId( int id ) { return int3{id & 0xfff, (id >> 12) & 0xfff, id >> 24}; }

  private:
    TextureInfo         m_info;
    std::vector<float4> m_pixels;

    const int m_tileWidth = 64;  // float4 texture tiles are 64x64
    const int m_tileHeight = 64; 

    std::vector<float4> m_brushPixels;

    std::set<int> m_dirtyTiles;

    float4* getPixel( unsigned int x, unsigned int y )
    {
        return &m_pixels[y * m_info.width + x];
    }

    static float4 blendColor( float4& a, float4& b ) 
    {
        const float w = b.w;
        return float4{a.x * ( 1.0f - w ) + b.x * w, a.y * ( 1.0f - w ) + b.y * w, a.z * ( 1.0f - w ) + b.z * w, 0.0f};
    }

    static int clamp( int x, int a, int b ) { return std::min( std::max( x, a ), b ); }
};

}  // namespace imageSource
