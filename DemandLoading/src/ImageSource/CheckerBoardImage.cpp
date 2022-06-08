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
#include <ImageSource/CheckerBoardImage.h>

#include "Exception.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>  // for make_float4

namespace imageSource {

CheckerBoardImage::CheckerBoardImage( unsigned int width, unsigned int height, unsigned int squaresPerSide, bool useMipmaps, bool tiled )
    : m_squaresPerSide( squaresPerSide )
{
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = CU_AD_FORMAT_FLOAT;
    m_info.numChannels  = 4;
    m_info.numMipLevels = useMipmaps ? imageSource::calculateNumMipLevels( width, height ) : 1;
    m_info.isValid      = true;
    m_info.isTiled      = tiled;

    // Use a different color per miplevel.
    std::vector<float4> colors{
        {255, 0, 0, 0},    // red
        {255, 127, 0, 0},  // orange
        {255, 255, 0, 0},  // yellow
        {0, 255, 0, 0},    // green
        {0, 0, 255, 0},    // blue
        {127, 0, 0, 0},    // dark red
        {127, 63, 0, 0},   // dark orange
        {127, 127, 0, 0},  // dark yellow
        {0, 127, 0, 0},    // dark green
        {0, 0, 127, 0},    // dark blue
    };
    // Normalize the miplevel colors to [0,1]
    for( float4& color : colors )
    {
        color.x /= 255.f;
        color.y /= 255.f;
        color.z /= 255.f;
    }
    m_mipLevelColors.swap( colors );
}

void CheckerBoardImage::open( TextureInfo* info )
{
    if( info != nullptr )
        *info = m_info;
}

inline bool CheckerBoardImage::isOddChecker( float x, float y, unsigned int squaresPerSide )
{
    int cx = static_cast<int>( x * squaresPerSide );
    int cy = static_cast<int>( y * squaresPerSide );
    return ( ( cx + cy ) & 1 ) != 0;
}

void CheckerBoardImage::readTile( char*        dest,
                                  unsigned int mipLevel,
                                  unsigned int tileX,
                                  unsigned int tileY,
                                  unsigned int tileWidth,
                                  unsigned int tileHeight,
                                  CUstream     stream )
{
    DEMAND_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const float4 black = make_float4( 0.f, 0.f, 0.f, 0.f );
    const float4 color = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];

    unsigned int levelWidth     = std::max( 1u, m_info.width >> mipLevel );
    unsigned int levelHeight    = std::max( 1u, m_info.height >> mipLevel );
    unsigned int squaresPerSide = std::min( levelWidth, m_squaresPerSide );

    const unsigned int startX   = tileX * tileWidth;
    const unsigned int startY   = tileY * tileHeight;
    const unsigned int rowPitch = tileWidth * m_info.numChannels * getBytesPerChannel( m_info.format );

    for( unsigned int destY = 0; destY < tileHeight; ++destY )
    {
        float4* row = reinterpret_cast<float4*>( dest + destY * rowPitch );
        for( unsigned int destX = 0; destX < tileWidth; ++destX )
        {
            float tx   = static_cast<float>( destX + startX ) / static_cast<float>( levelWidth );
            float ty   = static_cast<float>( destY + startY ) / static_cast<float>( levelHeight );
            bool  odd  = isOddChecker( tx, ty, squaresPerSide );
            row[destX] = odd ? black : color;
        }
    }
}

void CheckerBoardImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream )
{
    DEMAND_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const float4 black  = make_float4( 0.f, 0.f, 0.f, 0.f );
    const float4 color  = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];
    float4*      pixels = reinterpret_cast<float4*>( dest );

    unsigned int squaresPerSide = std::min( width, m_squaresPerSide );

    for( unsigned int y = 0; y < height; ++y )
    {
        float4* row = pixels + y * width;
        for( unsigned int x = 0; x < width; ++x )
        {
            float tx  = static_cast<float>( x ) / static_cast<float>( width );
            float ty  = static_cast<float>( y ) / static_cast<float>( height );
            bool  odd = isOddChecker( tx, ty, squaresPerSide );
            row[x]    = odd ? black : color;
        }
    }
}

}  // namespace imageSource
