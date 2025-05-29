// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <vector_functions.h> // from CUDA toolkit

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

bool CheckerBoardImage::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream /*stream*/  )
{
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const float4 black = make_float4( 0.f, 0.f, 0.f, 0.f );
    const float4 color = m_mipLevelColors[static_cast<int>( mipLevel % m_mipLevelColors.size() )];

    unsigned int levelWidth     = std::max( 1u, m_info.width >> mipLevel );
    unsigned int levelHeight    = std::max( 1u, m_info.height >> mipLevel );
    unsigned int squaresPerSide = std::min( levelWidth, m_squaresPerSide );

    const PixelPosition start    = pixelPosition( tile );
    const unsigned int  rowPitch = ( tile.width * getBitsPerPixel( m_info ) ) / BITS_PER_BYTE;

    for( unsigned int destY = 0; destY < tile.height; ++destY )
    {
        float4* row = reinterpret_cast<float4*>( dest + destY * rowPitch );
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

bool CheckerBoardImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream /*stream*/ )
{
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

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

    return true;
}

}  // namespace imageSource
