// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <cmath>

#include <cstring>

#include "CanvasImage.h"
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>

namespace imageSource {

void CanvasBrush::set( int width, int height, float4 color )
{
    m_color  = color;
    m_width  = width;
    m_height = height;
    m_pixels.resize( m_width * m_height );

    float cx = m_width * 0.5f;
    float cy = m_height * 0.5f;

    float alpha = color.w;
    for( int y = 0; y < m_height; ++y )
    {
        for( int x = 0; x < m_width; ++x )
        {
            // Compute alpha based on distance from center of brush
            float d2 = ( ( x - cx ) / cx ) * ( ( x - cx ) / cx ) + ( ( y - cy ) / cy ) * ( ( y - cy ) / cy );
            color.w  = alpha * ( 1.0f - powf( std::min( d2, 1.0f ), 2.0f ) );
            m_pixels[y * m_width + x] = color;
        }
    }
}

CanvasImage::CanvasImage( unsigned int width, unsigned int height )
{
    float4 TYPE;
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = getFormat( TYPE );
    m_info.numChannels  = getNumChannels( TYPE );
    m_info.numMipLevels = 1;
    m_info.isValid      = true;
    m_info.isTiled      = true;

    m_pixels.resize( m_info.width * m_info.height );
    clearImage( float4{1.0f, 1.0f, 1.0f, 0.0f} );
}

void CanvasImage::open( TextureInfo* info )
{
    if( info != nullptr ) 
        *info = m_info;
}

bool CanvasImage::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream /*stream*/  )
{
    if( mipLevel >= m_info.numMipLevels )
    {
        std::stringstream ss;
        ss << "Attempt to read from non-existent mip-level."
           << ": " << __FILE__ << " (" << __LINE__ << "): mipLevel >= m_info.numMipLevels";
        throw std::runtime_error( ss.str().c_str() );
    }

    const unsigned int srcStartX  = tile.x * tile.width;
    const unsigned int srcStartY  = tile.y * tile.height;
    float4*            destPixels = reinterpret_cast<float4*>( dest );

    unsigned int levelWidth  = std::max( m_info.width >> mipLevel, 1u );
    unsigned int levelHeight = std::max( m_info.height >> mipLevel, 1u );
    unsigned int copyWidth   = std::min( tile.width, levelWidth - srcStartX );
    unsigned int copyHeight  = std::min( tile.height, levelHeight - srcStartY );

    for( unsigned int destY = 0; destY < copyHeight; ++destY )
    {
        float4* destRow = destPixels + tile.width * destY;
        float4* srcRow  = getPixel( srcStartX, srcStartY + destY );
        memcpy( destRow, srcRow, copyWidth * sizeof( float4 ) );
    }

    return true;
}

bool CanvasImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int /*width*/, unsigned int /*height*/, CUstream /*stream*/ )
{
    if( mipLevel >= m_info.numMipLevels )
    {
        std::stringstream ss;
        ss << "Attempt to read from non-existent mip-level."
           << ": " << __FILE__ << " (" << __LINE__ << "): mipLevel >= m_info.numMipLevels";
        throw std::runtime_error( ss.str().c_str() );
    }

    float4* destPixels = reinterpret_cast<float4*>( dest );
    float4* srcPixels  = getPixel( 0, 0 );

    unsigned int levelWidth  = std::max( m_info.width >> mipLevel, 1u );
    unsigned int levelHeight = std::max( m_info.height >> mipLevel, 1u );

    memcpy( destPixels, srcPixels, levelWidth * levelHeight * sizeof( float4 ) );
    return true;
}

void CanvasImage::drawBrush( CanvasBrush& brush, int xcenter, int ycenter )
{
    int xstart = xcenter - brush.m_width / 2;
    int ystart = ycenter - brush.m_height / 2;
    for( int y = 0; y < brush.m_height; ++y ) 
    {
        if( y+ystart < 0 || y+ystart >= static_cast<int>( m_info.height ) )
            continue;

        for( int x = 0; x < brush.m_width; ++x )
        {
            if( x+xstart < 0 || x+xstart >= static_cast<int>( m_info.width ) ) 
                continue;
            float4* pixel = getPixel( x+xstart, y+ystart );
            *pixel = blendColor( *pixel, brush.m_pixels[ y * brush.m_width + x ] );
        }
    }

    int tileX0 = clamp( xstart / m_tileWidth, 0, m_info.width / m_tileWidth );
    int tileY0 = clamp( ystart / m_tileHeight, 0, m_info.height / m_tileHeight );
    int tileX1 = clamp( ( xstart + brush.m_width ) / m_tileWidth, 0, ( m_info.width + m_tileWidth - 1 ) / m_tileWidth - 1 );
    int tileY1 = clamp( ( ystart + brush.m_height ) / m_tileHeight, 0, ( m_info.height + m_tileHeight - 1 ) / m_tileHeight - 1 );
    setDirtyTilesRegion( tileX0, tileY0, tileX1, tileY1 );
}

void CanvasImage::drawStroke( CanvasBrush& brush, int x0, int y0, int x1, int y1 )
{
    float d = sqrtf( static_cast<float>( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) ) );
    int numIterations = std::max( static_cast<int>( 10 * d / brush.m_width ), 1 );
    for( int i=0; i<numIterations; ++i )
    {
        drawBrush( brush, x0 + (x1-x0)*i/numIterations, y0 + (y1-y0)*i/numIterations );
    }
}

void CanvasImage::setDirtyTilesRegion( int x0, int y0, int x1, int y1 )
{
    for( int y = y0; y <= y1; ++y )
    {
        for( int x = x0; x <= x1; ++x )
        {
            m_dirtyTiles.insert( packTileId( x, y, 0 ) );
        }
    }
}

}  // namespace imageSource
