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

#include <ImageSource/DeviceMandelbrotImage.h>

#include "Exception.h"

namespace imageSource {

void launchReadMandelbrotImage( const MandelbrotParams& params, CUstream stream );

DeviceMandelbrotImage::DeviceMandelbrotImage( unsigned int               width,
                                              unsigned int               height,
                                              double                     xmin,
                                              double                     ymin,
                                              double                     xmax,
                                              double                     ymax,
                                              int                        maxIterations,
                                              const std::vector<float4>& colors )
{
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = CU_AD_FORMAT_FLOAT; // float4
    m_info.numChannels  = 4;
    m_info.numMipLevels = imageSource::calculateNumMipLevels( width, height );
    m_info.isValid      = true;
    m_info.isTiled      = true;

    m_params                = {};
    m_params.width          = width;
    m_params.height         = height;
    m_params.clip_width     = width;
    m_params.clip_height    = height;
    m_params.all_mip_levels = false;
    m_params.xmin           = xmin;
    m_params.ymin           = ymin;
    m_params.xmax           = xmax;
    m_params.ymax           = ymax;
    m_params.max_iterations = maxIterations;
    m_params.num_colors     = std::min( static_cast<int>( colors.size() ), MAX_MANDELBROT_COLORS );
    for( int i = 0; i < m_params.num_colors; ++i )
    {
        m_params.colors[i] = colors[i];
    }
    m_params.output_buffer = nullptr;
}

void DeviceMandelbrotImage::open( TextureInfo* info )
{
    if( info != nullptr )
        *info = m_info;
}

void DeviceMandelbrotImage::readTile( char*        dest,
                                      unsigned int mipLevel,
                                      unsigned int tileX,
                                      unsigned int tileY,
                                      unsigned int tileWidth,
                                      unsigned int tileHeight,
                                      CUstream     stream )
{
    DEMAND_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = std::max( 1u, m_params.width >> mipLevel );
    const unsigned int levelHeight = std::max( 1u, m_params.height >> mipLevel );

    DEMAND_ASSERT_MSG( levelWidth > tileX * tileWidth && levelHeight > tileY * tileHeight,
                       "Requesting tile outside image bounds." );

    MandelbrotParams params = m_params;

    params.width          = tileWidth;
    params.height         = tileHeight;
    params.clip_width     = std::min( levelWidth - ( tileX * tileWidth ), tileWidth );
    params.clip_height    = std::min( levelHeight - ( tileY * tileHeight ), tileHeight );
    params.all_mip_levels = false;

    params.xmin = m_params.xmin + ( m_params.xmax - m_params.xmin ) * ( tileX * tileWidth ) / levelWidth;
    params.ymin = m_params.ymin + ( m_params.ymax - m_params.ymin ) * ( tileY * tileHeight ) / levelHeight;
    params.xmax = m_params.xmin + ( m_params.xmax - m_params.xmin ) * ( tileX * tileWidth + params.clip_width ) / levelWidth;
    params.ymax = m_params.ymin + ( m_params.ymax - m_params.ymin ) * ( tileY * tileHeight + params.clip_height ) / levelHeight;

    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadMandelbrotImage( params, stream );
}

void DeviceMandelbrotImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream )
{
    DEMAND_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = std::max( 1u, m_info.width >> mipLevel );
    const unsigned int levelHeight = std::max( 1u, m_info.height >> mipLevel );

    DEMAND_ASSERT_MSG( levelWidth == width && levelHeight == height, "Mismatch between parameter and calculated mip level size." );

    MandelbrotParams params = m_params;
    params.width            = levelWidth;
    params.height           = levelHeight;
    params.clip_width       = levelWidth;
    params.clip_height      = levelHeight;
    params.all_mip_levels   = false;
    params.output_buffer    = reinterpret_cast<float4*>( dest );

    launchReadMandelbrotImage( params, stream );
}

void DeviceMandelbrotImage::readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      unsigned int pixelSizeInBytes,
                      CUstream     stream ) 
{
    DEMAND_ASSERT_MSG( mipTailFirstLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = mipLevelDims[mipTailFirstLevel].x;
    const unsigned int levelHeight = mipLevelDims[mipTailFirstLevel].y;

    MandelbrotParams params = m_params;
    params.width            = levelWidth;
    params.height           = levelHeight;
    params.clip_width       = levelWidth;
    params.clip_height      = levelHeight;
    params.all_mip_levels   = true;
    params.output_buffer    = reinterpret_cast<float4*>( dest );

    launchReadMandelbrotImage( params, stream );
}

bool DeviceMandelbrotImage::readBaseColor( float4& dest )
{
    double x = 0.5 * ( m_params.xmin + m_params.xmax );
    double y = 0.5 * ( m_params.ymin + m_params.ymax );
    dest     = mandelbrotColor( x, y, m_params );
    return true;
}

}  // namespace imageSource
