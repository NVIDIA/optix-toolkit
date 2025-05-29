// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSources/DeviceMandelbrotImage.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

namespace imageSources {

void launchReadMandelbrotImage( const MandelbrotParams& params, CUstream stream );

DeviceMandelbrotImage::DeviceMandelbrotImage( unsigned int width, unsigned int height, double xmin, double ymin, double xmax, double ymax )
{
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = CU_AD_FORMAT_FLOAT;
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
    m_params.output_buffer  = nullptr;

    // Set default colors and max iterations
    std::vector<float4> colors = {
        {1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.5f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 0.0f}
    };
    setColors( colors, 512 );
}

void DeviceMandelbrotImage::setColors( const std::vector<float4>& colors, int maxIterations )
{
    m_params.max_iterations = maxIterations;
    m_params.num_colors     = std::min( static_cast<int>( colors.size() ), MAX_MANDELBROT_COLORS );
    for( int i = 0; i < m_params.num_colors; ++i )
    {
        m_params.colors[i] = colors[i];
    }
}

void DeviceMandelbrotImage::open( imageSource::TextureInfo* info )
{
    if( info != nullptr )
        *info = m_info;
}

bool DeviceMandelbrotImage::readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream )
{
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = std::max( 1u, m_params.width >> mipLevel );
    const unsigned int levelHeight = std::max( 1u, m_params.height >> mipLevel );

    const imageSource::PixelPosition start = pixelPosition( tile );
    OTK_ASSERT_MSG( levelWidth > start.x && levelHeight > start.y, "Requesting tile outside image bounds." );

    MandelbrotParams params = m_params;

    params.width          = tile.width;
    params.height         = tile.height;
    params.clip_width     = std::min( levelWidth - start.x, tile.width );
    params.clip_height    = std::min( levelHeight - start.y, tile.height );
    params.all_mip_levels = false;

    params.xmin = m_params.xmin + ( m_params.xmax - m_params.xmin ) * start.x / levelWidth;
    params.ymin = m_params.ymin + ( m_params.ymax - m_params.ymin ) * start.y / levelHeight;
    params.xmax = m_params.xmin + ( m_params.xmax - m_params.xmin ) * ( start.x + params.clip_width ) / levelWidth;
    params.ymax = m_params.ymin + ( m_params.ymax - m_params.ymin ) * ( start.y + params.clip_height ) / levelHeight;

    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadMandelbrotImage( params, stream );

    return true;
}

bool DeviceMandelbrotImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream )
{
    (void)width;  // silence unused variable warning
    (void)height;
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = std::max( 1u, m_info.width >> mipLevel );
    const unsigned int levelHeight = std::max( 1u, m_info.height >> mipLevel );

    OTK_ASSERT_MSG( levelWidth == width && levelHeight == height, "Mismatch between parameter and calculated mip level size." );

    MandelbrotParams params = m_params;
    params.width            = levelWidth;
    params.height           = levelHeight;
    params.clip_width       = levelWidth;
    params.clip_height      = levelHeight;
    params.all_mip_levels   = false;
    params.output_buffer    = reinterpret_cast<float4*>( dest );

    launchReadMandelbrotImage( params, stream );

    return true;
}

bool DeviceMandelbrotImage::readMipTail( char* dest,
                                         unsigned int mipTailFirstLevel,
                                         unsigned int /*numMipLevels*/,
                                         const uint2* mipLevelDims,
                                         CUstream stream ) 
{
    OTK_ASSERT_MSG( mipTailFirstLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

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

    return true;
}

bool DeviceMandelbrotImage::readBaseColor( float4& dest )
{
    double x = 0.5 * ( m_params.xmin + m_params.xmax );
    double y = 0.5 * ( m_params.ymin + m_params.ymax );
    dest     = mandelbrotColor( x, y, m_params );
    return true;
}

}  // namespace imageSources
