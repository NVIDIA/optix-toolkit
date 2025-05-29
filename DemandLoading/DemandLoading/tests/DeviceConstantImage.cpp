// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DeviceConstantImage.h"

#include <OptiXToolkit/Error/ErrorCheck.h>

using namespace imageSource;

void launchReadConstantImage( const DeviceConstantImageParams& params, CUstream stream );

DeviceConstantImage::DeviceConstantImage( unsigned int width, unsigned int height, const std::vector<float4>& colors )
{
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = CU_AD_FORMAT_FLOAT; // float4
    m_info.numChannels  = 4;
    m_info.numMipLevels = imageSource::calculateNumMipLevels( width, height );
    m_info.isValid      = true;
    m_info.isTiled      = true;

    m_mipColors = colors;
}

void DeviceConstantImage::open( TextureInfo* info )
{
    if( info != nullptr )
        *info = m_info;
}

bool DeviceConstantImage::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream )
{
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    DeviceConstantImageParams params;
    params.num_pixels = tile.width * tile.height;
    params.color = m_mipColors[ mipLevel % m_mipColors.size() ];
    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadConstantImage( params, stream );

    return true;
}

bool DeviceConstantImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream )
{
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = std::max( 1u, m_info.width >> mipLevel );
    const unsigned int levelHeight = std::max( 1u, m_info.height >> mipLevel );

    OTK_ASSERT_MSG( levelWidth == width && levelHeight == height,
                    "Mismatch between parameter and calculated mip level size." );
    (void)width;  // silence unused variable warning
    (void)height;
    
    DeviceConstantImageParams params{};
    params.num_pixels    = levelWidth * levelHeight;
    params.color         = m_mipColors[mipLevel % m_mipColors.size()];
    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadConstantImage( params, stream );

    return true;
}

bool DeviceConstantImage::readMipTail( char* dest,
                                       unsigned int mipTailFirstLevel,
                                       unsigned int /*numMipLevels*/,
                                       const uint2* mipLevelDims,
                                       CUstream stream )
{
    OTK_ASSERT_MSG( mipTailFirstLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = mipLevelDims[mipTailFirstLevel].x;
    const unsigned int levelHeight = mipLevelDims[mipTailFirstLevel].y;

    DeviceConstantImageParams params{};
    params.num_pixels = levelWidth * levelHeight * 4 / 3;
    params.color = m_mipColors[ mipTailFirstLevel % m_mipColors.size() ];
    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadConstantImage( params, stream );

    return true;
}

bool DeviceConstantImage::readBaseColor( float4& dest )
{
    const int TILE_SIZE = 64;
    unsigned int mipLevel = 0;
    unsigned int levelSize = (m_info.width > m_info.height) ? m_info.width : m_info.height;
    // The mip tail all uses the same color, so find where the mip tail starts
    while( levelSize > TILE_SIZE )
    {
        levelSize /= 2;
        mipLevel++;
    }
    dest = m_mipColors[ mipLevel % m_mipColors.size() ];
    return true;
}
