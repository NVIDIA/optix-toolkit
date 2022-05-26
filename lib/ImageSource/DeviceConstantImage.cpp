//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <ImageSource/DeviceConstantImage.h>

#include "Exception.h"

namespace imageSource {

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

void DeviceConstantImage::readTile( char*        dest,
                                      unsigned int mipLevel,
                                      unsigned int tileX,
                                      unsigned int tileY,
                                      unsigned int tileWidth,
                                      unsigned int tileHeight,
                                      CUstream     stream )
{
    DEMAND_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    DeviceConstantImageParams params;
    params.num_pixels = tileWidth * tileHeight;
    params.color = m_mipColors[ mipLevel % m_mipColors.size() ];
    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadConstantImage( params, stream );
}

void DeviceConstantImage::readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream )
{
    DEMAND_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = std::max( 1u, m_info.width >> mipLevel );
    const unsigned int levelHeight = std::max( 1u, m_info.height >> mipLevel );

    DEMAND_ASSERT_MSG( levelWidth == width && levelHeight == height,
                       "Mismatch between parameter and calculated mip level size." );

    DeviceConstantImageParams params{};
    params.num_pixels    = levelWidth * levelHeight;
    params.color         = m_mipColors[mipLevel % m_mipColors.size()];
    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadConstantImage( params, stream );
}

void DeviceConstantImage::readMipTail( char*        dest,
                                       unsigned int mipTailFirstLevel,
                                       unsigned int numMipLevels,
                                       const uint2* mipLevelDims,
                                       unsigned int pixelSizeInBytes,
                                       CUstream     stream )
{
    DEMAND_ASSERT_MSG( mipTailFirstLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );

    const unsigned int levelWidth  = mipLevelDims[mipTailFirstLevel].x;
    const unsigned int levelHeight = mipLevelDims[mipTailFirstLevel].y;

    DeviceConstantImageParams params{};
    params.num_pixels = levelWidth * levelHeight * 4 / 3;
    params.color = m_mipColors[ mipTailFirstLevel % m_mipColors.size() ];
    params.output_buffer = reinterpret_cast<float4*>( dest );

    launchReadConstantImage( params, stream );
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

}  // namespace imageSource
