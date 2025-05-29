// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/PbrtAlphaMapImageSource.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace imageSource;

namespace demandPbrtScene {

PbrtAlphaMapImageSource::PbrtAlphaMapImageSource( std::shared_ptr<ImageSource> baseImage )
    : WrappedImageSource( baseImage )
{
    if( baseImage->isOpen() )
    {
        getBaseInfo();
    }
}

void PbrtAlphaMapImageSource::getBaseInfo()
{
    m_baseInfo              = WrappedImageSource::getInfo();
    m_alphaInfo             = m_baseInfo;
    m_alphaInfo.format      = CU_AD_FORMAT_UNSIGNED_INT8;
    m_alphaInfo.numChannels = 1;
    m_basePixelStride       = getBitsPerPixel( m_baseInfo ) / BITS_PER_BYTE;
}

void PbrtAlphaMapImageSource::open( imageSource::TextureInfo* info )
{
    if( !WrappedImageSource::isOpen() )
    {
        WrappedImageSource::open( nullptr );
    }
    getBaseInfo();
    if( info )
    {
        *info = m_alphaInfo;
    }
}

void PbrtAlphaMapImageSource::convertBasePixels( char* buffer, unsigned int width, unsigned int height )
{
    const char*   source = m_basePixels.data();
    std::uint8_t* dest   = reinterpret_cast<std::uint8_t*>( buffer );
    for( unsigned int y = 0; y < height; ++y )
    {
        for( unsigned int x = 0; x < width; ++x )
        {
            // rely on the fact that floating-point zero is a bit-pattern of zeros.
            *dest = std::all_of( &source[0], &source[m_basePixelStride], []( char c ) { return c == 0; } ) ? 0U : 255U;
            ++dest;
            source += m_basePixelStride;
        }
    }
}

bool PbrtAlphaMapImageSource::readTile( char* buffer, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    m_basePixels.resize( tile.width * tile.height * m_basePixelStride );
    if( !WrappedImageSource::readTile( m_basePixels.data(), mipLevel, tile, stream) )
    {
        return false;
    }

    convertBasePixels( buffer, tile.width, tile.height );
    return true;
}

bool PbrtAlphaMapImageSource::readMipLevel( char* buffer, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    m_basePixels.resize( expectedWidth * expectedHeight * m_basePixelStride );
    if( !WrappedImageSource::readMipLevel( m_basePixels.data(), mipLevel, expectedWidth, expectedHeight, stream ) )
    {
        return false;
    }

    convertBasePixels( buffer, expectedWidth, expectedHeight );
    return true;
}

bool PbrtAlphaMapImageSource::readMipTail( char*        dest,
                                           unsigned int mipTailFirstLevel,
                                           unsigned int numMipLevels,
                                           const uint2* mipLevelDims,
                                           CUstream     stream )
{
    size_t offset = 0;
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims = mipLevelDims[mipLevel];
        readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream );
        offset += static_cast<size_t>( ( levelDims.x * levelDims.y * getBitsPerPixel( m_baseInfo ) ) / BITS_PER_BYTE );
    }

    return true;
}

}  // namespace demandPbrtScene
