//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "PbrtAlphaMapImageSource.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
    m_basePixelStride       = imageSource::getBytesPerChannel( m_baseInfo.format ) * m_baseInfo.numChannels;
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
                                           unsigned int pixelSizeInBytes,
                                           CUstream     stream )
{
    size_t offset = 0;
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims = mipLevelDims[mipLevel];
        readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream );
        offset += static_cast<size_t>( levelDims.x * levelDims.y * pixelSizeInBytes );
    }

    return true;
}

}  // namespace demandPbrtScene
