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

#include <OptiXToolkit/ImageSource/TiledImageSource.h>

#include <OptiXToolkit/Error/ErrorCheck.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace imageSource {

TiledImageSource::TiledImageSource( std::shared_ptr<ImageSource> baseImage )
    : WrappedImageSource( std::move( baseImage ) )
{
}

void TiledImageSource::open( TextureInfo* info )
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    WrappedImageSource::open( &m_tiledInfo );
    m_baseIsTiled       = m_tiledInfo.isTiled;
    m_tiledInfo.isTiled = true;
    if( info != nullptr )
    {
        *info = m_tiledInfo;
    }
}

void TiledImageSource::close()
{
    WrappedImageSource::close();
    m_tiledInfo = TextureInfo{};
}

const TextureInfo& TiledImageSource::getInfo() const
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    if( m_baseIsTiled )
    {
        return WrappedImageSource::getInfo();
    }
    return m_tiledInfo;
}

bool TiledImageSource::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream )
{
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        if( m_baseIsTiled )
        {
            return WrappedImageSource::readTile( dest, mipLevel, tile, stream);
        }
    }

    const char* mipLevelBuffer;
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );

        if( m_buffer.empty() )
        {
            m_buffer.resize( getTextureSizeInBytes( m_tiledInfo ) );
            m_mipLevels.resize( m_tiledInfo.numMipLevels );
        }

        if( m_mipLevels[mipLevel] == nullptr )
        {
            size_t levelSize = static_cast<size_t>( getBytesPerChannel( m_tiledInfo.format ) ) * m_tiledInfo.numChannels
                               * m_tiledInfo.width * m_tiledInfo.height;
            char*        ptr            = m_buffer.data();
            unsigned int mipLevelWidth  = m_tiledInfo.width;
            unsigned int mipLevelHeight = m_tiledInfo.height;
            for( unsigned int i = 0; i < mipLevel; ++i )
            {
                ptr += levelSize;
                levelSize /= 4;
                mipLevelWidth /= 2;
                mipLevelHeight /= 2;
            }
            m_mipLevels[mipLevel] = ptr;
            if( !WrappedImageSource::readMipLevel( m_mipLevels[mipLevel], mipLevel, mipLevelWidth, mipLevelHeight, stream ) )
            {
                OTK_ASSERT(false);
                return false;
            }    
        }

        ++m_numTilesRead;
        mipLevelBuffer = m_mipLevels[mipLevel];
    }

    OTK_ASSERT_MSG( mipLevelBuffer != nullptr, ( "Bad pointer for level " + std::to_string( mipLevel ) ).c_str() );
    const size_t        pixelSizeInBytes           = getBytesPerChannel( m_tiledInfo.format ) * m_tiledInfo.numChannels;
    const size_t        imageRowStrideInBytes      = m_tiledInfo.width * pixelSizeInBytes;
    // Partial tile dimensions might be less than the nominal dimensions.
    const size_t        sourceTileWidth            = std::min( tile.width, m_tiledInfo.width - tile.x * tile.width );
    const size_t        sourceTileHeight           = std::min( tile.height, m_tiledInfo.height - tile.y * tile.height );
    const size_t        sourceTileRowStrideInBytes = sourceTileWidth * pixelSizeInBytes;
    const size_t        destTileRowStrideInBytes   = tile.width * pixelSizeInBytes;
    const PixelPosition start                      = pixelPosition( tile );
    const char*         source = mipLevelBuffer + start.y * imageRowStrideInBytes + start.x * pixelSizeInBytes;
    for( unsigned int i = 0; i < sourceTileHeight; ++i )
    {
        std::copy_n( source, sourceTileRowStrideInBytes, dest );
        dest += destTileRowStrideInBytes;
        source += imageRowStrideInBytes;
    }

    return true;
}

bool TiledImageSource::readMipTail( char*        dest,
                                    unsigned int mipTailFirstLevel,
                                    unsigned int numMipLevels,
                                    const uint2* mipLevelDims,
                                    unsigned int pixelSizeInBytes,
                                    CUstream     stream )
{
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        if( m_baseIsTiled )
        {
            return WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, pixelSizeInBytes, stream );
        }
    }

    size_t offset = 0;
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims = mipLevelDims[mipLevel];
        readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream );
        offset += static_cast<size_t>( levelDims.x * levelDims.y * pixelSizeInBytes );
    }

    return true;
}

unsigned long long TiledImageSource::getNumTilesRead() const
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    if( m_baseIsTiled )
    {
        return WrappedImageSource::getNumTilesRead();
    }
    return m_numTilesRead;
}

}  // namespace imageSource
