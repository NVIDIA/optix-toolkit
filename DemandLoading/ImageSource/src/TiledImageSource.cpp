// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/TiledImageSource.h>

#include <OptiXToolkit/Error/ErrorCheck.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace imageSource {

TiledImageSource::TiledImageSource( std::shared_ptr<ImageSource> baseImage )
    : WrappedImageSource( baseImage )
{
    if( baseImage->isOpen() )
    {
        getBaseInfo();
    }
}

void TiledImageSource::getBaseInfo()
{
    m_tiledInfo = WrappedImageSource::getInfo();
    m_baseIsTiled       = m_tiledInfo.isTiled;
    m_tiledInfo.isTiled = true;
}

void TiledImageSource::open( TextureInfo* info )
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    if( !WrappedImageSource::isOpen() )
    {
        WrappedImageSource::open( nullptr );
    }
    getBaseInfo();
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
    uint2 mipDimensions;
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );

        if( m_buffer.empty() )
        {
            m_buffer.resize( getTextureSizeInBytes( m_tiledInfo ) );
            m_mipLevels.resize( m_tiledInfo.numMipLevels );
            m_mipDimensions.resize( m_tiledInfo.numMipLevels );
        }

        if( m_mipLevels[mipLevel] == nullptr )
        {
            size_t levelSize = ( m_tiledInfo.width * m_tiledInfo.height * getBitsPerPixel( m_tiledInfo ) ) / BITS_PER_BYTE;
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
            m_mipDimensions[mipLevel].x = mipLevelWidth;
            m_mipDimensions[mipLevel].y = mipLevelHeight;
            if( !WrappedImageSource::readMipLevel( m_mipLevels[mipLevel], mipLevel, mipLevelWidth, mipLevelHeight, stream ) )
            {
                OTK_ASSERT(false);
                return false;
            }    
        }

        ++m_numTilesRead;
        mipLevelBuffer = m_mipLevels[mipLevel];
        mipDimensions = m_mipDimensions[mipLevel];
    }

    OTK_ASSERT_MSG( mipLevelBuffer != nullptr, ( "Bad pointer for level " + std::to_string( mipLevel ) ).c_str() );
    const size_t        pixelSizeInBytes         = getBitsPerPixel( m_tiledInfo ) / BITS_PER_BYTE;
    // Partial tile dimensions might be less than the nominal dimensions.
    const size_t        sourceWidth              = std::min( tile.width, mipDimensions.x - tile.x * tile.width );
    const size_t        sourceHeight             = std::min( tile.height, mipDimensions.y - tile.y * tile.height );
    const size_t        sourceRowWidthInBytes    = sourceWidth * pixelSizeInBytes;
    const size_t        sourceRowStrideInBytes   = mipDimensions.x * pixelSizeInBytes;
    const size_t        destTileRowStrideInBytes = tile.width * pixelSizeInBytes;
    const PixelPosition start                    = pixelPosition( tile );
    const char*         source = mipLevelBuffer + start.y * sourceRowStrideInBytes + start.x * pixelSizeInBytes;
    for( unsigned int i = 0; i < sourceHeight; ++i )
    {
        std::copy_n( source, sourceRowWidthInBytes, dest );
        dest += destTileRowStrideInBytes;
        source += sourceRowStrideInBytes;
    }

    return true;
}

bool TiledImageSource::readMipTail( char*        dest,
                                    unsigned int mipTailFirstLevel,
                                    unsigned int numMipLevels,
                                    const uint2* mipLevelDims,
                                    CUstream     stream )
{
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        if( m_baseIsTiled )
        {
            return WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
        }
    }

    size_t offset = 0;
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims = mipLevelDims[mipLevel];
        readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream );
        offset += static_cast<size_t>( ( levelDims.x * levelDims.y * getBitsPerPixel( m_tiledInfo ) ) / BITS_PER_BYTE );
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
