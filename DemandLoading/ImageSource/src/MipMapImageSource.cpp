// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/MipMapImageSource.h>

#include <OptiXToolkit/Error/ErrorCheck.h>

#include <algorithm>

namespace imageSource {

MipMapImageSource::MipMapImageSource( std::shared_ptr<ImageSource> baseImage )
    : WrappedImageSource( baseImage )
{
    if( baseImage->isOpen() )
    {
        getBaseInfo();
    }
}

void MipMapImageSource::getBaseInfo()
{
    m_mipMapInfo = WrappedImageSource::getInfo();
    m_mipMappedBase = m_mipMapInfo.numMipLevels > 1;
    if( m_mipMappedBase )
    {
        return;
    }
    unsigned int width        = m_mipMapInfo.width;
    unsigned int height       = m_mipMapInfo.height;
    unsigned int numMipLevels = 1;
    while( width > 1 || height > 1 )
    {
        width /= 2;
        height /= 2;
        ++numMipLevels;
    }
    m_mipMapInfo.numMipLevels = numMipLevels;
    m_pixelStrideInBytes      = getBitsPerPixel( m_mipMapInfo ) / BITS_PER_BYTE;
    m_mipLevels.resize( numMipLevels );
}

void MipMapImageSource::open( TextureInfo* info )
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    if( !WrappedImageSource::isOpen() )
    {
        WrappedImageSource::open( nullptr );
    }
    getBaseInfo();
    if( m_mipMappedBase )
    {
        return;
    }

    if( info != nullptr )
    {
        *info = m_mipMapInfo;
    }
}

void MipMapImageSource::close()
{
    WrappedImageSource::close();
    m_mipMapInfo = TextureInfo{};
}

const TextureInfo& MipMapImageSource::getInfo() const
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    if( m_mipMappedBase )
    {
        return WrappedImageSource::getInfo();
    }
    return m_mipMapInfo;
}

const char* MipMapImageSource::getMipLevelBuffer( unsigned int mipLevel, CUstream stream )
{
    // Set up the mip levels if they haven't been set up yet
    if( m_buffer.empty() )
    {
        // Allocate memory for all the mip levels, and construct them
        m_buffer.resize( getTextureSizeInBytes( m_mipMapInfo ) );
        m_mipLevels.resize( m_mipMapInfo.numMipLevels );
        size_t offset = 0;

        for( unsigned int level = 0; level < m_mipMapInfo.numMipLevels; ++level )
        {
            unsigned int mipLevelWidth = std::max(1u, m_mipMapInfo.width >> level);
            unsigned int mipLevelHeight = std::max(1u, m_mipMapInfo.height >> level);
            m_mipLevels[level] = m_buffer.data() + offset;
            offset += m_pixelStrideInBytes * mipLevelWidth * mipLevelHeight;
        }

        // Read mip level 0
        if( !WrappedImageSource::readMipLevel( m_mipLevels[0], 0, m_mipMapInfo.width, m_mipMapInfo.height, stream ) )
        {
            for( unsigned int level = 0; level < m_mipMapInfo.numMipLevels; ++level )
            {
                m_mipLevels[level] = nullptr;
            }
            return nullptr;
        }

        // Construct the rest of the mip levels, copying pixels from the previous level
        for( unsigned int level = 1; level < m_mipMapInfo.numMipLevels; ++level )
        {
            const char* source = m_mipLevels[level - 1];
            char* dest   = m_mipLevels[level];
            int srcWidth = std::max(1u, m_mipMapInfo.width >> (level - 1));
            int srcHeight = std::max(1u, m_mipMapInfo.height >> (level - 1));
            int destWidth = std::max(1u, m_mipMapInfo.width >> level);
            int destHeight = std::max(1u, m_mipMapInfo.height >> level);

            for( int destY = 0; destY < destHeight; ++destY )
            {
                for( int destX = 0; destX < destWidth; ++destX )
                {
                    float u = static_cast<float>(destX+0.5f) / static_cast<float>(destWidth);
                    float v = static_cast<float>(destY+0.5f) / static_cast<float>(destHeight);
                    int srcX = static_cast<int>(u * srcWidth);
                    int srcY = static_cast<int>(v * srcHeight);

                    const char* sourcePixel = source + (srcY * srcWidth + srcX) * m_pixelStrideInBytes;
                    char* destPixel = dest + (destY * destWidth + destX) * m_pixelStrideInBytes;
                    std::copy_n( sourcePixel, m_pixelStrideInBytes, destPixel );
                }
            }
        }
    }

    return ( mipLevel < m_mipMapInfo.numMipLevels ) ? m_mipLevels[mipLevel] : nullptr;
}

bool MipMapImageSource::readTile( char* dest, unsigned mipLevel, const Tile& tile, CUstream stream )
{
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        if( m_mipMappedBase )
        {
            return WrappedImageSource::readTile( dest, mipLevel, tile, stream);
        }
    }

    const char* mipLevelBuffer;
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        mipLevelBuffer = getMipLevelBuffer( mipLevel, stream );
        if( mipLevelBuffer == nullptr )
            return false;

        ++m_numTilesRead;
    }
    unsigned int mipLevelWidth{ m_mipMapInfo.width };
    unsigned int level{ mipLevel };
    if( level > 0 )
    {
        do
        {
            mipLevelWidth /= 2;
            level--;
        } while( level > 0 );
    }
    const size_t        mipLevelRowStrideInBytes{ mipLevelWidth * m_pixelStrideInBytes };
    const size_t        tileRowStrideInBytes{ tile.width * m_pixelStrideInBytes };
    const PixelPosition start = pixelPosition( tile );
    const char*         source{ &mipLevelBuffer[start.y * mipLevelRowStrideInBytes + start.x * m_pixelStrideInBytes] };
    for( unsigned int i = 0; i < tile.height; ++i )
    {
        std::copy_n( source, tileRowStrideInBytes, dest );
        dest += tileRowStrideInBytes;
        source += mipLevelRowStrideInBytes;
    }

    return true;
}

bool MipMapImageSource::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        if( m_mipMappedBase )
        {
            return WrappedImageSource::readMipLevel( dest, mipLevel, expectedWidth, expectedHeight, stream );
        }
    }

    const char* mipLevelBuffer;
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        mipLevelBuffer = getMipLevelBuffer( mipLevel, stream );
        if( mipLevelBuffer == nullptr )
            return false;
    }

    std::copy_n( mipLevelBuffer, expectedWidth * expectedHeight * m_pixelStrideInBytes, dest );
    return true;
}

bool MipMapImageSource::readMipTail( char*        dest,
                                     unsigned int mipTailFirstLevel,
                                     unsigned int numMipLevels,
                                     const uint2* mipLevelDims,
                                     CUstream     stream )
{
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        if( m_mipMappedBase )
        {
            return WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
        }
    }

    size_t offset = 0;
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims = mipLevelDims[mipLevel];
        readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream );
        offset += static_cast<size_t>( ( levelDims.x * levelDims.y * getBitsPerPixel( m_mipMapInfo ) ) / BITS_PER_BYTE );
    }

    return true;
}

unsigned long long MipMapImageSource::getNumTilesRead() const
{
    std::unique_lock<std::mutex> lock( m_dataMutex );
    if( m_mipMappedBase )
    {
        return WrappedImageSource::getNumTilesRead();
    }

    return m_numTilesRead;
}

}  // namespace imageSource
