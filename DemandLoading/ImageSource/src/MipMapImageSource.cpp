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
    m_pixelStrideInBytes      = getBytesPerChannel( m_mipMapInfo.format ) * m_mipMapInfo.numChannels;
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

inline void filterPixel( const char* source, char* dest, unsigned int pixelStride, CUarray_format /*format*/ )
{
    std::copy_n( source, pixelStride, dest );
}

const char* MipMapImageSource::getMipLevelBuffer( unsigned int mipLevel, CUstream stream )
{
    if( m_buffer.empty() )
    {
        m_buffer.resize( getTextureSizeInBytes( m_mipMapInfo ) );
        m_mipLevels.resize( m_mipMapInfo.numMipLevels );
    }

    if( m_mipLevels[mipLevel] == nullptr )
    {
        unsigned int mipLevelWidth{ m_mipMapInfo.width };
        unsigned int mipLevelHeight{ m_mipMapInfo.height };
        {
            size_t       levelSize{ m_pixelStrideInBytes * m_mipMapInfo.width * m_mipMapInfo.height };
            char*        ptr{ m_buffer.data() };
            for( unsigned int i = 0; i < mipLevel; ++i )
            {
                ptr += levelSize;
                levelSize /= 4;
                mipLevelWidth /= 2;
                mipLevelHeight /= 2;
            }
            m_mipLevels[mipLevel] = ptr;
        }
        if( mipLevel == 0 )
        {
            if( !WrappedImageSource::readMipLevel( m_mipLevels[mipLevel], mipLevel, mipLevelWidth, mipLevelHeight, stream ) )
            {
                return nullptr;
            }
        }
        else
        {
            // point sample from source to dest
            const char* source = getMipLevelBuffer( mipLevel - 1, stream );
            char*       dest   = m_mipLevels[mipLevel];
            for( unsigned int y = 0; y < mipLevelHeight; ++y )
            {
                for( unsigned int x = 0; x < mipLevelWidth; ++x )
                {
                    filterPixel( source, dest, m_pixelStrideInBytes, m_mipMapInfo.format );
                    // skip every other source pixel
                    source += m_pixelStrideInBytes * 2ULL;
                    dest += m_pixelStrideInBytes;
                }
                // skip every other source scanline
                source += static_cast<size_t>( m_pixelStrideInBytes * mipLevelWidth );
            }
        }
    }

    return m_mipLevels[mipLevel];
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
                                     unsigned int pixelSizeInBytes,
                                     CUstream     stream )
{
    {
        std::unique_lock<std::mutex> lock( m_dataMutex );
        if( m_mipMappedBase )
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
