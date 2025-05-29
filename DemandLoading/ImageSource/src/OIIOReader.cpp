// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/OIIOReader.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <half.h>
#include <mutex>
#include <vector>

#include "Stopwatch.h"

namespace imageSource {

CUarray_format pixelTypeToArrayFormat( const OIIO::TypeDesc& type )
{
    if( type == OIIO::TypeDesc::UINT8 || type == OIIO::TypeDesc::UCHAR )
        return CU_AD_FORMAT_UNSIGNED_INT8;
    if( type == OIIO::TypeDesc::UINT16 || type == OIIO::TypeDesc::USHORT )
        return CU_AD_FORMAT_UNSIGNED_INT16;
    if( type == OIIO::TypeDesc::UINT32 || type == OIIO::TypeDesc::UINT )
        return CU_AD_FORMAT_UNSIGNED_INT32;
    if( type == OIIO::TypeDesc::INT8 || type == OIIO::TypeDesc::CHAR )
        return CU_AD_FORMAT_SIGNED_INT8;
    if( type == OIIO::TypeDesc::INT16 || type == OIIO::TypeDesc::SHORT )
        return CU_AD_FORMAT_SIGNED_INT16;
    if( type == OIIO::TypeDesc::INT32 || type == OIIO::TypeDesc::INT )
        return CU_AD_FORMAT_SIGNED_INT32;
    if( type == OIIO::TypeDesc::HALF )
        return CU_AD_FORMAT_HALF;
    if( type == OIIO::TypeDesc::FLOAT )
        return CU_AD_FORMAT_FLOAT;

    OTK_ASSERT_MSG( false, "Invalid pixel type" );
    return CU_AD_FORMAT_FLOAT;
}

namespace {

float toFloat( const char* src, const CUarray_format format )
{
    switch( format )
    {
        case CU_AD_FORMAT_SIGNED_INT8:
            return static_cast<float>( static_cast<int8_t>( *src ) );
        case CU_AD_FORMAT_UNSIGNED_INT8:
            return static_cast<float>( static_cast<uint8_t>( *src ) );

        case CU_AD_FORMAT_SIGNED_INT16:
            return static_cast<float>( *( reinterpret_cast<const int16_t*>( src ) ) );
        case CU_AD_FORMAT_UNSIGNED_INT16:
            return static_cast<float>( *( reinterpret_cast<const uint16_t*>( src ) ) );

        case CU_AD_FORMAT_SIGNED_INT32:
            return static_cast<float>( *( reinterpret_cast<const int32_t*>( src ) ) );
        case CU_AD_FORMAT_UNSIGNED_INT32:
            return static_cast<float>( *( reinterpret_cast<const uint32_t*>( src ) ) );

        case CU_AD_FORMAT_HALF:
            return static_cast<float>( *( reinterpret_cast<const half*>( src ) ) );

        case CU_AD_FORMAT_FLOAT:
            return static_cast<float>( *( reinterpret_cast<const float*>( src ) ) );

        default:
            OTK_ASSERT_MSG( false, "Invalid CUDA array format" );
    }

    return 0.f;
}
}

// Open the image and read header info, including dimensions and format.
void OIIOReader::open( TextureInfo* info )
{
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        // Check to see if the image is already open
        if( !m_input )
        {
            m_input = OIIO::ImageInput::open( m_filename );
            if( !m_input )
            {
                throw std::runtime_error( OIIO::geterror().c_str() );
            }                

            OIIO::ImageSpec spec = m_input->spec();

            m_info.width  = spec.width;
            m_info.height = spec.height;
            m_depth       = spec.depth;

            m_info.format = pixelTypeToArrayFormat( spec.format );

            // CUDA textures don't support float3, so we round up to four channels.
            m_info.numChannels = ( spec.nchannels >= 3 ) ? 4 : spec.nchannels;

            m_info.numMipLevels = 0;
            while( m_input->seek_subimage( 0, m_info.numMipLevels ) )
            {
                spec = m_input->spec();
                ++m_info.numMipLevels;
                m_levelWidths.push_back( spec.width );
                m_levelHeights.push_back( spec.height );
            }

            m_levelWidths.shrink_to_fit();
            m_levelHeights.shrink_to_fit();

            m_info.isTiled = spec.tile_width > 0;
            m_info.isValid = true;
            m_tileWidth    = spec.tile_width;
            m_tileHeight   = spec.tile_height;
        }
    }

    if( m_readBaseColor && !m_baseColorWasRead && m_info.numMipLevels > 1 )
    {
        std::vector<char> tmp( getBitsPerPixel( m_info.format ) / BITS_PER_BYTE, 0 );
        readMipLevel( tmp.data(), m_info.numMipLevels - 1, 1, 1, 0 );

        float out[4]{};

        for( unsigned int i = 0; i < m_info.numChannels; ++i )
            out[i] = toFloat( tmp.data() + ( getBitsPerChannel( m_info ) / BITS_PER_BYTE ) * i, m_info.format );

        m_baseColor = float4{out[0], out[1], out[2], out[3]};

        m_baseColorWasRead = true;
    }

    if( info != nullptr )
        *info = m_info;
}

// Close the image.
void OIIOReader::close()
{
    std::lock_guard<std::mutex> guard( m_mutex );
    if( m_input )
    {
        m_input->close();
        m_input.reset();
    }
}

void OIIOReader::readActualTile( char* dest, unsigned int rowPitch, unsigned int mipLevel, unsigned int tileX, unsigned int tileY )
{
    std::lock_guard<std::mutex> guard( m_mutex );
    OTK_ASSERT( m_input.get() );

    OIIO::ImageSpec spec;
    m_input->seek_subimage( 0, mipLevel );
    spec = m_input->spec();
    m_input->read_tile( tileX * spec.tile_width, tileY * spec.tile_height, 0, spec.format, dest,
                        getBitsPerPixel( m_info ) / BITS_PER_BYTE, rowPitch );
}

bool OIIOReader::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream /*stream*/  )
{
    OTK_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );

    Stopwatch stopwatch;
    OIIO::ImageSpec spec;
    {
        std::lock_guard<std::mutex> guard( m_mutex );
        m_input->seek_subimage( 0, mipLevel );
        spec = m_input->spec();
    }

    if( spec.tile_width && spec.tile_height )
    {
        // We require that the requested tile size is an integer multiple of the file's tile size.
        const unsigned int actualTileWidth  = spec.tile_width;
        const unsigned int actualTileHeight = spec.tile_height;

        if( actualTileWidth > tile.width || tile.width % actualTileWidth != 0
            || actualTileHeight > tile.height || tile.height % actualTileHeight != 0 )
        {
            std::stringstream str;
            str << "Unsupported tile size (" << actualTileWidth << "x" << actualTileHeight << ").  Expected "
                << tile.width << "x" << tile.height << " (or a whole fraction thereof) for this pixel format";
            throw std::runtime_error( str.str().c_str() );
        }

        const unsigned int actualTileX    = tile.x * tile.width / actualTileWidth;
        const unsigned int actualTileY    = tile.y * tile.height / actualTileHeight;
        const unsigned int bytesPerPixel  = getBitsPerPixel( m_info ) / BITS_PER_BYTE;
        const unsigned int rowPitch       = tile.width * bytesPerPixel;
        const size_t       actualTileSize = actualTileWidth * actualTileHeight * bytesPerPixel;

        // Don't request non-existent tiles on the edge of the texture
        unsigned int levelWidthInSourceTiles  = ( m_levelWidths[mipLevel] + actualTileWidth - 1 ) / actualTileWidth;
        unsigned int levelHeightInSourceTiles = ( m_levelHeights[mipLevel] + actualTileHeight - 1 ) / actualTileHeight;
        const unsigned int numTilesX = std::min( tile.width / actualTileWidth, levelWidthInSourceTiles - actualTileX );
        const unsigned int numTilesY = std::min( tile.height / actualTileHeight, levelHeightInSourceTiles - actualTileY );

        for( unsigned int j = 0; j < numTilesY; ++j )
        {
            for( unsigned int i = 0; i < numTilesX; ++i )
            {
                char* start = dest + j * numTilesX * actualTileSize + i * actualTileWidth * bytesPerPixel;
                readActualTile( start, rowPitch, mipLevel, actualTileX + i, actualTileY + j );
            }
        }
    }
    else  // Scanline image
    {
        const unsigned int start_x = tile.x * tile.width;
        const unsigned int end_x   = std::min<int>( spec.width, start_x + tile.width );
        const unsigned int start_y = tile.y * tile.height;
        const unsigned int end_y   = std::min<int>( spec.height, start_y + tile.height );

        const unsigned int bytesPerPixel    = getBitsPerPixel( m_info.format ) / BITS_PER_BYTE;
        const unsigned int file_pixel_bytes = spec.pixel_bytes();
        std::vector<char>  tmp( spec.width * file_pixel_bytes );

        char* _dest = dest;
        for( unsigned int y = start_y; y < end_y; ++y )
        {
            {
                std::lock_guard<std::mutex> guard( m_mutex );
                m_input->read_scanline( y, 0, spec.format, tmp.data() );
            }

            for( unsigned int x = start_x; x < end_x; ++x )
            {
                memcpy( _dest, tmp.data() + x * file_pixel_bytes, file_pixel_bytes );
                _dest += bytesPerPixel;
            }
        }
    }
    {
        std::unique_lock<std::mutex> lock( m_statsMutex );
        m_totalReadTime += stopwatch.elapsed();
        ++m_numTilesRead;
    }
    return true;
}


bool OIIOReader::readBaseColor( float4& dest )
{
    dest = m_baseColor;
    return m_baseColorWasRead;
}


bool OIIOReader::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    return readMipLevel( dest, mipLevel, expectedWidth, expectedHeight, 1, stream );
}


bool OIIOReader::readMipLevel( char*        dest,
                               unsigned int mipLevel,
                               unsigned int expectedWidth,
                               unsigned int expectedHeight,
                               unsigned int expectedDepth,
                               CUstream     /*stream*/ )
{
    OTK_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read missing mip level" );

    Stopwatch stopwatch;
    OIIO::ImageSpec spec;
    unsigned int    bytesPerPixel;
    {
        std::lock_guard<std::mutex> guard( m_mutex );
        m_input->seek_subimage( 0, mipLevel );
        spec = m_input->spec();

        OTK_ASSERT( spec.width == static_cast<int>( expectedWidth ) );
        OTK_ASSERT( spec.height == static_cast<int>( expectedHeight ) );
        OTK_ASSERT( spec.depth == static_cast<int>( expectedDepth ) );
        (void)expectedWidth;  // silence unused variable warning.
        (void)expectedHeight;
        (void)expectedDepth;

        bytesPerPixel = getBitsPerPixel( m_info.format ) / BITS_PER_BYTE;

        m_input->read_image( 0, mipLevel, 0, spec.nchannels, spec.format, dest, bytesPerPixel );
    }

    if( spec.tile_width )
    {
        const unsigned int actualTileWidth  = spec.tile_width;
        const unsigned int actualTileHeight = spec.tile_height;
        const unsigned int actualTileDepth  = spec.tile_depth;
        const size_t       actualTileSize   = spec.tile_bytes();
        const int          numXTiles        = 1 + ( ( spec.width - 1 ) / actualTileWidth );
        const int          numYTiles        = 1 + ( ( spec.height - 1 ) / actualTileHeight );
        const int          numZTiles        = 1 + ( ( spec.depth - 1 ) / actualTileDepth );

        std::lock_guard<std::mutex> guard( m_statsMutex );
        m_numTilesRead += numXTiles * numYTiles * numZTiles;
        m_numBytesRead += numXTiles * numYTiles * numZTiles * actualTileSize;
        m_totalReadTime += stopwatch.elapsed();
    }
    else
    {
        std::lock_guard<std::mutex> guard( m_statsMutex );
        m_numTilesRead += 1;
        m_numBytesRead += spec.width * spec.height * spec.depth * bytesPerPixel;
        m_totalReadTime += stopwatch.elapsed();
    }
    return true;
}


}  // namespace imageSource
