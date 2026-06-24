// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/OIIOReader.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <half.h>
#include <iostream>
#include <memory>
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

// Best-effort capability probe: OIIO's lock-free OpenEXR Core reader needs the package to
// have been built against OpenEXR >= 3.1.10. OIIO exposes no direct "core compiled" query,
// so we parse the linked OpenEXR version out of "library_list" and warn if it's too old.
void warnIfExrCoreUnavailable()
{
    std::string libs = OIIO::get_string_attribute( "library_list" );
    std::transform( libs.begin(), libs.end(), libs.begin(), []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

    // "library_list" is "name:descr version;name:descr version;...", e.g. "openexr:openexr 3.3.2",
    // so scan for the first numeric token after the "openexr:" key (not immediately after it).
    const std::string key = "openexr:";
    const size_t      pos = libs.find( key );
    std::string       version = "unknown";
    if( pos != std::string::npos )
    {
        const size_t      start = pos + key.size();
        const size_t      end   = libs.find( ';', start );
        const std::string entry = libs.substr( start, end == std::string::npos ? std::string::npos : end - start );
        const size_t      digit = entry.find_first_of( "0123456789" );
        if( digit != std::string::npos )
        {
            version = entry.substr( digit );

            int major = 0, minor = 0, patch = 0;
            if( std::sscanf( version.c_str(), "%d.%d.%d", &major, &minor, &patch ) >= 2 )
            {
                if( ( major > 3 ) || ( major == 3 && ( minor > 1 || ( minor == 1 && patch >= 10 ) ) ) )
                    return;
            }
        }
    }

    std::cerr << "OIIOReader: OpenImageIO's linked OpenEXR (" << version
              << ") may lack the lock-free OpenEXR Core reader (need >= 3.1.10); "
                 "EXR tile reads may fall back to the serialized C++ reader.\n";
}
}

// Open the image and read header info, including dimensions and format.
void OIIOReader::open( TextureInfo* info )
{
    // Prefer OIIO's lock-free OpenEXR Core reader. The attribute is process-wide and is
    // consulted when the EXR plugin is instantiated (during open), so set it once up front.
    static std::once_flag exrCoreOnce;
    std::call_once( exrCoreOnce, [] {
        OIIO::attribute( "openexr:core", 1 );
        warnIfExrCoreUnavailable();
    } );

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

            // Cache the file pixel layout (assumed uniform across mip levels; validated below).
            m_pixelFormat     = spec.format;
            m_numFileChannels = spec.nchannels;
            m_filePixelBytes  = static_cast<unsigned int>( m_pixelFormat.size() ) * m_numFileChannels;

            // Enumerate mip levels, caching per-level geometry so the lock-free, stateless
            // read paths never need seek_subimage()/spec().
            m_info.numMipLevels = 0;
            while( m_input->seek_subimage( 0, m_info.numMipLevels ) )
            {
                spec = m_input->spec();

                // m_info carries a single format/channel count, so every level must match.
                if( spec.format != m_pixelFormat || spec.nchannels != m_numFileChannels )
                    throw std::runtime_error( "OIIOReader: mip levels have differing pixel format or channel count: " + m_filename );

                ++m_info.numMipLevels;
                m_levelWidths.push_back( spec.width );
                m_levelHeights.push_back( spec.height );
                m_levelDepths.push_back( spec.depth );
                m_tileWidths.push_back( spec.tile_width );
                m_tileHeights.push_back( spec.tile_height );
                m_tileDepths.push_back( spec.tile_depth );
            }

            m_levelWidths.shrink_to_fit();
            m_levelHeights.shrink_to_fit();
            m_levelDepths.shrink_to_fit();
            m_tileWidths.shrink_to_fit();
            m_tileHeights.shrink_to_fit();
            m_tileDepths.shrink_to_fit();

            m_info.isTiled = !m_tileWidths.empty() && m_tileWidths.front() > 0;
            m_info.isValid = true;
            m_tileWidth    = m_tileWidths.empty() ? 0u : static_cast<unsigned int>( m_tileWidths.front() );
            m_tileHeight   = m_tileHeights.empty() ? 0u : static_cast<unsigned int>( m_tileHeights.front() );
        }
    }

    if( m_readBaseColor && !m_baseColorWasRead && m_info.numMipLevels > 1 )
    {
        std::vector<char> tmp( getBitsPerPixel( m_info ) / BITS_PER_BYTE, 0 );
        readMipLevel( tmp.data(), m_info.numMipLevels - 1, 1, 1, 0 );

        float out[4]{};

        for( unsigned int i = 0; i < m_info.numChannels; ++i )
            out[i] = toFloat( tmp.data() + ( getBitsPerChannel( m_info.format ) / BITS_PER_BYTE ) * i, m_info.format );

        m_baseColor = float4{out[0], out[1], out[2], out[3]};

        m_baseColorWasRead = true;
    }

    if( info != nullptr )
        *info = m_info;
}

// Close the image.
void OIIOReader::close()
{
    // Drop our reference under the lock. We must not call m_input->close() here: a concurrent
    // read may still be decoding through its own shared_ptr copy. The OIIO ImageInput closes the
    // file in its destructor, which runs once the last in-flight read releases its copy.
    std::lock_guard<std::mutex> guard( m_mutex );
    m_input.reset();
}

bool OIIOReader::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream /*stream*/  )
{
    OTK_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read missing mip level" );

    // Copy the input handle under the lock, then decode without it: reads run concurrently, and a
    // racing close() that resets m_input can't free the input out from under an in-flight decode.
    std::shared_ptr<OIIO::ImageInput> input;
    {
        std::lock_guard<std::mutex> guard( m_mutex );
        input = m_input;
    }
    if( !input )
        return false;

    Stopwatch stopwatch;

    // All metadata is cached in open(); the read uses the stateless, thread-safe overloads
    // (explicit subimage/miplevel) so concurrent tile reads run in parallel.
    const unsigned int bytesPerPixel    = getBitsPerPixel( m_info ) / BITS_PER_BYTE;
    const int          levelWidth       = m_levelWidths[mipLevel];
    const int          levelHeight      = m_levelHeights[mipLevel];
    const unsigned int actualTileWidth  = static_cast<unsigned int>( m_tileWidths[mipLevel] );
    const unsigned int actualTileHeight = static_cast<unsigned int>( m_tileHeights[mipLevel] );

    // Pixels outside the level must be black. The reads below only fill the in-bounds rectangle,
    // so zero the destination tile first when the request crosses the right/bottom edge.
    if( tile.x * tile.width + tile.width > static_cast<unsigned int>( levelWidth )
        || tile.y * tile.height + tile.height > static_cast<unsigned int>( levelHeight ) )
    {
        memset( dest, 0, static_cast<size_t>( tile.width ) * tile.height * bytesPerPixel );
    }

    if( actualTileWidth && actualTileHeight )
    {
        // We require that the requested tile size is an integer multiple of the file's tile size.
        if( actualTileWidth > tile.width || tile.width % actualTileWidth != 0
            || actualTileHeight > tile.height || tile.height % actualTileHeight != 0 )
        {
            std::stringstream str;
            str << "Unsupported tile size (" << actualTileWidth << "x" << actualTileHeight << ").  Expected "
                << tile.width << "x" << tile.height << " (or a whole fraction thereof) for this pixel format";
            throw std::runtime_error( str.str().c_str() );
        }

        const unsigned int rowPitch = tile.width * bytesPerPixel;

        // Read the (possibly multi-source-tile) region in a single stateless call. Clamp to
        // the level bounds so non-existent edge tiles aren't requested; OIIO places each
        // source tile at the correct destination offset via the strides.
        const int x0 = tile.x * tile.width;
        const int y0 = tile.y * tile.height;
        const int x1 = std::min<int>( x0 + tile.width, levelWidth );
        const int y1 = std::min<int>( y0 + tile.height, levelHeight );

        if( x1 > x0 && y1 > y0
            && !input->read_tiles( 0, mipLevel, x0, x1, y0, y1, 0, 1, 0, m_numFileChannels, m_pixelFormat, dest,
                                   /*xstride=*/bytesPerPixel, /*ystride=*/rowPitch ) )
        {
            throw std::runtime_error( input->geterror() );
        }
    }
    else  // Scanline image
    {
        const unsigned int start_x = tile.x * tile.width;
        const unsigned int end_x   = std::min<int>( levelWidth, start_x + tile.width );
        const unsigned int start_y = tile.y * tile.height;
        const unsigned int end_y   = std::min<int>( levelHeight, start_y + tile.height );

        const unsigned int file_pixel_bytes = m_filePixelBytes;
        std::vector<char>  tmp( levelWidth * file_pixel_bytes );

        char* _dest = dest;
        for( unsigned int y = start_y; y < end_y; ++y )
        {
            if( !input->read_scanlines( 0, mipLevel, y, y + 1, 0, 0, m_numFileChannels, m_pixelFormat, tmp.data() ) )
                throw std::runtime_error( input->geterror() );

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

    // Copy the input handle under the lock, then decode without it (see readTile).
    std::shared_ptr<OIIO::ImageInput> input;
    {
        std::lock_guard<std::mutex> guard( m_mutex );
        input = m_input;
    }
    if( !input )
        return false;

    Stopwatch stopwatch;

    // Metadata is cached in open(); the whole-level read uses the stateless, thread-safe
    // read_image overload (explicit subimage/miplevel) and holds no lock.
    const int levelWidth  = m_levelWidths[mipLevel];
    const int levelHeight = m_levelHeights[mipLevel];
    const int levelDepth  = m_levelDepths[mipLevel];

    OTK_ASSERT( levelWidth == static_cast<int>( expectedWidth ) );
    OTK_ASSERT( levelHeight == static_cast<int>( expectedHeight ) );
    OTK_ASSERT( levelDepth == static_cast<int>( expectedDepth ) );
    (void)expectedWidth;  // silence unused variable warning.
    (void)expectedHeight;
    (void)expectedDepth;

    const unsigned int bytesPerPixel = getBitsPerPixel( m_info ) / BITS_PER_BYTE;

    if( !input->read_image( 0, mipLevel, 0, m_numFileChannels, m_pixelFormat, dest, bytesPerPixel ) )
        throw std::runtime_error( input->geterror() );

    const unsigned int actualTileWidth = static_cast<unsigned int>( m_tileWidths[mipLevel] );
    if( actualTileWidth )
    {
        const unsigned int actualTileHeight = static_cast<unsigned int>( m_tileHeights[mipLevel] );
        const unsigned int actualTileDepth  = static_cast<unsigned int>( m_tileDepths[mipLevel] );
        const size_t       actualTileSize =
            static_cast<size_t>( actualTileWidth ) * actualTileHeight * actualTileDepth * m_filePixelBytes;
        const int numXTiles = 1 + ( ( levelWidth - 1 ) / static_cast<int>( actualTileWidth ) );
        const int numYTiles = 1 + ( ( levelHeight - 1 ) / static_cast<int>( actualTileHeight ) );
        const int numZTiles = 1 + ( ( levelDepth - 1 ) / static_cast<int>( actualTileDepth ) );

        std::lock_guard<std::mutex> guard( m_statsMutex );
        m_numTilesRead += numXTiles * numYTiles * numZTiles;
        m_numBytesRead += numXTiles * numYTiles * numZTiles * actualTileSize;
        m_totalReadTime += stopwatch.elapsed();
    }
    else
    {
        std::lock_guard<std::mutex> guard( m_statsMutex );
        m_numTilesRead += 1;
        m_numBytesRead += static_cast<unsigned long long>( levelWidth ) * levelHeight * levelDepth * bytesPerPixel;
        m_totalReadTime += stopwatch.elapsed();
    }
    return true;
}


}  // namespace imageSource
