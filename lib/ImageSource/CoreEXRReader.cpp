//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <ImageSource/CoreEXRReader.h>

#include "Exception.h"
#include "Stopwatch.h"

#include <cuda_runtime.h>

#include <half.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace imageSource {

CUarray_format pixelTypeToArrayFormat( exr_pixel_type_t type )
{
    switch( type )
    {
        case EXR_PIXEL_UINT:
            return CU_AD_FORMAT_UNSIGNED_INT32;
        case EXR_PIXEL_HALF:
            return CU_AD_FORMAT_HALF;
        case EXR_PIXEL_FLOAT:
            return CU_AD_FORMAT_FLOAT;
        default:
            DEMAND_ASSERT_MSG( false, "Invalid EXR pixel type" );
            return CU_AD_FORMAT_FLOAT;
    }
}

// Open the image and read header info, including dimensions and format.  Throws an exception on error.
void CoreEXRReader::open( TextureInfo* info )
{
    if( !m_exrCtx )
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_info.isValid = false;

            exr_context_initializer_t cinit = EXR_DEFAULT_CONTEXT_INITIALIZER;
            cinit.error_handler_fn          = nullptr;

            DEMAND_ASSERT( exr_start_read( &m_exrCtx, m_filename.c_str(), &cinit ) == EXR_ERR_SUCCESS );

            // Get the width and height from the data window of the finest mipLevel.
            exr_attr_box2i_t dw;
            exr_get_data_window( m_exrCtx, m_partIndex, &dw );
            m_info.width  = dw.max.x - dw.min.x + 1;
            m_info.height = dw.max.y - dw.min.y + 1;

            exr_storage_t storageType;
            DEMAND_ASSERT( exr_get_storage( m_exrCtx, m_partIndex, &storageType ) == EXR_ERR_SUCCESS );
            DEMAND_ASSERT_MSG( storageType == EXR_STORAGE_SCANLINE || storageType == EXR_STORAGE_TILED, "CoreEXR Reader doesn't support deep files." );
            m_isScanline = storageType == EXR_STORAGE_SCANLINE;

            // Note that non-power-of-two EXR files often have one fewer miplevel than one would expect
            // (they don't round up from 1+log2(max(width/height))).
            int numMipLevelsX = 1, numMipLevelsY = 1;
            if( !m_isScanline )
                DEMAND_ASSERT( exr_get_tile_levels( m_exrCtx, m_partIndex, &numMipLevelsX, &numMipLevelsY ) == EXR_ERR_SUCCESS );

            DEMAND_ASSERT_MSG( numMipLevelsX == numMipLevelsY, "Number of mip levels must match for X and Y" );
            m_info.numMipLevels = static_cast<unsigned int>( numMipLevelsX );

            if( !m_isScanline )
            {
                // Get the tile specifications.
                DEMAND_ASSERT( exr_get_tile_descriptor( m_exrCtx, m_partIndex, &m_tileWidth, &m_tileHeight, &m_levelMode, &m_roundMode )
                               == EXR_ERR_SUCCESS );

                // Cache the level dimensions and tile tile dimensions for each mip level
                for( int mipLevel = 0; mipLevel < numMipLevelsX; ++mipLevel )
                {
                    int tileWidth, tileHeight, levelSizeX, levelSizeY;
                    DEMAND_ASSERT( exr_get_tile_sizes( m_exrCtx, m_partIndex, mipLevel, mipLevel, &tileWidth, &tileHeight ) == EXR_ERR_SUCCESS );
                    DEMAND_ASSERT( exr_get_level_sizes( m_exrCtx, m_partIndex, mipLevel, mipLevel, &levelSizeX, &levelSizeY ) == EXR_ERR_SUCCESS );

                    m_tileWidths[mipLevel]   = tileWidth;
                    m_tileHeights[mipLevel]  = tileHeight;
                    m_levelWidths[mipLevel]  = levelSizeX;
                    m_levelHeights[mipLevel] = levelSizeY;
                }
            }

            // Get channel list.
            const exr_attr_chlist_t* chlist = nullptr;
            DEMAND_ASSERT( exr_get_channels( m_exrCtx, m_partIndex, &chlist ) == EXR_ERR_SUCCESS );
            DEMAND_ASSERT_MSG( chlist->num_channels > 0, "No channels found in EXR file" );
            DEMAND_ASSERT_MSG( chlist->num_channels <= 4, "More than four channels found in EXR file" );

            // CUDA textures don't support float3, so we round up to four channels.
            m_info.numChannels = ( chlist->num_channels == 3 ) ? 4 : chlist->num_channels;
            m_pixelType        = chlist->entries[0].pixel_type;
            m_info.format      = pixelTypeToArrayFormat( m_pixelType );
            
            m_info.isTiled = !m_isScanline;
            m_info.isValid = true;
        }

        // Read the base color from the file
        // FIXME: There should be an option to have this available in the metadata, so
        // we don't have to read the level.
        if( m_readBaseColor && ( m_info.numMipLevels > 1 || ( m_info.width == 1 && m_info.height == 1 ) ) )
        {
            char buff[16] = { 0 };

            // Need to use internal read methods here, because we are already holding a lock on the mutex.
            if( m_isScanline )
                readScanlineData( buff );
            else
                readActualTile( buff, getBytesPerChannel( m_info.format ) * m_info.numChannels, m_info.numMipLevels - 1, 0, 0 );

            if( m_info.format == CU_AD_FORMAT_HALF )
            {
                Imath::half* h     = reinterpret_cast<Imath::half*>( buff );
                m_baseColor        = float4{ float( h[0] ), float( h[1] ), float( h[2] ), float( h[3] ) };
                m_baseColorWasRead = true;
            }
            else if( m_info.format == CU_AD_FORMAT_FLOAT )
            {
                float* f           = reinterpret_cast<float*>( buff );
                m_baseColor        = float4{ f[0], f[1], f[2], f[3] };
                m_baseColorWasRead = true;
            }
            else if( m_info.format == CU_AD_FORMAT_UNSIGNED_INT32 )
            {
                unsigned int* f    = reinterpret_cast<unsigned int*>( buff );
                m_baseColor        = float4{ float( f[0] ), float( f[1] ), float( f[2] ), float( f[3] ) };
                m_baseColorWasRead = true;
            }
        }
    }

    if( info != nullptr )
        *info = m_info;
}

// Close the image.
void CoreEXRReader::close()
{
    if( m_exrCtx != nullptr )
    {
        DEMAND_ASSERT( exr_finish( &m_exrCtx ) == EXR_ERR_SUCCESS );
    }
    m_exrCtx = nullptr;
}

void CoreEXRReader::readActualTile( char* dest, int rowPitch, int mipLevel, int tileX, int tileY )
{
    DEMAND_ASSERT( !m_isScanline );

    const int numXTiles = ( m_levelWidths[mipLevel] + m_tileWidths[mipLevel] - 1 ) / m_tileWidths[mipLevel];
    const int numYTiles = ( m_levelHeights[mipLevel] + m_tileHeights[mipLevel] - 1 ) / m_tileHeights[mipLevel];

    if( tileX >= numXTiles || tileY > numYTiles )
    {
        std::cerr << "Warning: Attempting to read non-existent tile [" << tileX << ", " << tileY << "]" << std::endl;
        return;
    }

    // Determine if we are reading a boundary tile, and adjust the tile dimensions to account for
    // partial tiles as necessary.
    const int  sourceTileWidth  = m_tileWidths[mipLevel];
    const int  sourceTileHeight = m_tileHeights[mipLevel];
    const bool partialX         = ( tileX == numXTiles - 1 ) && ( m_levelWidths[mipLevel] % sourceTileWidth );
    const bool partialY         = ( tileY == numYTiles - 1 ) && ( m_levelHeights[mipLevel] % sourceTileHeight );
    const int  actualTileWidth  = partialX ? m_levelWidths[mipLevel] % sourceTileWidth : sourceTileWidth;
    const int  actualTileHeight = partialY ? m_levelHeights[mipLevel] % sourceTileHeight : sourceTileHeight;

    exr_chunk_info_t      cinfo;
    exr_decode_pipeline_t decoder;
    DEMAND_ASSERT( exr_read_tile_chunk_info( m_exrCtx, m_partIndex, tileX, tileY, mipLevel, mipLevel, &cinfo ) == EXR_ERR_SUCCESS );
    DEMAND_ASSERT( exr_decoding_initialize( m_exrCtx, 0, &cinfo, &decoder ) == EXR_ERR_SUCCESS );

    const int bytesPerChannel = decoder.channels[0].bytes_per_element;

    // Setup the outputs
    for( int c = 0; c < decoder.channel_count; ++c )
    {
        DEMAND_ASSERT_MSG( decoder.channels[c].bytes_per_element == bytesPerChannel,
                           "All channels must have same bit depth" );

        int channelIdx = -1;
        if( strcmp( "R", decoder.channels[c].channel_name ) == 0 )
            channelIdx = 0;
        else if( strcmp( "G", decoder.channels[c].channel_name ) == 0 )
            channelIdx = 1;
        else if( strcmp( "B", decoder.channels[c].channel_name ) == 0 )
            channelIdx = 2;
        else if( strcmp( "A", decoder.channels[c].channel_name ) == 0 )
            channelIdx = 3;

        DEMAND_ASSERT_MSG( channelIdx >= 0 && channelIdx < 4, "Channel index out of range" );

        decoder.channels[c].decode_to_ptr = reinterpret_cast<uint8_t*>( dest ) + channelIdx * decoder.channels[c].bytes_per_element;
        decoder.channels[c].user_pixel_stride      = m_info.numChannels * decoder.channels[c].bytes_per_element;
        decoder.channels[c].user_line_stride       = rowPitch;
        decoder.channels[c].user_bytes_per_element = decoder.channels[c].bytes_per_element;
    }

    // Run the decoder
    DEMAND_ASSERT( exr_decoding_choose_default_routines( m_exrCtx, 0, &decoder ) == EXR_ERR_SUCCESS );
    DEMAND_ASSERT( exr_decoding_run( m_exrCtx, 0, &decoder ) == EXR_ERR_SUCCESS );
    DEMAND_ASSERT( exr_decoding_destroy( m_exrCtx, &decoder ) == EXR_ERR_SUCCESS );

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_numTilesRead += 1;
        m_numBytesRead += actualTileWidth * actualTileHeight * bytesPerChannel * m_info.numChannels;
    }
}

void CoreEXRReader::readScanlineData( char* dest )
{
    DEMAND_ASSERT( m_isScanline );

    int scanlinesPerChunk;
    DEMAND_ASSERT( exr_get_scanlines_per_chunk( m_exrCtx, m_partIndex, &scanlinesPerChunk ) == EXR_ERR_SUCCESS );

    size_t offset = 0;
    for( int y = 0; y < (int)m_info.height; y += scanlinesPerChunk )
    {
        exr_chunk_info_t      cinfo;
        exr_decode_pipeline_t decoder;
        DEMAND_ASSERT( exr_read_scanline_chunk_info( m_exrCtx, m_partIndex, y, &cinfo ) == EXR_ERR_SUCCESS );
        DEMAND_ASSERT( exr_decoding_initialize( m_exrCtx, 0, &cinfo, &decoder ) == EXR_ERR_SUCCESS );

        const int bytesPerElement = decoder.channels[0].bytes_per_element;

        // Setup the outputs
        for( int c = 0; c < decoder.channel_count; ++c )
        {
            DEMAND_ASSERT_MSG( decoder.channels[c].bytes_per_element == bytesPerElement,
                                "All channels must have same bit depth" );

            int channelIdx = -1;
            if( strcmp( "R", decoder.channels[c].channel_name ) == 0 || strcmp( "Y", decoder.channels[c].channel_name ) == 0 )
                channelIdx = 0;
            else if( strcmp( "G", decoder.channels[c].channel_name ) == 0 )
                channelIdx = 1;
            else if( strcmp( "B", decoder.channels[c].channel_name ) == 0 )
                channelIdx = 2;
            else if( strcmp( "A", decoder.channels[c].channel_name ) == 0 )
                channelIdx = 3;

            DEMAND_ASSERT_MSG( channelIdx >= 0 && channelIdx < 4, "Channel index out of range" );

            decoder.channels[c].decode_to_ptr = reinterpret_cast<uint8_t*>( dest ) + offset + channelIdx * decoder.channels[c].bytes_per_element;
            decoder.channels[c].user_pixel_stride      = m_info.numChannels * decoder.channels[c].bytes_per_element;
            decoder.channels[c].user_line_stride       = m_info.width * decoder.channels[c].user_pixel_stride;
            decoder.channels[c].user_bytes_per_element = decoder.channels[c].bytes_per_element;
        }

        // Run the decoder
        DEMAND_ASSERT( exr_decoding_choose_default_routines( m_exrCtx, 0, &decoder ) == EXR_ERR_SUCCESS );
        DEMAND_ASSERT( exr_decoding_run( m_exrCtx, 0, &decoder ) == EXR_ERR_SUCCESS );
        DEMAND_ASSERT( exr_decoding_destroy( m_exrCtx, &decoder ) == EXR_ERR_SUCCESS );

        offset += m_info.width * m_info.numChannels * bytesPerElement * scanlinesPerChunk;
    }

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_numTilesRead += 1;
        m_numBytesRead += m_info.height * m_info.width * m_info.numChannels * getBytesPerChannel( m_info.format );
    }
}

void CoreEXRReader::readTile( char*        dest,
                              unsigned int mipLevel,
                              unsigned int tileX,
                              unsigned int tileY,
                              unsigned int destTileWidth,
                              unsigned int destTileHeight,
                              CUstream     stream )
{
    DEMAND_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );
    DEMAND_ASSERT_MSG( !m_isScanline, "Attempting to read tiled data from scanline image." );

    // Stats tracking
    Stopwatch stopwatch;

    const int sourceTileWidth  = m_tileWidths[mipLevel];
    const int sourceTileHeight = m_tileHeights[mipLevel];

    // We require that the requested tile size is an integer multiple of the EXR tile size.
    if( !( sourceTileWidth <= static_cast<int>( destTileWidth ) && destTileWidth % sourceTileWidth == 0 )
        || !( sourceTileHeight <= static_cast<int>( destTileHeight ) && destTileHeight % sourceTileHeight == 0 ) )
    {
        std::stringstream str;
        str << "Unsupported EXR tile size (" << sourceTileWidth << "x" << sourceTileHeight << ").  Expected "
            << destTileWidth << "x" << destTileHeight << " (or a whole fraction thereof) for this pixel format";
        throw imageSource::Exception( str.str().c_str() );
    }

    const int actualTileX    = tileX * ( destTileWidth / sourceTileWidth );
    const int actualTileY    = tileY * ( destTileHeight / sourceTileHeight );
    const int numTilesX      = destTileWidth / sourceTileWidth;
    const int numTilesY      = destTileHeight / sourceTileHeight;
    const int bytesPerPixel  = getBytesPerChannel( m_info.format ) * m_info.numChannels;
    const int rowPitch       = destTileWidth * bytesPerPixel;
    const int sourceTileSize = sourceTileWidth * sourceTileHeight * bytesPerPixel;

    for( int j = 0; j < numTilesY; ++j )
    {
        for( int i = 0; i < numTilesX; ++i )
        {
            char* start = dest + j * numTilesX * sourceTileSize + i * sourceTileWidth * bytesPerPixel;
            readActualTile( start, rowPitch, mipLevel, actualTileX + i, actualTileY + j );
        }
    }

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_totalReadTime += stopwatch.elapsed();
    }
}

void CoreEXRReader::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    DEMAND_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );

    // Stats tracking
    Stopwatch stopwatch;

    if( m_isScanline )
    {
        DEMAND_ASSERT( mipLevel == 0 );
        DEMAND_ASSERT( expectedWidth == m_info.width && expectedHeight == m_info.height );
        readScanlineData( dest );
    }
    else
    {
        const int numXTiles     = ( m_levelWidths[mipLevel] + m_tileWidths[mipLevel] - 1 ) / m_tileWidths[mipLevel];
        const int numYTiles     = ( m_levelHeights[mipLevel] + m_tileHeights[mipLevel] - 1 ) / m_tileHeights[mipLevel];
        const int bytesPerPixel = getBytesPerChannel( m_info.format ) * m_info.numChannels;

        for( int rowIdx = 0; rowIdx < numYTiles; ++rowIdx )
        {
            const int rowOffset = rowIdx * m_levelWidths[mipLevel] * m_tileHeights[mipLevel];
            for( int colIdx = 0; colIdx < numXTiles; ++colIdx )
            {
                const int colOffset = colIdx * m_tileWidths[mipLevel];
                char*     outPtr    = &dest[( rowOffset + colOffset ) * bytesPerPixel];
                readActualTile( outPtr, expectedWidth * bytesPerPixel, mipLevel, colIdx, rowIdx );
            }
        }
    }

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_totalReadTime += stopwatch.elapsed();
    }
}

bool CoreEXRReader::readBaseColor( float4& dest )
{
    dest = m_baseColor;
    return m_baseColorWasRead;
}

}  // namespace demandLoading
