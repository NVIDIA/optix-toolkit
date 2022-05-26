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

#include <ImageSource/EXRReader.h>

#include "Exception.h"
#include "Stopwatch.h"

#include <cuda_runtime.h>

#include <half.h>
#include <ImfChannelList.h>
#include <ImfInputFile.h>
#include <ImfTiledInputFile.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

using namespace Imf;
using namespace Imath;

namespace imageSource {

CUarray_format pixelTypeToArrayFormat( PixelType type )
{
    switch( type )
    {
        case UINT:
            return CU_AD_FORMAT_UNSIGNED_INT32;
        case HALF:
            return CU_AD_FORMAT_HALF;
        case FLOAT:
            return CU_AD_FORMAT_FLOAT;
        default:
            DEMAND_ASSERT_MSG( false, "Invalid EXR pixel type" );
            return CU_AD_FORMAT_FLOAT;
    }
}

// Open the image and read header info, including dimensions and format. Throws an exception on error.
void EXRReader::open( TextureInfo* info )
{
    {
        Stopwatch stopwatch;
        std::unique_lock<std::mutex> lock( m_mutex );

        // Check to see if the image is already open
        if( !m_inputFile && !m_tiledInputFile )
        {
            m_info.isValid = false;

            // Open input file. May throw.
            m_inputFile.reset( new InputFile( m_filename.c_str() ) );

            if( m_inputFile->header().hasTileDescription() )
            {
                m_tiledInputFile.reset( new TiledInputFile( m_filename.c_str() ) );

                // Note that non-power-of-two EXR files often have one fewer miplevel than one would expect
                // (they don't round up from 1+log2(max(width/height))).
                DEMAND_ASSERT( m_tiledInputFile->numLevels() != 0 );
                m_info.numMipLevels = m_tiledInputFile->numLevels();
                m_tileWidth         = m_tiledInputFile->tileXSize();
                m_tileHeight        = m_tiledInputFile->tileYSize();
                m_info.isTiled      = true;
            }
            else
            {
                m_info.numMipLevels = 1;
                m_info.isTiled      = false;
            }

            // Get the width and height from the data window of the finest mipLevel.
            const Box2i dw = m_inputFile->header().dataWindow();
            m_info.width   = dw.max.x - dw.min.x + 1;
            m_info.height  = dw.max.y - dw.min.y + 1;

            // Get channel info from the header.  Missing channels will be filled with zeros
            // by the FrameBuffer/Slice logic below.
            const ChannelList& channels = m_inputFile->header().channels();

            const Channel* R = channels.findChannel( m_firstChannelName );
            const Channel* G = channels.findChannel( "G" );
            const Channel* B = channels.findChannel( "B" );
            const Channel* A = channels.findChannel( "A" );

            if( !R )
            {
                m_firstChannelName = "Y";
                R = channels.findChannel(m_firstChannelName); // Single channel files may name it 'Y' (for luminance)
            }

            DEMAND_ASSERT_MSG( R, "First channel is missing in EXR file" );
            m_pixelType   = R->type;
            m_info.format = pixelTypeToArrayFormat( m_pixelType );

            // CUDA textures don't support float3, so we round up to four channels.
            m_info.numChannels = A ? 4 : ( B ? 4 : ( G ? 2 : 1 ) );

            if( m_tiledInputFile && m_inputFile )
                m_inputFile.reset();

            // Read the base color from the file
            // FIXME: There should be an option to have this available in the metadata, so
            // we don't have to read the level.
            if( m_readBaseColor && ( m_info.numMipLevels > 1 || ( m_info.width == 1 && m_info.height == 1 ) ) )
            {
                char buff[16] = { 0 };

                // Need to use internal read methods here, because we are already holding a lock on the mutex.
                if( m_inputFile )
                    readScanlineData( buff );
                else
                    readActualTile( buff, getBytesPerChannel( m_info.format ) * m_info.numChannels, m_info.numMipLevels - 1, 0, 0 );

                if( m_info.format == CU_AD_FORMAT_HALF )
                {
                    half* h            = reinterpret_cast<half*>( buff );
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

            m_info.isValid = true;
        }

        m_totalReadTime += stopwatch.elapsed();
    }

    if( info != nullptr )
        *info = m_info;
}

// Do the setup work for a FrameBuffer, putting in the slices as needed based on the channelDesc
void EXRReader::setupFrameBuffer( Imf::FrameBuffer& frameBuffer, char* base, size_t xStride, size_t yStride )
{
    const unsigned int channelSize = getBytesPerChannel( m_info.format );
    frameBuffer.insert( m_firstChannelName, Slice( m_pixelType, base, xStride, yStride ) );
    if( m_info.numChannels > 1 )
    {
        frameBuffer.insert( "G", Slice( m_pixelType, &base[1 * channelSize], xStride, yStride ) );
    }
    if( m_info.numChannels > 2 )
    {
        // CUDA textures don't support float3, so we round up to four channels.
        frameBuffer.insert( "B", Slice( m_pixelType, &base[2 * channelSize], xStride, yStride ) );
        frameBuffer.insert( "A", Slice( m_pixelType, &base[3 * channelSize], xStride, yStride ) );
    }
}

// Close the image.
void EXRReader::close()
{
    if( m_inputFile )
        m_inputFile.reset();
    
    if( m_tiledInputFile )
        m_tiledInputFile.reset();
}

void EXRReader::readActualTile( char* dest, unsigned int rowPitch, unsigned int mipLevel, unsigned int tileX, unsigned int tileY )
{
    DEMAND_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );
    DEMAND_ASSERT( !m_inputFile );

    const Box2i dw = m_tiledInputFile->dataWindowForTile( tileX, tileY, mipLevel );

    // Compute base pointer and strides for frame buffer
    const unsigned int bytesPerPixel = getBytesPerChannel( m_info.format ) * m_info.numChannels;
    const size_t       xStride       = bytesPerPixel;
    const size_t       yStride       = rowPitch;
    char*              base          = dest - ( ( dw.min.x + dw.min.y * rowPitch / bytesPerPixel ) * bytesPerPixel );

    // Create frame buffer.
    FrameBuffer frameBuffer;
    setupFrameBuffer( frameBuffer, base, xStride, yStride );

    m_tiledInputFile->setFrameBuffer( frameBuffer );
    m_tiledInputFile->readTile( tileX, tileY, mipLevel );
}

void EXRReader::readScanlineData( char* dest )
{
    DEMAND_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );
    DEMAND_ASSERT( !m_tiledInputFile );

    const Box2i dw = m_inputFile->header().dataWindow();

    // Compute base pointer and strides for frame buffer
    const unsigned int bytesPerPixel = getBytesPerChannel( m_info.format ) * m_info.numChannels;
    const size_t       xStride       = bytesPerPixel;
    const size_t       yStride       = xStride * m_info.width;
    char*              base          = dest - ( ( dw.min.x + dw.min.y * yStride / bytesPerPixel ) * bytesPerPixel );

    // Create frame buffer.
    FrameBuffer frameBuffer;
    setupFrameBuffer( frameBuffer, base, xStride, yStride );

    m_inputFile->setFrameBuffer( frameBuffer );
    m_inputFile->readPixels( dw.min.y, dw.max.y );
}

void EXRReader::readTile( char* dest, unsigned int mipLevel, unsigned int tileX, unsigned int tileY, unsigned int tileWidth, unsigned int tileHeight, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );
    DEMAND_ASSERT_MSG( !m_inputFile, "Attempting to read tiled data from scanline image." );

    // Stats tracking
    Stopwatch stopwatch;

    // We require that the requested tile size is an integer multiple of the EXR tile size.
    const unsigned int actualTileWidth  = m_tiledInputFile->tileXSize();
    const unsigned int actualTileHeight = m_tiledInputFile->tileYSize();
    if( !( actualTileWidth <= tileWidth && tileWidth % actualTileWidth == 0 )
        || !( actualTileHeight <= tileHeight && tileHeight % actualTileHeight == 0 ) )
    {
        std::stringstream str;
        str << "Unsupported EXR tile size (" << actualTileWidth << "x" << actualTileHeight << ").  Expected "
            << tileWidth << "x" << tileHeight << " (or a whole fraction thereof) for this pixel format";
        throw Exception( str.str().c_str() );
    }
                
    const unsigned int actualTileX    = tileX * tileWidth / actualTileWidth;
    const unsigned int actualTileY    = tileY * tileHeight / actualTileHeight;
    unsigned int       numTilesX      = tileWidth / actualTileWidth;
    unsigned int       numTilesY      = tileHeight / actualTileHeight;
    const unsigned int bytesPerPixel  = getBytesPerChannel( m_info.format ) * m_info.numChannels;
    const unsigned int rowPitch       = tileWidth * bytesPerPixel;
    const size_t       actualTileSize = actualTileWidth * actualTileHeight * bytesPerPixel;

    // Don't request non-existent tiles on the edge of the texture
    unsigned int levelWidthInActualTiles  = ( m_tiledInputFile->levelWidth( mipLevel ) + actualTileWidth - 1 ) / actualTileWidth;
    unsigned int levelHeightInActualTiles = ( m_tiledInputFile->levelHeight( mipLevel ) + actualTileHeight - 1 ) / actualTileHeight;
    numTilesX                             = std::min( numTilesX, levelWidthInActualTiles - actualTileX );
    numTilesY                             = std::min( numTilesY, levelHeightInActualTiles - actualTileY );
                
    for( unsigned int j = 0; j < numTilesY; ++j )
    {
        for( unsigned int i = 0; i < numTilesX; ++i )
        {
            char* start = dest + j * numTilesX * actualTileSize + i * actualTileWidth * bytesPerPixel;
            readActualTile( start, rowPitch, mipLevel, actualTileX + i, actualTileY + j );

            // Stats tracking
            m_numBytesRead += actualTileSize;
        }
    }

    // Stats tracking
    m_numTilesRead += 1;
    m_totalReadTime += stopwatch.elapsed();
}

void EXRReader::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_ASSERT_MSG( isOpen(), "Attempting to read from image that isn't open." );

    // Stats tracking
    Stopwatch stopwatch;
    const Box2i dw = m_tiledInputFile ? m_tiledInputFile->dataWindowForLevel( mipLevel, mipLevel ) : m_inputFile->header().dataWindow();

    if( m_inputFile )
    {
        // Get window offset and dimensions.
        const int width  = dw.max.x - dw.min.x + 1;
        const int height = dw.max.y - dw.min.y + 1;
        DEMAND_ASSERT( width  == static_cast<int>( expectedWidth ) );
        DEMAND_ASSERT( height == static_cast<int>( expectedHeight ) );
        DEMAND_ASSERT( mipLevel == 0 );

        readScanlineData( dest );
                
        // Stats tracking
        {
            m_numTilesRead  += 1;
            m_numBytesRead  += width * height * getBytesPerChannel(m_info.format) * m_info.numChannels;
            m_totalReadTime += stopwatch.elapsed();
        }
    }
    else
    {
        // Get miplevel data window offset and dimensions.
        const int width = dw.max.x - dw.min.x + 1;
        DEMAND_ASSERT( width == static_cast<int>( expectedWidth ) );
        DEMAND_ASSERT( ( dw.max.y - dw.min.y + 1 ) == static_cast<int>( expectedHeight ) );

        // Compute base pointer and strides for frame buffer
        const unsigned int bytesPerPixel = getBytesPerChannel( m_info.format ) * m_info.numChannels;
        const size_t       xStride       = bytesPerPixel;
        const size_t       yStride       = width * xStride;
        char*              base          = dest - ( ( dw.min.x + dw.min.y * width ) * bytesPerPixel );

        // Create frame buffer and read the tiles for the specified mipLevel.
        FrameBuffer frameBuffer;
        setupFrameBuffer( frameBuffer, base, xStride, yStride );
        m_tiledInputFile->setFrameBuffer( frameBuffer );
        m_tiledInputFile->readTiles( 0, m_tiledInputFile->numXTiles( mipLevel ) - 1, 0, m_tiledInputFile->numYTiles( mipLevel ) - 1, mipLevel, mipLevel );

        // Stats tracking
        {
            const unsigned int actualTileWidth  = m_tiledInputFile->tileXSize();
            const unsigned int actualTileHeight = m_tiledInputFile->tileYSize();
            const size_t       actualTileSize   = actualTileWidth * actualTileHeight * bytesPerPixel;
            const int          numXTiles        = m_tiledInputFile->numXTiles( mipLevel );
            const int          numYTiles        = m_tiledInputFile->numYTiles( mipLevel );

            m_numTilesRead += numXTiles * numYTiles;
            m_numBytesRead += numXTiles * numYTiles * actualTileSize;
            m_totalReadTime += stopwatch.elapsed();
        }
    }
}

bool EXRReader::readBaseColor( float4& dest )
{
    dest = m_baseColor;
    return m_baseColorWasRead;
}

void EXRReader::serialize( std::ostream& stream ) const
{
    // Serialize the filename, preceded by its length.
    size_t size = m_filename.size();
    stream.write( reinterpret_cast<const char*>( &size ), sizeof( size_t ) );
    stream.write( m_filename.data(), size );

    // Serialize other constructor parameters.
    stream.write( reinterpret_cast<const char*>( &m_readBaseColor ), sizeof( bool ) );
}

std::shared_ptr<ImageSource> EXRReader::deserialize( std::istream& stream )
{
    // Deserialize filename, which is preceded by its length.
    size_t length;
    stream.read( reinterpret_cast<char*>( &length ), sizeof( size_t ) );

    std::vector<char> buffer( length );
    stream.read( buffer.data(), length );
    std::string filename( buffer.data(), length );

    // Deserialize other constructor parameters.
    bool readBaseColor;
    stream.read( reinterpret_cast<char*>( &readBaseColor ), sizeof( bool ) );

    // Construct the EXRReader.
    return std::shared_ptr<ImageSource>( new EXRReader( filename.c_str(), readBaseColor ) );
}


}  // namespace imageSource
