// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/DDSImageReader.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <vector_functions.h> // from CUDA toolkit

#include "Stopwatch.h"

namespace imageSource {

DDSImageReader::DDSImageReader( const std::string& fileName, bool readBaseColor )
    : m_fileName( fileName )
    , m_readBaseColor( readBaseColor )
{
}

void DDSImageReader::open( TextureInfo* info )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    if( isOpen() )
    {
        if( info != nullptr )
            *info = m_info;
        return;
    }

    m_info.isValid = false;
    m_file.open( m_fileName, std::ios::binary );
    if( !m_file.is_open() ) 
        return;
    m_file.unsetf( std::ios::skipws );

    // Read standard dds file header
    m_ddsFileHeader = DDSFileHeader{};
    m_file.read( reinterpret_cast<char*>( &m_ddsFileHeader ), sizeof( DDSFileHeader ) );
    m_fileHeaderOffset = static_cast<int>( sizeof( DDSFileHeader ) );
    if( m_ddsFileHeader.magicNumber != DDS_MAGIC_NUMBER )
        return;

    // Read the file header extension if the fourCC code is "DX10"
    m_ddsHeaderExtension = DDSHeaderExtension{};
    char* fourCC = m_ddsFileHeader.pixelFormat.fourCCcode;
    if( fourCC[0] == 'D' && fourCC[1] == 'X' && fourCC[2] == '1' && fourCC[3] == '0' )
    {
        m_fileHeaderOffset += static_cast<int>( sizeof( DDSHeaderExtension ) );
        m_file.read( reinterpret_cast<char*>( &m_ddsHeaderExtension ), sizeof( DDSHeaderExtension ) );
    }
    int dxgiFormat = m_ddsHeaderExtension.dxgiFormat;
    m_fileIsTiled = ( m_ddsFileHeader.flags & MISC_TILED_PIXEL_LAYOUT ) != 0;

    // Get the translation to convert from dds file format to cuda texture format
    DDSFormatTranslation formatTranslation = DDSFormatTranslation{};
    size_t numTranslations = sizeof( DDS_FORMAT_TRANSLATIONS ) / sizeof( DDSFormatTranslation );
    for( unsigned int tnum = 0; tnum < numTranslations; ++tnum )
    {
        int tDxgiFormat = DDS_FORMAT_TRANSLATIONS[tnum].dgxiFormat;
        const char* tFourCC = DDS_FORMAT_TRANSLATIONS[tnum].fourCCcode;
        if( tFourCC[0] == fourCC[0] && tFourCC[1] == fourCC[1] && tFourCC[2] == fourCC[2] && tFourCC[3] == fourCC[3] 
            && tDxgiFormat == dxgiFormat )
        {
            formatTranslation = DDS_FORMAT_TRANSLATIONS[tnum];
            break;
        }
    }
    if( formatTranslation.fourCCcode == nullptr )
        return;

    // Fill in the m_info struct
    m_blockSizeInBytes = formatTranslation.blockSize;
    m_info.width = m_ddsFileHeader.width;
    m_info.height = m_ddsFileHeader.height;
    m_info.format = formatTranslation.format;
    m_info.numChannels = formatTranslation.numChannels;
    m_info.numMipLevels = m_ddsFileHeader.mipMapCount;
    m_info.isValid = true;

    // m_info.isTiled=true means that the image reader can return tiles if requested.
    // m_fileIsTiled=true means that the image file stores the image in a tiled format on disk.
    m_info.isTiled = true;

    // Limit mip levels to those that are muliples of 4 in size
    for( uint32_t mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        if( ( m_info.width >> mipLevel ) % 4 != 0 || ( m_info.height >> mipLevel ) % 4 != 0 )
        {
            m_info.numMipLevels = mipLevel;
            if( mipLevel == 0 )
            {
                m_info.isValid = false;
                return;
            }
            break;
        }
    }

    // Resize mip level cache if not a tiled file
    if( !m_fileIsTiled )
    {
        std::unique_lock<std::mutex> lk( m_mipCacheMutex );
        m_mipCache.resize( m_info.numMipLevels );
    }

    if( info != nullptr )
        *info = m_info;
}

bool DDSImageReader::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream /*stream*/  )
{
    open( nullptr );
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read tile from non-existent mip-level." );

    if( m_fileIsTiled )
        return readTileTiled( dest, mipLevel, tile );
    else
        return readTileFlat( dest, mipLevel, tile );
}

bool DDSImageReader::readMipLevel( char* dest, unsigned int mipLevel, unsigned int /*width*/, unsigned int /*height*/, CUstream /*stream*/ )
{
    open( nullptr );
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read non-existent mip-level." );

    if( m_fileIsTiled )
        return readMipLevelTiled( dest, mipLevel );
    else
        return readMipLevelFlat( dest, mipLevel );
}

bool DDSImageReader::readBaseColor( float4& /*dest*/ )
{
    // FIXME: Actually read the base color. Reading the base color is difficult because
    // it is stored in a compressed block.
    return false;
}

bool DDSImageReader::readMipTail( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, 
                                  const uint2* mipLevelDims, CUstream stream )
{
    open( nullptr );
    if( m_fileIsTiled && mipTailFirstLevel == (unsigned int)getMipTailStartLevel() )
        return readMipTailTiled( dest, mipTailFirstLevel );
    else
        return ImageSourceBase::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
}

int DDSImageReader::getMipLevelWidthInBytes( int mipLevel )
{
    open( nullptr );
    int mipWidthInBlocks = ( m_info.width / BC_BLOCK_WIDTH ) >> mipLevel;
    return mipWidthInBlocks * m_blockSizeInBytes;
}

int DDSImageReader::getMipLevelSizeInBytes( int mipLevel )
{
    open( nullptr );

    if( m_fileIsTiled )
        return getMipLevelSizeInBytesTiled( mipLevel );
    else
        return getMipLevelSizeInBytesFlat( mipLevel );
}

bool DDSImageReader::saveAsTiledFile( const char* fileName )
{
    open( nullptr );

    // Open the file
    std::ofstream ofile;
    ofile.open( fileName, std::ios::binary );
    if( !ofile.is_open() )
        return false;
    ofile.unsetf( std::ios::skipws );

    // Save Header
    DDSFileHeader ddsFileHeader = m_ddsFileHeader;
    ddsFileHeader.flags = ddsFileHeader.flags | MISC_TILED_PIXEL_LAYOUT;
    ofile.write( (char*)&ddsFileHeader, sizeof(DDSFileHeader) );

    // Save extended header if needed
    char* fourCC = m_ddsFileHeader.pixelFormat.fourCCcode;
    if( fourCC[0] == 'D' && fourCC[1] == 'X' && fourCC[2] == '1' && fourCC[3] == '0' )
    {
        DDSHeaderExtension ddsHeaderExtension = m_ddsHeaderExtension;
        ddsHeaderExtension.miscFlag = ddsHeaderExtension.miscFlag | D3D11_RESOURCE_MISC_TILED;
        ofile.write( (char*)&ddsHeaderExtension, sizeof(DDSHeaderExtension) );
    }

    // Save Tiles
    std::vector<char> tileBuff( TILE_SIZE_IN_BYTES );
    int mipTailFirstLevel = getMipTailStartLevel();

    for( int mipLevel = 0; mipLevel < static_cast<int>( m_info.numMipLevels ); ++mipLevel )
    {
        if( mipLevel >= mipTailFirstLevel )
            break;
        unsigned int mipLevelWidthInTiles = getMipLevelWidthInTiles( mipLevel );
        unsigned int mipLevelHeightInTiles = getMipLevelHeightInTiles( mipLevel );

        for( unsigned int tileY = 0; tileY < mipLevelHeightInTiles; ++tileY )
        {
            for( unsigned int tileX = 0; tileX < mipLevelWidthInTiles; ++tileX )
            {
                Tile tile{tileX, tileY, getTileWidth(), getTileHeight()};
                if ( !readTile( tileBuff.data(), mipLevel, tile, CUstream{0} ) )
                    return false;
                ofile.write( tileBuff.data(), TILE_SIZE_IN_BYTES );
            }
        }
    }

    // Save mip tail
    if( mipTailFirstLevel < static_cast<int>( m_info.numMipLevels ) )
    {
        tileBuff.clear();
        tileBuff.resize( getMipTailSize(), 0 );
        std::vector<uint2> mipLevelDims;
        for( unsigned int mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
        {
            mipLevelDims.push_back( uint2{ m_info.width >> mipLevel, m_info.height >> mipLevel } );
        }

        if( !readMipTail( tileBuff.data(), mipTailFirstLevel, m_info.numMipLevels, mipLevelDims.data(), CUstream{0} ) )
            return false;

        ofile.write( tileBuff.data(), tileBuff.size() );
    }

    // Close file
    ofile.close();
    return true;
}

int DDSImageReader::getMipTailStartLevel()
{
    unsigned int tileWidth = getTileWidth();
    unsigned int tileHeight = getTileHeight();

    for( unsigned int mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        if( ( m_info.width >> mipLevel ) < tileWidth || ( m_info.height >> mipLevel ) < tileHeight )
            return mipLevel;
    }
    return m_info.numMipLevels;
}

int DDSImageReader::getMipTailSize()
{
    int startLevel = getMipTailStartLevel();
    int tWidth = m_info.width >> startLevel;
    int tHeight = m_info.height >> startLevel;

    int mipTailSize = ( ( tWidth / BC_BLOCK_WIDTH ) * ( tHeight / BC_BLOCK_HEIGHT ) * m_blockSizeInBytes * 4 ) / 3;
    mipTailSize += TILE_SIZE_IN_BYTES - 1;
    mipTailSize -= mipTailSize % TILE_SIZE_IN_BYTES;
    return mipTailSize;
}

//---------------- Flat (non-tiled) reading functions

bool DDSImageReader::readTileFlat( char* dest, unsigned int mipLevel, const Tile& tile )
{
    // Read the mip level and extract the tile
    {
        std::unique_lock<std::mutex> lock( m_mipCacheMutex );
        if( m_mipCache[mipLevel].size() == 0 )
        {
            m_mipCache[mipLevel].resize( getMipLevelSizeInBytes( mipLevel ) );
            if( !readMipLevelFlat( m_mipCache[mipLevel].data(), mipLevel ) )
                return false;
        }
    }

    const char* mipSrc = m_mipCache[mipLevel].data();
    int mipWidthInBlocks = ( m_info.width / BC_BLOCK_WIDTH ) >> mipLevel; 
    int mipHeightInBlocks = ( m_info.height / BC_BLOCK_HEIGHT ) >> mipLevel;
    unsigned int tileWidthInBlocks = tile.width / BC_BLOCK_WIDTH;
    unsigned int tileHeightInBlocks = tile.height / BC_BLOCK_HEIGHT;

    Tile blockTile = Tile{tile.x, tile.y, tileWidthInBlocks, tileHeightInBlocks};
    blockTile.width = std::min( blockTile.width, mipWidthInBlocks - blockTile.x * tileWidthInBlocks );
    blockTile.height = std::min( blockTile.height, mipHeightInBlocks - blockTile.y * tileHeightInBlocks );

    for( uint32_t tileRow = 0; tileRow < blockTile.height; ++tileRow )
    {
        int mipRow = blockTile.y * tileHeightInBlocks + tileRow;
        int mipSourceOffset = ( mipRow * mipWidthInBlocks + blockTile.x * tileWidthInBlocks ) * m_blockSizeInBytes;
        int tileDestOffset = ( tileRow * tileWidthInBlocks ) * m_blockSizeInBytes;
        memcpy( dest + tileDestOffset, mipSrc + mipSourceOffset, m_blockSizeInBytes * blockTile.width );
    }

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock( m_statsMutex );
        m_numTilesRead += 1;
    }
    
    return true;
}

bool DDSImageReader::readMipLevelFlat( char* dest, unsigned int mipLevel )
{
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );
    std::unique_lock<std::mutex> lock( m_mutex );
    Stopwatch stopwatch;

    m_file.seekg( getMipLevelOffsetInBytesFlat( mipLevel ) );
    m_file.read( reinterpret_cast<char*>( dest ), getMipLevelSizeInBytesFlat( mipLevel ) );

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock( m_statsMutex );
        m_numBytesRead += getMipLevelSizeInBytesFlat( mipLevel );
        m_totalReadTime += stopwatch.elapsed();
    }

    return true;
}

int DDSImageReader::getMipLevelOffsetInBytesFlat( int mipLevel )
{
    int offset = m_fileHeaderOffset;
    for( int i = 0; i < mipLevel; ++i )
        offset += getMipLevelSizeInBytesFlat( i );
    return offset;
}

int DDSImageReader::getMipLevelSizeInBytesFlat( int mipLevel )
{
    int widthInBlocks = std::max( 1U, ( m_info.width / BC_BLOCK_WIDTH ) >> mipLevel );
    int heightInBlocks = std::max( 1U, ( m_info.height / BC_BLOCK_HEIGHT ) >> mipLevel );
    return widthInBlocks * heightInBlocks * m_blockSizeInBytes;
}

//---------------- Tiled reading functions

bool DDSImageReader::readTileTiled( char* dest, unsigned int mipLevel, const Tile& tile )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    Stopwatch stopwatch;
    m_file.seekg( getTileOffsetInBytesTiled( mipLevel, tile ) );
    m_file.read( dest, TILE_SIZE_IN_BYTES );

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock( m_statsMutex );
        m_numTilesRead++;
        m_numBytesRead += TILE_SIZE_IN_BYTES;
        m_totalReadTime += stopwatch.elapsed();
    }

    return true;
}

bool DDSImageReader::readMipLevelTiled( char* dest, unsigned int mipLevel )
{
    //std::unique_lock<std::mutex> lock( m_mutex );
    std::vector<char> tileBuff;

    // If the mip level is part of the mip tail, load the mip tail and
    // copy the mip level directly to dest.
    unsigned int mipTailFirstLevel = getMipTailStartLevel();
    if( mipLevel >= mipTailFirstLevel )
    {
        tileBuff.resize( getMipTailSize(), 0 );
        if( !readMipTailTiled( tileBuff.data(), mipTailFirstLevel ) )
            return false;
        int offset = 0;
        for( unsigned int tailLevel = mipTailFirstLevel; tailLevel < mipLevel; ++tailLevel )
        {
            offset += getMipLevelSizeInBytesFlat( tailLevel );
        }
        memcpy( dest, &tileBuff[offset], getMipLevelSizeInBytesFlat( mipLevel ) );
        return true;
    }

    // Read each tile and copy the tile lines into the dest buffer.
    // Make sure to account for partial tiles lines on the right edge,
    // and end of lines on the top.
    tileBuff.resize( TILE_SIZE_IN_BYTES, 0 );
    unsigned int mipLevelWidthInTiles = getMipLevelWidthInTiles( mipLevel );
    unsigned int mipLevelHeightInTiles = getMipLevelHeightInTiles( mipLevel );
    for( unsigned int tileY = 0; tileY < mipLevelHeightInTiles; ++tileY )
    {
        for( unsigned int tileX = 0; tileX < mipLevelWidthInTiles; ++tileX )
        {
            Tile tile{tileX, tileY, getTileWidth(), getTileHeight()};
            if( !readTileTiled( tileBuff.data(), mipLevel, tile ) )
                return false;

            // Get the height of the tile in lines, including partial tiles on right edge of image.
            int mipLevelHeight = m_info.height >> mipLevel;
            int standardTileHeight = getTileHeight();
            int tileHeight = std::min( (unsigned int)standardTileHeight, mipLevelHeight - standardTileHeight * tileY );
            int tileHeightInLines = tileHeight / BC_BLOCK_HEIGHT;

            // Get the width of the tile in bytes, including partial tiles on the top of the image.
            int mipLevelWidth = m_info.width >> mipLevel;
            int standardTileWidth = getTileWidth();
            int tileWidth = std::min( (unsigned int)standardTileWidth, mipLevelWidth - standardTileWidth * tileX );
            int tileWidthInBytes = ( tileWidth / BC_BLOCK_WIDTH ) * m_blockSizeInBytes;
            int standardTileWidthInBytes = ( standardTileWidth / BC_BLOCK_HEIGHT ) * m_blockSizeInBytes;
            int mipLineWidthInBytes = ( mipLevelWidth / BC_BLOCK_WIDTH ) * m_blockSizeInBytes;

            // Copy each line from the tile buffer to the mip level buffer.
            for( int y=0; y<tileHeightInLines; ++y )
            {
                char* lineSrc = &tileBuff[y * standardTileWidthInBytes];

                int my = tileY * standardTileHeight + y * BC_BLOCK_HEIGHT;
                int mx = tileX * standardTileWidth;
                int mipOffset = ( my / BC_BLOCK_HEIGHT ) * mipLineWidthInBytes + ( mx / BC_BLOCK_WIDTH ) * m_blockSizeInBytes;
                char* lineDest = &dest[mipOffset]; // FIXME

                memcpy( lineDest, lineSrc, tileWidthInBytes );
            }
        }
    }
    return true;
}

bool DDSImageReader::readMipTailTiled( char* dest, unsigned int mipTailFirstLevel )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    Stopwatch stopwatch;
    OTK_ASSERT_MSG( mipTailFirstLevel == static_cast<unsigned int>(getMipTailStartLevel()), "Improper mip tail first level for tiled file." );
    m_file.seekg( getMipLevelOffsetInBytesTiled( mipTailFirstLevel ) );
    m_file.read( dest, getMipTailSize() );

    // Stats tracking
    {
        std::unique_lock<std::mutex> lock( m_statsMutex );
//        m_numTilesRead++;
        m_numBytesRead += getMipTailSize();
        m_totalReadTime += stopwatch.elapsed();
    }

    return true;
}

int DDSImageReader::getMipLevelOffsetInBytesTiled( int mipLevel )
{
    int offset = m_fileHeaderOffset;
    for( int i = 0; i < mipLevel; ++i )
        offset += getMipLevelSizeInBytesTiled( i );
    return offset;
}

int DDSImageReader::getMipLevelSizeInBytesTiled( int mipLevel )
{
    return getMipLevelWidthInTiles( mipLevel ) * getMipLevelHeightInTiles( mipLevel ) * TILE_SIZE_IN_BYTES;
}

int DDSImageReader::getTileOffsetInBytesTiled( int mipLevel, const Tile& tile )
{
    int mipOffset = getMipLevelOffsetInBytesTiled( mipLevel );
    int tileOffset = ( tile.y * getMipLevelWidthInTiles( mipLevel ) + tile.x ) * TILE_SIZE_IN_BYTES;
    return mipOffset + tileOffset;
}

}  // namespace imageSource
