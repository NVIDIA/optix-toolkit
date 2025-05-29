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

namespace imageSource {

DDSImageReader::DDSImageReader( const std::string& fileName, bool readBaseColor )
    : m_fileName( fileName )
    , m_readBaseColor( readBaseColor )
{
}

void DDSImageReader::open( TextureInfo* info )
{
    std::unique_lock<std::mutex> lock(m_mutex);
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
    DDSFileHeader fileHeader = DDSFileHeader{};
    m_file.read( reinterpret_cast<char*>( &fileHeader ), sizeof( fileHeader ) );
    m_fileHeaderOffset = static_cast<int>( sizeof( fileHeader ) );
    if( fileHeader.magicNumber != DDS_MAGIC_NUMBER )
        return;

    // Read the file header extension if the fourCC code is "DX10"
    char* fourCC = fileHeader.pixelFormat.fourCCcode;
    DDSHeaderExtension headerExtension = DDSHeaderExtension{};
    if( fourCC[0] == 'D' && fourCC[1] == 'X' && fourCC[2] == '1' && fourCC[3] == '0' )
    {
        m_fileHeaderOffset += static_cast<int>( sizeof( DDSHeaderExtension ) );
        m_file.read( reinterpret_cast<char*>( &headerExtension ), sizeof( DDSHeaderExtension ) );
    }
    int dxgiFormat = headerExtension.dxgiFormat;

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
    m_info.width = fileHeader.width;
    m_info.height = fileHeader.height;
    m_info.format = formatTranslation.format;
    m_info.numChannels = formatTranslation.numChannels;
    m_info.numMipLevels = fileHeader.mipMapCount;
    m_info.isValid = true;
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

    // Resize mip level cache
    {
        std::unique_lock<std::mutex> lock( m_mipCacheMutex );
        m_mipCache.resize( m_info.numMipLevels );
    }

    if( info != nullptr )
        *info = m_info;
}

bool DDSImageReader::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream /*stream*/  )
{
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );
    {
        std::unique_lock<std::mutex> lock( m_mipCacheMutex );
        if( m_mipCache[mipLevel].size() == 0 )
        {
            m_mipCache[mipLevel].resize( getMipLevelSizeInBytes( mipLevel ) );
            readMipLevel( m_mipCache[mipLevel].data(), mipLevel, m_info.width >> mipLevel, m_info.height >> mipLevel, CUstream{0} );
        }
    }

    const char* mipSrc = m_mipCache[mipLevel].data();
    int mipWidthInBlocks = ( m_info.width / BC_BLOCK_WIDTH ) >> mipLevel; 
    int mipHeightInBlocks = ( m_info.height / BC_BLOCK_HEIGHT ) >> mipLevel;
    int tileWidthInBlocks = tile.width / BC_BLOCK_WIDTH;
    int tileHeightInBlocks = tile.height / BC_BLOCK_HEIGHT;

    Tile blockTile = Tile{tile.x, tile.y, tile.width / BC_BLOCK_WIDTH, tile.height / BC_BLOCK_HEIGHT};
    blockTile.width = std::min( blockTile.width, mipWidthInBlocks - blockTile.x * tileWidthInBlocks );
    blockTile.height = std::min( blockTile.height, mipHeightInBlocks - blockTile.y * tileHeightInBlocks );

    for( uint32_t tileRow = 0; tileRow < blockTile.height; ++tileRow )
    {
        int mipRow = blockTile.y * tileHeightInBlocks + tileRow;
        int mipSourceOffset = ( mipRow * mipWidthInBlocks + blockTile.x * tileWidthInBlocks ) * m_blockSizeInBytes;
        int tileDestOffset = ( tileRow * tileWidthInBlocks ) * m_blockSizeInBytes;
        memcpy( dest + tileDestOffset, mipSrc + mipSourceOffset, m_blockSizeInBytes * blockTile.width );
    }
    
    return true;
}

bool DDSImageReader::readMipLevel( char* dest, unsigned int mipLevel, unsigned int /*width*/, unsigned int /*height*/, CUstream /*stream*/ )
{
    open( nullptr );
    OTK_ASSERT_MSG( mipLevel < m_info.numMipLevels, "Attempt to read from non-existent mip-level." );
    std::unique_lock<std::mutex> lock( m_mutex );

    m_file.seekg( getMipLevelFileOffsetInBytes( mipLevel ) );
    m_file.read( reinterpret_cast<char*>( dest ), getMipLevelSizeInBytes( mipLevel ) );
    return true;
}

bool DDSImageReader::readBaseColor( float4& /*dest*/ )
{
    // FIXME: Actually read the base color
    return false;
}

int DDSImageReader::getMipLevelWidthInBytes( int mipLevel )
{
    int widthInBlocks = ( m_info.width / BC_BLOCK_WIDTH ) >> mipLevel;
    return widthInBlocks * m_blockSizeInBytes;
}

int DDSImageReader::getMipLevelSizeInBytes( int mipLevel )
{
    int widthInBlocks = ( m_info.width / BC_BLOCK_WIDTH ) >> mipLevel; 
    int heightInBlocks = ( m_info.height / BC_BLOCK_HEIGHT ) >> mipLevel;
    return widthInBlocks * heightInBlocks * m_blockSizeInBytes;
}

int DDSImageReader::getMipLevelFileOffsetInBytes( int mipLevel )
{
    int offset = m_fileHeaderOffset;
    for( int i = 0; i < mipLevel; ++i )
        offset += getMipLevelSizeInBytes( i );
    return offset;
}

}  // namespace imageSource
