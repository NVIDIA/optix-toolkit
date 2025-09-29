// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>

#include <OptiXToolkit/ImageSource/CompressedTextureCacheManager.h>

#include <half.h>
#include <openexr.h>

namespace imageSource {

std::string CompressedTextureCacheManager::getCacheFilePath( const std::string& inputFile )
{
#ifdef _WIN32
    const char separator = '\\';
#else
    const char separator = '/';
#endif

    std::string baseFileName = inputFile.substr( inputFile.find_last_of( "/\\") + 1 );
    return m_options.cacheFolder + separator + baseFileName + ".dds";
}

bool CompressedTextureCacheManager::isSupportedFileType( const std::string& extension )
{
    std::string ext = extension;
    std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );
    for( int i = 0; i < NUM_SUPPORTED_FILE_TYPES; ++i )
    {
        if( ext == SUPPORTED_FILE_TYPES[i] )
            return true;
    }
    return false;
}

bool CompressedTextureCacheManager::cacheFile( const std::string& inputFilePath )
{
    if( inputFilePath.length() == 0 )
        return false;
    std::string suffix = inputFilePath.substr( inputFilePath.rfind('.') + 1 );
    std::transform( suffix.begin(), suffix.end(), suffix.begin(), ::tolower );
    std::string cacheFilePath = getCacheFilePath( inputFilePath );
    printCommand( "\n=========== Converting file: " + inputFilePath );

    // Return if the cache file already exists, or the input  file does not
    {
        std::ifstream cacheFile( cacheFilePath );
        if( cacheFile.good() )
        {
            return true;
        }
        std::ifstream inputFile( inputFilePath );
        if( !inputFile.good() )
            return false;
    }

    // If DDS image, convert directly to tiled
    if( suffix == "dds" )
    {
        return convertDDSToTiledDDS( inputFilePath, cacheFilePath );
    }

    // If not EXR, use the default format
    if( suffix != "exr" )
    {
        std::string tempDdsFilePath = cacheFilePath + ".tmp.dds";
        std::string ddsFormat = m_options.exrToDdsFormats[DDS_DEFAULT_INDEX];
        if( !convertImageToDDS( inputFilePath, tempDdsFilePath, ddsFormat ) )
            return false;
        if( !convertDDSToTiledDDS( tempDdsFilePath, cacheFilePath ) )
            return false;
        return deleteFile( tempDdsFilePath );
    }

    // If it's EXR:
    {
        CoreEXRReader exrReader( inputFilePath, false );
        TextureInfo texInfo{};
        exrReader.open( &texInfo );
        if( !exrReader.isOpen() )
            return false;
        bool ishdr = isHighDynamicRange( exrReader, texInfo );
        int ddsFormatIndex = ishdr ? DDS_HDR_INDEX : texInfo.numChannels;
        std::string ddsFormat = m_options.exrToDdsFormats[ddsFormatIndex];

        std::string intermediateFilePath = "";
        if( ishdr ) 
            intermediateFilePath = cacheFilePath + ".tmp.hdr";
        else if( texInfo.numChannels == 4 )
            intermediateFilePath = cacheFilePath + ".tmp.tga";
        else if( texInfo.numChannels >= 2 )
            intermediateFilePath = cacheFilePath + ".tmp.hdr"; // ".tmp.ppm";
        else // numChannels == 1
            intermediateFilePath = cacheFilePath + ".tmp.hdr"; // ".tmp.pgm";

        std::string tempDdsFilePath = cacheFilePath + ".tmp.dds";
        if( !convertImageToIntermediateFormat( inputFilePath, intermediateFilePath, texInfo.width, texInfo.height ) )
            return false;
        if( !convertImageToDDS( intermediateFilePath, tempDdsFilePath, ddsFormat ) )
            return false;
        if( !convertDDSToTiledDDS( tempDdsFilePath, cacheFilePath ) )
            return false;
        if( !deleteFile( intermediateFilePath ) )
            return false;
        return deleteFile( tempDdsFilePath );
    }
}

void CompressedTextureCacheManager::printCommand( const std::string& command )
{
    if( m_verbose )
        printf( "%s\n", command.c_str() );
}

bool CompressedTextureCacheManager::isHighDynamicRange( CoreEXRReader& exrReader, TextureInfo& texInfo )
{
printf("NUM EXR CHANNELS: %d\n", exrReader.getNumExrChannels() );
return true;
    // Assuming images with alpha are not hdr. (Note: BC6 can't handle alpha)
    if( exrReader.getNumExrChannels() > 3 )
        return false;
return true;

    // Just assume hdr with only one mip level to prevent very large read
    if( texInfo.numMipLevels <= 1 && ( texInfo.width * texInfo.height > (1024 * 1024) ) )
        return true;

    // Determine mip level to sample to figure out hdr
    unsigned int mipLevel = 0; 
    unsigned int levelWidth = texInfo.width; 
    unsigned int levelHeight = texInfo.height;
    for( mipLevel = 0; mipLevel < texInfo.numMipLevels-1; ++mipLevel )
    {
        levelWidth = texInfo.width >> mipLevel;
        levelHeight = texInfo.height >> mipLevel;
        if( levelWidth < m_options.hdrCheckSize * 2 || levelHeight < m_options.hdrCheckSize * 2 )
            break;
    }

    // Check pixel values to see if they are in [0,1] range
    const float HDR_THRESHOLD = 1.0f;
    unsigned int levelChannels = levelWidth * levelHeight * texInfo.numChannels;
    unsigned int levelSizeInBytes = ( levelWidth * levelHeight * getBitsPerPixel( texInfo ) ) / BITS_PER_BYTE;

    std::vector<char> buff( levelSizeInBytes * 2 );
    if( !exrReader.readMipLevel( buff.data(), mipLevel, levelWidth, levelHeight, CUstream{0} ) )
        return true;

    if( texInfo.format == CU_AD_FORMAT_HALF )
    {
        half* p = reinterpret_cast<half*>( buff.data() );
        for( unsigned int i = 0; i < levelChannels; ++i )
        {
            if( float( p[i] ) >  HDR_THRESHOLD || float( p[i] ) < 0.0f )
                return true;
        }
        return false;
    }
    else if( texInfo.format == CU_AD_FORMAT_FLOAT )
    {
        float* p = reinterpret_cast<float*>( buff.data() );
        for( unsigned int i = 0; i < levelChannels; ++i )
        {
            if( p[i] >  HDR_THRESHOLD || p[i] < 0.0f )
                return true;
        }
        return false;
    }
    return true;
}

bool CompressedTextureCacheManager::convertImageToIntermediateFormat( const std::string& inFile, const std::string& outFile, 
    unsigned int width, unsigned int height )
{
    const std::string resize[5] = {"", "-scale 50%", "-scale 25%", "-scale 12.5%", "-scale 6.25%"};

    unsigned int droppedLevels = 0;
    for( droppedLevels = 0; droppedLevels < m_options.droppedMipLevels; ++droppedLevels )
    {
        if( (width >> droppedLevels) < m_options.minImageSize || (height >> droppedLevels) < m_options.minImageSize )
            break;
    }
    droppedLevels = std::min( droppedLevels, 4u );

    std::string command = m_options.imageMagick + " \"" + inFile + "\" " + resize[droppedLevels] 
        + " \"" + outFile + "\"";
    printCommand( command );
    return ( std::system( command.c_str() ) == 0 );
}

bool CompressedTextureCacheManager::convertImageToDDS( const std::string& inFile, const std::string& outFile, const std::string& ddsFormat )
{
    std::string command = m_options.nvcompress + " -" + ddsFormat + " -silent" 
        + " \"" + inFile + "\"" + " \"" + outFile + "\"";
    printCommand( command );
    return ( std::system( command.c_str() ) == 0 );
}

bool CompressedTextureCacheManager::convertDDSToTiledDDS( const std::string& inFile, const std::string& outFile )
{
    std::string command = "Convert DDS File To Tiled DDS: " + inFile + " ==> " + outFile;
    printCommand( command );
    DDSImageReader reader( inFile, false );
    return reader.saveAsTiledFile( outFile.c_str() );
}

bool CompressedTextureCacheManager::deleteFile( const std::string &fileName )
{
#ifdef _WIN32
    const std::string deleteCommand = "del";
#else
    const std::string deleteCommand = "rm";
#endif

    std::string command = deleteCommand + " \"" + fileName + "\"";
    printCommand( command );
    return ( std::system( command.c_str() ) == 0 );
}

}  // namespace imageSource
