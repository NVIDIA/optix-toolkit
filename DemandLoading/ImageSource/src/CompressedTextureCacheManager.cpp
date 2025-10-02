// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <filesystem>

#include <OptiXToolkit/ImageSource/CompressedTextureCacheManager.h>

#include <half.h>
#include <openexr.h>

namespace fs = std::filesystem;

namespace imageSource {

std::string CompressedTextureCacheManager::getCacheFilePath( const std::string& inputFile )
{
    std::string baseFileName = inputFile.substr( inputFile.find_last_of( "/\\") + 1 );
    return m_options.cacheFolder + FILE_SEPARATOR + baseFileName + ".dds";
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

    // Return if the cache file already exists, or the input file does not exist.
    {
        if( fs::exists( cacheFilePath ) )
        {
            printCommand( "\"" + cacheFilePath + "\" exists, not converting." );
            return true;
        }
        if( !fs::exists( inputFilePath ) )
        {
            printCommand( "Could not find input file \"" + inputFilePath + "\"." );
            return false;
        }
    }

    // If already a DDS image, copy or convert directly to tiled.
    if( suffix == "dds" )
    {
        if( !m_options.saveTiled )
        {
            std::string command = std::string(MOVE_COMMAND) + " " + inputFilePath + " " + cacheFilePath;
            printCommand( command );
            return ( std::system( command.c_str() ) == 0 );
        }
        return convertDDSToTiledDDS( inputFilePath, cacheFilePath );
    }

    // If not EXR, convert directly to dds using the default format.
    if( suffix != "exr" )
    {
        std::string tempDdsFilePath = cacheFilePath + ".tmp.dds";
        std::string ddsFormat = m_options.exrToDdsFormats[DDS_DEFAULT_INDEX];

        if( !m_options.saveTiled )
        {
            return convertImageToDDS( inputFilePath, cacheFilePath, ddsFormat );
        }
        else
        {
            if( !convertImageToDDS( inputFilePath, tempDdsFilePath, ddsFormat ) )
                return false;
            if( !convertDDSToTiledDDS( tempDdsFilePath, cacheFilePath ) )
                return false;
            return deleteFile( tempDdsFilePath );
        }
    }

    // If an EXR image, figure out what format to use and convert.
    {
        CoreEXRReader exrReader( inputFilePath, false );
        TextureInfo texInfo{};
        exrReader.open( &texInfo );
        if( !exrReader.isOpen() )
            return false;

        // Determine dds (bc) format to use
        bool ishdr = isHighDynamicRange( exrReader, texInfo );
        int ddsFormatIndex = ishdr ? DDS_HDR_INDEX : texInfo.numChannels;
        std::string ddsFormat = m_options.exrToDdsFormats[ddsFormatIndex];

        // Convert input file to intermediate format (exr or tga)
        std::string intermediateFilePath = "";
        if( exrReader.getNumExrChannels() < 4 ) // Use HDR for images without alpha
        {
            intermediateFilePath = cacheFilePath + ".tmp.hdr";
            std::string command = "Converting EXR file \"" + inputFilePath + "\" to HDR file \"" 
                + intermediateFilePath + "\"";
            printCommand( command );
            if( !fs::exists( intermediateFilePath ) )
            {
                if( !convertEXRtoHDR( exrReader, texInfo, intermediateFilePath ) )
                    return false;
            }
        }
        else // Use TGA for images with alpha
        {
            intermediateFilePath = cacheFilePath + ".tmp.tga";
            std::string command = "Converting EXR file \"" + inputFilePath + "\" to TGA file \"" 
                + intermediateFilePath + "\"";
            printCommand( command );
            if( !fs::exists( intermediateFilePath ) )
            {
                if( !convertEXRtoTGA4( exrReader, texInfo, 1.0f, 1.0f, intermediateFilePath ) )
                    return false;
            }
        }

        // Convert the intermediate file to dds, and tile it if needed
        if( !m_options.saveTiled )
        {
            if( !convertImageToDDS( intermediateFilePath, cacheFilePath, ddsFormat ) )
                return false;
            return deleteFile( intermediateFilePath );
        }
        else
        {
            std::string tempDdsFilePath = cacheFilePath + ".tmp.dds";
            if( !convertImageToDDS( intermediateFilePath, tempDdsFilePath, ddsFormat ) )
                return false;
            if( !convertDDSToTiledDDS( tempDdsFilePath, cacheFilePath ) )
                return false;
            if( !deleteFile( intermediateFilePath ) )
                return false;
            return deleteFile( tempDdsFilePath );
        }
    }
}

void CompressedTextureCacheManager::printCommand( const std::string& command )
{
    if( m_verbose )
        printf("    %s\n", command.c_str());
}

bool CompressedTextureCacheManager::isHighDynamicRange( CoreEXRReader& exrReader, TextureInfo& texInfo )
{
    // Assume images with alpha are not hdr. (BC6 can't handle alpha)
    if( exrReader.getNumExrChannels() > 3 )
        return false;

    // Determine mip level to sample to figure out hdr
    unsigned int mipLevel = 0; 
    unsigned int levelWidth = texInfo.width; 
    unsigned int levelHeight = texInfo.height;
    for( mipLevel = 0; mipLevel < texInfo.numMipLevels-1; ++mipLevel )
    {
        levelWidth = texInfo.width >> mipLevel;
        levelHeight = texInfo.height >> mipLevel;
        if( levelWidth * levelHeight <= m_options.hdrCheckSize * m_options.hdrCheckSize )
            break;
    }
    // Assume hdr if the image does not have a small enough mip level
    if( levelWidth * levelHeight > m_options.hdrCheckSize * m_options.hdrCheckSize )
        return true;

    // Read mip level
    std::vector<char> buff( levelWidth * levelHeight * ( getBitsPerPixel( texInfo ) / BITS_PER_BYTE ) );
    if( !exrReader.readMipLevel( buff.data(), mipLevel, levelWidth, levelHeight, CUstream{0} ) )
        return true;

    // Check pixel values to see if they are in [0,1] range
    const float HDR_THRESHOLD = 1.0f;
    unsigned int levelChannels = levelWidth * levelHeight * texInfo.numChannels;

    for( unsigned int i = 0; i < levelChannels; ++i )
    {
        if( texInfo.format == CU_AD_FORMAT_HALF )
        {
            const half* hbuff = reinterpret_cast<half*>( buff.data() );
            if( float( hbuff[i] ) >  HDR_THRESHOLD || float( hbuff[i] ) < 0.0f )
                return true;
        }
        else // CU_AD_FORMAT_FLOAT
        {
            const float* fbuff = reinterpret_cast<float*>( buff.data() );
            if( fbuff[i] >  HDR_THRESHOLD || fbuff[i] < 0.0f )
                return true;
        }
    }
    return false;
}


inline float4 getExrPixelColor( char* imageData, int width, TextureInfo& texInfo, int x, int y )
{
    float4 c;
    const int numChannels = texInfo.numChannels;
    if( texInfo.format == CU_AD_FORMAT_HALF )
    {
        half* hbuff = reinterpret_cast<half*>( imageData );
        half* pixel = &hbuff[(y * width  + x) * numChannels];
        c.x = float( pixel[0] );
        c.y = ( numChannels >= 2 ) ? float( pixel[1] ) : 0.0f;
        c.z = ( numChannels >= 3 ) ? float( pixel[2] ) : 0.0f;
        c.w = ( numChannels >= 4 ) ? float( pixel[3] ) : 0.0f;
    }
    else // CU_AD_FORMAT_FLOAT
    {
        float* fbuff = reinterpret_cast<float*>( imageData );
        float* pixel = &fbuff[(y * width  + x) * numChannels];
        c.x = pixel[0];
        c.y = ( numChannels >= 2 ) ? pixel[1] : 0.0f;
        c.z = ( numChannels >= 3 ) ? pixel[2] : 0.0f;
        c.w = ( numChannels >= 4 ) ? pixel[3] : 0.0f;
    }
    return c;
}

inline void float2rgbe(float r, float g, float b, unsigned char rgbe[4]) {
    float v = std::max(std::max(r, g), b);
    
    if (v < 1e-32f) {
        rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
    } else {
        int e;
        v = frexpf(v, &e) * 256.0f / v;
        rgbe[0] = (unsigned char)(r * v);
        rgbe[1] = (unsigned char)(g * v);
        rgbe[2] = (unsigned char)(b * v);
        rgbe[3] = (unsigned char)(e + 128);
    }
}

bool CompressedTextureCacheManager::convertEXRtoHDR( CoreEXRReader& exrReader, TextureInfo& texInfo, 
    const std::string& outFileName )
{
    try {
        // Read mip level from EXR
        int mipLevel = std::min( texInfo.numMipLevels - 1, m_options.droppedMipLevels );
        int width = static_cast<int>( texInfo.width >> mipLevel ); 
        int height = static_cast<int>( texInfo.height >> mipLevel );
        std::vector<char> buff( width * height * ( getBitsPerPixel( texInfo ) / BITS_PER_BYTE ) );
        if( !exrReader.readMipLevel( buff.data(), mipLevel, width, height, CUstream{0} ) )
        {
            std::cerr << "Failed to read EXR data" << std::endl;
            return false;
        }
        
        // Open output HDR file and write header
        std::ofstream outFile( outFileName, std::ios::binary );
        if ( !outFile ) {
            std::cerr << "Failed to open output file: " << outFileName << std::endl;
            return false;
        }
        outFile << "#?RADIANCE\n";
        outFile << "FORMAT=32-bit_rle_rgbe\n\n";
        outFile << "-Y " << height << " +X " << width << "\n";

        // Convert and write pixels
        std::vector<unsigned char> hdrScanline(width * 4);
        for ( int y = height - 1; y >= 0; --y ) // HDR files are top-to-bottom
        {
            for( int x = 0; x < width; ++x ) 
            {
                float4 color = getExrPixelColor( buff.data(), width, texInfo, x, y );
                float2rgbe( color.x, color.y, color.z, &hdrScanline[x * 4] );
            }
            outFile.write( reinterpret_cast<char*>(hdrScanline.data()), width * 4 );
        }

        return true;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error converting EXR to HDR: " << e.what() << std::endl;
        return false;
    }
}


// TGA Header structure
#pragma pack(push, 1)
struct TGAHeader {
    uint8_t  idLength = 0;
    uint8_t  colorMapType = 0;
    uint8_t  imageType = 2;  // Uncompressed RGB/RGBA
    uint16_t colorMapStart = 0;
    uint16_t colorMapLength = 0;
    uint8_t  colorMapBits = 0;
    uint16_t xOrigin = 0;
    uint16_t yOrigin = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t  bitsPerPixel = 32;  // RGBA8
    uint8_t  imageDescriptor = 0x28;  // Top-left origin, 8 bits alpha
};
#pragma pack(pop)

// Helper function to convert float to 8-bit with basic tone mapping
inline uint8_t float2byte( float v, float exposure, float gamma )
{
    v *= exposure;
    if( gamma != 1.0f )
        v = std::pow( v, 1.0f / gamma );
    v = std::max( 0.0f, std::min( 1.0f, v ) );
    return static_cast<uint8_t>( v * 255.0f + 0.5f );
}

bool CompressedTextureCacheManager::convertEXRtoTGA4( CoreEXRReader& exrReader, TextureInfo& texInfo,
    float exposure, float gamma, std::string& outFileName )
{
    try {
        // Read mip level from EXR
        int mipLevel = std::min( texInfo.numMipLevels - 1, m_options.droppedMipLevels );
        int width = static_cast<int>( texInfo.width >> mipLevel ); 
        int height = static_cast<int>( texInfo.height >> mipLevel );
        std::vector<char> buff( width * height * ( getBitsPerPixel( texInfo ) / BITS_PER_BYTE ) );
        if( !exrReader.readMipLevel( buff.data(), mipLevel, width, height, CUstream{0} ) )
        {
            std::cerr << "Failed to read EXR data" << std::endl;
            return false;
        }
        
        // Open output TGA file and write header
        std::ofstream outFile( outFileName, std::ios::binary );
        if( !outFile ) 
        {
            std::cerr << "Failed to open output file: " << outFileName << std::endl;
            return false;
        }
        TGAHeader header;
        header.width = width;
        header.height = height;
        outFile.write( reinterpret_cast<const char*>( &header ), sizeof( header ) );

        // Convert and write pixels
        std::vector<uint8_t> tgaScanline( width * 4 );
        for ( int y = height - 1; y >= 0; --y )
        {
            for ( int x = 0; x < width; ++x ) 
            {
                // Get pixel and convert to BGRA
                float4 color = getExrPixelColor( buff.data(), width, texInfo, x, y );
                tgaScanline[x*4 + 0] = float2byte( color.z, exposure, gamma );
                tgaScanline[x*4 + 1] = float2byte( color.y, exposure, gamma );
                tgaScanline[x*4 + 2] = float2byte( color.x, exposure, gamma );
                tgaScanline[x*4 + 3] = float2byte( color.w, 1.0f, 1.0f );
            }
            outFile.write( reinterpret_cast<const char*>(tgaScanline.data()), tgaScanline.size() );
        }

        return true;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error converting EXR to TGA: " << e.what() << std::endl;
        return false;
    }
}

bool CompressedTextureCacheManager::convertImageToDDS( const std::string& inFile, const std::string& outFile, const std::string& ddsFormat )
{
    std::string command = m_options.nvcompress + " -" + ddsFormat + " -silent "
        + " \"" + inFile + "\"" + " \"" + outFile + "\"";
    printCommand( command );
    if( std::system( command.c_str() ) != 0 )
        return false;

    // Check to to see if nvcompress failed. This can happen when cudaMalloc fails because
    // multiple instances of nvcompress are running at the same time.
    const unsigned int failSize = 148;
    std::uintmax_t fileSize = fs::file_size( fs::path( outFile ) );
    if( fileSize <= failSize )
        deleteFile( outFile );
    return ( fileSize > failSize );
}

bool CompressedTextureCacheManager::convertDDSToTiledDDS( const std::string& inFile, const std::string& outFile )
{
    std::string command = "Converting DDS file \"" + inFile + "\" to tiled DDS file \"" + outFile + "\"";
    printCommand( command );
    DDSImageReader reader( inFile, false );
    return reader.saveAsTiledFile( outFile.c_str() );
}

bool CompressedTextureCacheManager::deleteFile( const std::string &fileName )
{
    std::string command = std::string(DELETE_COMMAND) + " \"" + fileName + "\"";
    printCommand( command );
    return ( std::system( command.c_str() ) == 0 );
}

}  // namespace imageSource
