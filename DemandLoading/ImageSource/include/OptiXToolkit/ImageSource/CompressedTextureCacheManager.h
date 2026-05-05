// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <fstream>
#include <string>
#include <array>

#include <OptiXToolkit/ImageSource/DDSImageReader.h>
#include <OptiXToolkit/ImageSource/CoreEXRReader.h>

namespace imageSource {

extern std::array<const char*, 6> EXR_TO_DDS_FORMATS_STANDARD;
extern std::array<const char*, 6> EXR_TO_DDS_FORMATS_SMALL;

struct CompressedTextureCacheOptions
{
    unsigned int droppedMipLevels = 0;            // How many mip levels to drop when images are put in the cache
    std::string cacheFolder = "compressedCache";  // Where to store the compressed cache
    bool showErrors = false;                      // Show errors from external programs
    std::string flags;                            // flags to pass to ntc-cli and nvcompress
    unsigned int minImageSize = 512;              // Don't drop mip levels of cached images to below this size
    unsigned int hdrCheckSize = 256;              // Size of image to read to see if an exr is an hdr image

    std::string nvcompress;                      // nvcompress executable
    bool saveTiled = true;                       // Save images as tiled or regular dds images
    const char** exrToDdsFormats = EXR_TO_DDS_FORMATS_STANDARD.data();

    std::string ntcCli;                          // ntc-cli executable
    unsigned int numNtcFeatures = 8;             // Number of features to use for Neural Texture Compression
};

class CompressedTextureCacheManager
{
  public:
    /// Create a compressed texture cache manager with the given options
    CompressedTextureCacheManager( const CompressedTextureCacheOptions& options ) { m_options = options; }

    /// Get the cache file path that will be used for a given input file
    std::string getCacheFilePath( const std::string& cacheFileName, const std::string& extension );

    /// Put a texture set into the compressed cache using Neural Texture Compression
    bool cacheTextureSetAsNtc( const std::vector<std::string>& inputFilePaths, int deviceId = 0 );

    /// Put an input file into the compressed cache as a DDS image
    bool cacheFileAsDDS( const std::string& inputFilePath, int deviceId = 0 );

    /// Determine if a file extension is of a known type for compression
    bool isSupportedFileType( const std::string& extension );

    /// Set the verbose mode. Verbose mode prints commands
    void setVerbose( bool verbose ) { m_verbose = verbose; }
    
  private:
    CompressedTextureCacheOptions m_options;
    bool m_verbose = false;

    void printCommand( const std::string& command );
    bool isHighDynamicRange( CoreEXRReader& exrReader, TextureInfo& texInfo );
    bool convertEXRtoHDR( CoreEXRReader& exrReader, TextureInfo& texInfo, const std::string& outFile );
    bool convertEXRtoTGA4( CoreEXRReader& exrReader, TextureInfo& texInfo, float exposure, float gamma, std::string& outFileName );
    bool convertImageToDDS( const std::string& inFile, const std::string& outFile, const std::string& ddsFormat, int deviceId );
    bool convertDDSToTiledDDS( const std::string& inFile, const std::string& outFile );
    bool deleteFile( const std::string &fileName );
};

}  // namespace imageSource
