// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <fstream>
#include <string>

#include <OptiXToolkit/ImageSource/DDSImageReader.h>
#include <OptiXToolkit/ImageSource/CoreEXRReader.h>

namespace imageSource {

// How to convert EXR images to DDS depending on hdr statas and num components
//                                   default, r,   rg,    rgb,   rgba,  hdr
#define EXR_TO_DDS_FORMATS_STANDARD {"bc7", "bc4", "bc5", "bc7", "bc7", "bc6"}
#define EXR_TO_DDS_FORMATS_SMALL    {"bc1", "bc4", "bc1", "bc1", "bc7", "bc6"}

const int DDS_DEFAULT_INDEX = 0;
const int DDS_HDR_INDEX = 5;

const int NUM_SUPPORTED_FILE_TYPES = 9;
const char* SUPPORTED_FILE_TYPES[NUM_SUPPORTED_FILE_TYPES] = {".exr", ".png", ".jpg", ".ppm", ".pgm", ".tga", ".hdr", ".dds", ".bmp"};

struct CompressedTextureCacheOptions
{
    std::string imageMagick = "magick";               // imageMagick executable
    std::string nvcompress  = "nvcompress";           // nvCompress executable
    std::string cacheFolder = "compressedCache"; // Where to store the compressed cache

    unsigned int droppedMipLevels = 1;    // How many mip levels to drop when images are put in the cache
    unsigned int minImageSize     = 512;  // Don't drop mip levels of cached images to below this size
    unsigned int hdrCheckSize     = 256;  // Size of image to read to see if an exr is an hdr image

    std::string exrToDdsFormats[6] = EXR_TO_DDS_FORMATS_STANDARD;
};

class CompressedTextureCacheManager
{
  public:
    /// Create a compressed texture cache manager with the given options
    CompressedTextureCacheManager( const CompressedTextureCacheOptions& options ) { m_options = options; }

    /// Get the cache file path that will be used for a given input file
    std::string getCacheFilePath( const std::string& inputFilePath );

    /// Put an input file into the compressed cache
    bool cacheFile( const std::string& inputFilePath );

    /// Determine if a file extension is of a known type for compression
    bool isSupportedFileType( const std::string& extension );

    /// Set the verbose mode. Verbose mode prints commands
    void setVerbose( bool verbose ) { m_verbose = verbose; }
    
  private:
    CompressedTextureCacheOptions m_options;
    bool m_verbose = false;

    void printCommand( const std::string& command );
    bool isHighDynamicRange( CoreEXRReader& exrReader, TextureInfo& texInfo );
    bool convertImageToIntermediateFormat( const std::string& inFile, const std::string& outFile, unsigned int width, unsigned int height );
    bool convertImageToDDS( const std::string& inFile, const std::string& outFile, const std::string& ddsFormat );
    bool convertDDSToTiledDDS( const std::string& inFile, const std::string& outFile );
    bool deleteFile( const std::string &fileName );
};

}  // namespace imageSource