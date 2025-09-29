// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <execution>
#include <iostream>
#include <string>
#include <filesystem>

#include <omp.h>

#include <OptiXToolkit/ImageSource/CompressedTextureCacheManager.h>

using namespace imageSource;
namespace fs = std::filesystem;

void printUsage( char* program )
{
    // clang-format off
    std::cerr << "\nUsage: " << program << "[options] <src folder> <cache folder>\n" <<
        "\n"
        "Options:\n"
        "   --imageMagick | -im <imageMagick path>     imageMagick executable path.\n"
        "   --nvcompress | -nc <nvcompress path>       nvcompress executable path.\n"
        "   --dropMipLevels | -dl <numLevels>          mip levels to drop when putting files in cache.\n"
        "   --minIMageSize | -mi <minSize>             minimum size to maintain for files put in cache.\n"
        "   --hdrCheckSize | -cs <checkSize>           size of mip level to check to see if a file is hdr.\n"
        "   --verbose | -v                             turn on verbose output.\n"
        "   --terse | -t                               turn off verbose output.\n";
    // clang-format on

    exit(0);
}

void cacheFiles( std::string srcFolder, CompressedTextureCacheManager& cacheManager, bool parallel )
{
    // Get a list of all the source images
    std::vector<std::string> sourceImages;
    try 
    {
        for ( const auto& entry : fs::directory_iterator( srcFolder ) ) 
        {
            if( entry.is_regular_file() && cacheManager.isSupportedFileType( entry.path().extension() ) )
            {
                sourceImages.push_back( entry.path().string() );
            }
        }
    } 
    catch (const fs::filesystem_error& e) 
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    // Convert the images, either in parallel or serial mode
    if( parallel )
    {
        cacheManager.setVerbose(false);
        #pragma omp parallel for num_threads(32)
        for( unsigned int i = 0; i < sourceImages.size(); ++i )
        {
            cacheManager.cacheFile( sourceImages[i] );
        }
    }
    else
    {
        for( unsigned int i = 0; i < sourceImages.size(); ++i )
        {
            cacheManager.cacheFile( sourceImages[i] );
        }
    }
}

int main( int argc, char* argv[] )
{
    CompressedTextureCacheOptions options{};
    bool verbose = true;
    bool parallel = true;

    for( int i = 1; i < argc - 2; ++i )
    {
        const std::string arg( argv[i] );
        const bool lastArg = ( i == argc - 1 );

        if( ( arg == "--imageMagick" || arg == "-im" ) && !lastArg )
            options.imageMagick = argv[++i];
        else if( ( arg == "--nvcompress" || arg == "-nc" ) && !lastArg )
            options.nvcompress = argv[++i];
        else if( ( arg == "--dropMipLevels" || arg == "-dl" ) && !lastArg )
            options.droppedMipLevels = atoi( argv[++i] );
        else if( ( arg == "--minImageSize" || arg == "-mi" ) && !lastArg )
            options.minImageSize = atoi( argv[++i] );
        else if( ( arg == "--hdrCheckSize" || arg == "--hc" ) && !lastArg )
            options.hdrCheckSize = atoi( argv[++i] );
        else if( arg == "--verbose" || arg == "-v" )
            verbose = true;
        else if( arg == "--terse" || arg == "-t" )
            verbose = false;
        else if( arg == "--parallel" )
            parallel = true;
        else if( arg == "--serial" )
            parallel = false;
        else 
            printUsage( argv[0] );
    }

    options.cacheFolder = argv[argc - 1];
    CompressedTextureCacheManager cacheManager( options );
    cacheManager.setVerbose( verbose );

    cacheFiles( argv[argc - 2], cacheManager, parallel );
}
