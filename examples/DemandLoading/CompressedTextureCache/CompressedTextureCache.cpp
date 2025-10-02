// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <execution>
#include <iostream>
#include <string>
#include <filesystem>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <omp.h>

#include <OptiXToolkit/ImageSource/CompressedTextureCacheManager.h>

using namespace imageSource;
namespace fs = std::filesystem;

void printUsage( char* program )
{
    // clang-format off
    std::cerr << "\nUsage: " << program << " [options] <src folders or files>\n" <<
        "\n"
        "Options:\n"
        "   --help | -h                                   print this message.\n"
        "   --nvcompress | -nc <nvcompress path + flags>  nvcompress executable path.\n"
        "   --cacheFolder | -cf <cache folder>            cache folder location.\n"
        "   --dropMipLevels | -dl <numLevels>             mip levels to drop when putting files in cache. (default 0)\n"
        "   --minImageSize | -mi <minSize>                minimum size to maintain for files put in cache. (default 512)\n"
        "   --hdrCheckSize | -cs <checkSize>              size of mip level to check to see if a file is hdr. (default 256)\n"
        "   --verbose | -v                                turn on verbose output.\n"
        "   --threads | -t <numThreads>                   number of threads to use. (default 3 to avoid running out of GPU memory.)\n"
        "   --small | -s                                  use lower quality, higher compression formats. (default off)\n"
        "   --noTile | -nt                                don't tile the dds outputs.\n";
    // clang-format on

    exit(0);
}

int main( int argc, char* argv[] )
{
    CompressedTextureCacheOptions options{};
    bool verbose = false;
    int numThreads = 5;

    //
    // Process command line options
    //

    int idx;
    for( idx = 1; idx < argc - 1; ++idx )
    {
        std::string arg( argv[idx] );
        std::transform( arg.begin(), arg.end(), arg.begin(), ::tolower );
        bool lastArg = ( idx == argc - 1 );

        if( arg == "--help" || arg == "-h" )
            printUsage( argv[0] );
        else if( ( arg == "--nvcompress" || arg == "-nc" ) && !lastArg )
            options.nvcompress = argv[++idx];
        else if( ( arg == "--cachefolder " || arg == "-cf" ) && !lastArg )
            options.cacheFolder = argv[++idx];
        else if( ( arg == "--dropmiplevels" || arg == "-dl" ) && !lastArg )
            options.droppedMipLevels = atoi( argv[++idx] );
        else if( ( arg == "--minimagesize" || arg == "-mi" ) && !lastArg )
            options.minImageSize = atoi( argv[++idx] );
        else if( ( arg == "--hdrchecksize" || arg == "--hc" ) && !lastArg )
            options.hdrCheckSize = atoi( argv[++idx] );
        else if( arg == "--verbose" || arg == "-v" )
            verbose = true;
        else if( ( arg == "--threads" || arg == "-t" ) && !lastArg )
            numThreads = atoi( argv[++idx] );
        else if( arg == "--notile" || arg == "-nt" )
            options.saveTiled = false;
        else if( arg == "--small" || arg == "-s" )
            options.exrToDdsFormats = EXR_TO_DDS_FORMATS_SMALL;
        else 
            break;
    }


    //
    // Check for nvcompress executable
    //

#ifdef _WIN32
    std::filesystem::path nvcompress( options.nvcompress );
    std::string extension = nvcompress.extension().string();
    bool nvcompressFound = ( extension == ".exr" );
#else // linux
    std::filesystem::path nvcompress( options.nvcompress );
    bool nvcompressFound = ( access( nvcompress.c_str(), X_OK ) == 0 );
#endif

    if( !nvcompressFound )
    {
        std::cerr << "Error: Unable to find nvcompress executable \"" << options.nvcompress << "\".\n";
        std::cerr << "  nvcompress is part of the nvidia texture tools suite. It can be downloaded from\n";
        std::cerr << "  https://developer.nvidia.com/gpu-accelerated-texture-compression\n\n";
        printUsage( argv[0] );
    }


    //
    // Gather a list of all the source images
    //

    CompressedTextureCacheManager cacheManager( options );
    cacheManager.setVerbose( verbose );
    std::vector<std::string> sourceImages;

    try 
    {
        for( idx = idx; idx < argc; ++idx )
        {
            if( fs::is_regular_file( argv[idx] ) )
            {
                std::filesystem::path file( argv[idx] );
                if( !cacheManager.isSupportedFileType( file.extension() ) )
                {
                    std::cerr << "Error: unknown image file: " << argv[idx] << "\n\n";
                    printUsage( argv[0] );
                }
                sourceImages.push_back( file.string() );
            }
            else if( fs::is_directory( argv[idx] ) )
            {
                for ( const auto& entry : fs::directory_iterator( argv[idx] ) ) 
                {
                    if( entry.is_regular_file() && cacheManager.isSupportedFileType( entry.path().extension() ) )
                        sourceImages.push_back( entry.path().string() );
                }
            }
            else
            {
                std::cerr << "Error: unknown file or folder: " << argv[idx] << "\n\n";
                printUsage( argv[0] );
            }
        }
    } 
    catch( const fs::filesystem_error& e ) 
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        printUsage( argv[0] );
    }

    if( sourceImages.size() == 0 )
    {
        std::cerr << "No input files given.\n\n";
        printUsage( argv[0] );
    }


    //
    // Convert the images
    //

    if( numThreads > 1 )
        cacheManager.setVerbose( false );

    // Sometimes nvcompress can fail due to insufficient device
    // memory if too many threads are in flight. Try each conversion
    // up to 3 times, reducing the number of threads on each pass.

    const int numPasses = 3;
    int passThreads[numPasses] = {numThreads, 3, 1};
    unsigned int conversionCount = 0;

    for( int pass = 0; pass < numPasses; ++pass )
    {
        omp_set_num_threads( passThreads[pass] );
        #pragma omp parallel for schedule(dynamic)
        for( unsigned int i = 0; i < sourceImages.size(); ++i )
        {
            std::string outputImage = cacheManager.getCacheFilePath( sourceImages[i] );
            if( verbose )
                printf( "Converting: \"%s\" ==> \"%s\"\n", sourceImages[i].c_str(), outputImage.c_str() );

            if( cacheManager.cacheFile( sourceImages[i] ) )
                conversionCount++;
            else if( verbose )
                printf( "Failed to convert %s. Will retry.\n", outputImage.c_str() );
        }

        // Turn off verbose printing after first pass.
        verbose = false;

        // Check if all the conversions are done.
        if( conversionCount >= sourceImages.size() )
            break;
    }

    return 0;
}
