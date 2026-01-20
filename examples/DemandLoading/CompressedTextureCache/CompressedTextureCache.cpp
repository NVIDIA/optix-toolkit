// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <iostream>
#include <string>
#include <filesystem>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <omp.h>

#include <OptiXToolkit/ImageSource/CompressedTextureCacheManager.h>

using namespace imageSource;
namespace fs = std::filesystem;

std::string trimString( const std::string& str )
{
    const char* whitespace = " \t\v\r\n";
    std::size_t start = str.find_first_not_of( whitespace );
    std::size_t end = str.find_last_not_of( whitespace );
    return start == end ? std::string() : str.substr( start, end - start + 1 );
}

bool loadTextureSetsFromFile( const std::string& filePath, std::vector<std::vector<std::string>>& textureSets )
{
    std::ifstream file( filePath );
    if( !file.is_open() )
        return false;

    std::vector<std::string> textureSet;
    std::string line;

    while( std::getline( file, line ) )
    {
        line = trimString( line );
        if( line.empty() )
        {
            textureSets.push_back( textureSet );
            textureSet.clear();
            continue;
        }
        textureSet.push_back( line );
    }
    if( !textureSet.empty() )
        textureSets.push_back( textureSet );

    return true;
}

void printUsage( char* program )
{
    // clang-format off
    std::cerr << "\nUsage: " << program << " [options] <src folders or files | image list files>\n" <<
        "\n"
        "General Options:\n"
        "   --cacheFolder | -cf <cache folder>           cache folder location.\n"
        "   --dropMipLevels | -dl <numLevels>            mip levels to drop when putting files in cache. (default 0)\n"
        "   --flags | -f <flags>                         flags to pass to nvcompress or ntc-cli.\n"
        "   --help | -h                                  print this message.\n"
        "   --verbose | -v                               turn on verbose output.\n"
        "   --multiGPU | -mg                             use multiple GPUs if available.\n"
        "   --showErrors | -e                            show errors from external programs.\n"
        "   --threads | -t <numThreads>                  number of threads to use. (default 8, 1 for NTC)\n"
        "\n"
        "DDS Compression Options:\n"
        "   --nvcompress | -nc <nvcompress executable>   nvcompress executable path.\n"
        "   --minImageSize | -mi <minSize>               minimum size to maintain for files put in cache. (default 512)\n"
        "   --hdrCheckSize | -cs <checkSize>             size of mip level to check to see if a file is hdr. (default 256)\n"
        "   --small | -s                                 use lower quality, higher compression formats. (default off)\n"
        "   --noTile | -nt                               don't tile the dds outputs.\n"
        "\n"
        "Neural Texture Compression Options:\n"
        "   --ntc-cli <ntc-cli executable>               ntc-cli executable.\n"
        "   --inputFolder | -if <input folder>           input folder to read texture sets from.\n"
        "   --numFeatures | -nf <numFeatures>            number of latent features. (default 8)\n"
        "\n";
    // clang-format on

    exit(0);
}

int main( int argc, char* argv[] )
{
    const int DEFAULT_NTC_THREADS = 1;
    const int DEFAULT_NVCOMPRESS_THREADS = 8;

    CompressedTextureCacheOptions options{};
    bool verbose = false;
    int numThreads = 0;
    int numCudaDevices = 1;
    std::string inputFolder = "";

    //
    // Process command line options
    //

    int idx;
    for( idx = 1; idx < argc; ++idx )
    {
        std::string arg( argv[idx] );
        std::transform( arg.begin(), arg.end(), arg.begin(), ::tolower );
        bool lastArg = ( idx == argc - 1 );

        if( arg == "--help" || arg == "-h" )
            printUsage( argv[0] );
        else if( ( arg == "--nvcompress" || arg == "-nc" ) && !lastArg )
            options.nvcompress = argv[++idx];
        else if( ( arg == "--flags" || arg == "-f" ) && !lastArg )
            options.flags = argv[++idx];
        else if( ( arg == "--cachefolder" || arg == "-cf" ) && !lastArg )
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
        else if( arg == "--multigpu" || arg == "-mg" )
            cudaGetDeviceCount( &numCudaDevices );
        else if( arg == "--notile" || arg == "-nt" )
            options.saveTiled = false;
        else if( arg == "--small" || arg == "-s" )
            options.exrToDdsFormats = EXR_TO_DDS_FORMATS_SMALL;
        else if( arg == "--showerrors" || arg == "-e" )
            options.showErrors = true;
        else if( ( arg == "--ntc-cli" ) && !lastArg )
            options.ntcCli = argv[++idx];
        else if( ( arg == "--inputfolder" || arg == "-if" ) && !lastArg )
            inputFolder = argv[++idx];
        else if( ( arg == "--numfeatures" || arg == "-nf" ) && !lastArg )
            options.numNtcFeatures = atoi( argv[++idx] );
        else 
            break;
    }

    //
    // Check for nvcompress and ntc-cli executables
    //

    bool nvcompressFound = false;
    bool ntcCliFound = false;

#ifdef _WIN32
    if( !options.nvcompress.empty() )
        nvCompressFound = fs::exists( fs::absolute( options.nvcompress ) );
    if( !options.ntcCli.empty() )
        ntcCliFound = fs::exists( fs::absolute( options.ntcCli ) );
#else // linux
    if( !options.nvcompress.empty() )
        nvcompressFound = access( fs::absolute( options.nvcompress ).c_str(), X_OK ) == 0;
    if( !options.ntcCli.empty() )
        ntcCliFound = access( fs::absolute( options.ntcCli ).c_str(), X_OK ) == 0;
#endif

    if( numThreads == 0 )
        numThreads = ntcCliFound ? DEFAULT_NTC_THREADS : DEFAULT_NVCOMPRESS_THREADS;

    if( !nvcompressFound && !ntcCliFound )
    {
        if( options.nvcompress.empty() && options.ntcCli.empty() )
            std::cerr << "Error: No nvcompress executable or ntc-cli executable provided.\n";
        else if( !options.nvcompress.empty() )
            std::cerr << "Could not find nvcompress executable \"" << options.nvcompress << "\".\n";
        else if( !options.ntcCli.empty() )
            std::cerr << "Could not find ntc-cli executable \"" << options.ntcCli << "\".\n";
        
        std::cerr << "  nvcompress is part of the nvidia texture tools suite. It can be downloaded from\n";
        std::cerr << "  https://developer.nvidia.com/gpu-accelerated-texture-compression\n";
        std::cerr << "  ntc-cli is part of the nvidia neural texture compression SDK. It can be downloaded from\n";
        std::cerr << "  https://github.com/NVIDIA-RTX/RTXNTC\n\n";
        printUsage( argv[0] );
    }

    //
    // Gather a list of all the source images and texture sets
    //

    CompressedTextureCacheManager cacheManager( options );
    cacheManager.setVerbose( verbose );
    std::vector<std::string> sourceImages;
    std::vector<std::vector<std::string>> textureSets;

    try 
    {
        for( idx = idx; idx < argc; ++idx )
        {
            std::filesystem::path file( argv[idx] );
            if( file.extension().string() == ".ntc" )
            {
                sourceImages.push_back( file.string() );
            }
            else if( file.extension().string() == ".txt" )
            {
                if( !loadTextureSetsFromFile( argv[idx], textureSets ) )
                {
                    std::cerr << "Error: failed to load texture sets from " << argv[idx] << "\n\n";
                    printUsage( argv[0] );
                }
            }
            else if( cacheManager.isSupportedFileType( file.extension().string() ) )
            {
                sourceImages.push_back( file.string() );
            }
            else if( fs::is_directory( argv[idx] ) )
            {
                for ( const auto& entry : fs::directory_iterator( argv[idx] ) ) 
                {
                    if( entry.is_regular_file() && cacheManager.isSupportedFileType( entry.path().extension().string() ) )
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

    // Gather texture sets specified on command line.
    for( int i = (int)sourceImages.size() - 1; i >= 0; --i )
    {
        std::filesystem::path sourceFile( sourceImages[i] );
        if( sourceFile.extension().string() == ".ntc" )
        {
            std::vector<std::string> textureSet;
            textureSet.push_back( sourceImages[i] );
            for( int j = i+1; j < (int)sourceImages.size(); ++j )
            {
                textureSet.push_back( sourceImages[j] );
            }
            sourceImages.resize( i );
            textureSets.push_back( textureSet );
        }
    }

    // Prepend the input folder to the texture set input image files.
    for( std::vector<std::string>& textureSet : textureSets )
    {
        for( unsigned int i = 1; i < textureSet.size(); ++i )
        {
            textureSet[i] = inputFolder + FILE_SEPARATOR + textureSet[i];
        }
    }

    if( textureSets.size() == 0 && sourceImages.size() == 0 )
    {
        std::cerr << "No input files given.\n\n";
        printUsage( argv[0] );
    }

    //
    // Convert the images
    //

    if( numThreads > 1 )
        cacheManager.setVerbose( false );

    // Convert the source images to DDS.
    if( nvcompressFound )
    {
        if ( verbose )
        {
            printf( "Found %d image files. Processing...\n", static_cast<int>( sourceImages.size() ) );
        }

        // Sometimes nvcompress can fail due to insufficient device
        // memory if too many threads are in flight. Try each conversion
        // several times, reducing the number of threads on each pass.

        const int numPasses = 4;
        int passThreads[numPasses] = {numThreads, 3, 1, 1};
        unsigned int conversionCount = 0;

        for( int pass = 0; pass < numPasses; ++pass )
        {
            omp_set_num_threads( passThreads[pass] );
            #pragma omp parallel for schedule(dynamic)
            for( unsigned int i = 0; i < sourceImages.size(); ++i )
            {
                std::string outputImage = cacheManager.getCacheFilePath( sourceImages[i], ".dds" );
                bool outFileExists = fs::exists( outputImage );
                bool doPrint = verbose && ( !outFileExists || ( pass == 0 ) );

                if( doPrint )
                    printf( "Converting: \"%s\" ==> \"%s\"\n", sourceImages[i].c_str(), outputImage.c_str() );

                if( outFileExists )
                {
                    if( pass == 0 )
                    conversionCount++;
                }
                else
                {
                    if( cacheManager.cacheFileAsDDS( sourceImages[i], idx % numCudaDevices ) )
                        conversionCount++;
                    else if( verbose )
                        printf( "GPU busy converting %s. Will retry.\n", outputImage.c_str() );
                }
            }

            // Turn off verbose printing after first pass.
            verbose = false;

            // Check if all the conversions are done.
            if( conversionCount >= sourceImages.size() )
                break;
        }

        if( conversionCount != sourceImages.size() )
        {
            std::cerr << "Error: Failed to convert all source images.\n";
            printUsage( argv[0] );
        }
    }

    // Convert the texture sets to NTC.
    if( ntcCliFound )
    {
        if( verbose )
        {
            printf( "Found %d texture sets. Processing...\n", static_cast<int>( textureSets.size() ) );
        }

        unsigned int conversionCount = 0;

        omp_set_num_threads( numThreads );
        #pragma omp parallel for schedule(dynamic)
        for( const auto& textureSet : textureSets )
        {
            std::string outputImage = cacheManager.getCacheFilePath( textureSet[0], "" );
            bool outFileExists = fs::exists( outputImage );
            bool doPrint = verbose && !outFileExists;
            if( doPrint )
                printf( "Compressing texture set: \"%s\"\n", outputImage.c_str() );
            else if( verbose )
                printf( "Skipping existing texture set \"%s\".\n", outputImage.c_str() );

            if( outFileExists )
            {
                conversionCount++;
            }
            else
            {
                if( cacheManager.cacheTextureSetAsNtc( textureSet, idx % numCudaDevices ) )
                    conversionCount++;
                else
                    printf( "Error: Compressing texture set \"%s\" failed.\n", outputImage.c_str() );
            }
        }

        if( conversionCount != textureSets.size() )
        {
            std::cerr << "Error: Failed to convert all texture sets.\n";
            printUsage( argv[0] );
        }
    }

    return 0;
}
