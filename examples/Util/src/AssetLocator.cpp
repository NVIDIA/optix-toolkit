// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Util/AssetLocator.h>

#include "BinaryDataDir.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <cstdlib>
#include <fstream>

namespace otk {

static bool fileExists( const char* path )
{
    return static_cast<bool>( std::ifstream( path ) );
}

static bool fileExists( const std::string& path )
{
    return fileExists( path.c_str() );
}

static std::string existingFilePath( const char* directory, const char* relativeSubDir, const char* relativePath )
{
    std::string path;
    if( directory )
        path = directory;
    if( relativeSubDir )
    {
        path += '/';
        path += relativeSubDir;
    }
    if( relativePath )
    {
        path += '/';
        path += relativePath;
    }
    if( fileExists( path ) )
        return path;
    return {};
}

std::string locateAsset( const char* relativeSubDir, const char* relativePath )
{
    static const char *directories[] = {
        std::getenv("OTK_ASSET_DIR"),
        OTK_BINARY_DATA_DIR,
    };

    for( const char* directory : directories )
    {
        // getenv returns nullptr when the environment variable is not set.
        if( directory == nullptr )
            continue;

        std::string s = existingFilePath( directory, relativeSubDir, relativePath );
        if( !s.empty() )
        {
            return s;
        }
    }
    throw std::runtime_error( ( std::string{ "Couldn't locate asset " } +relativePath ).c_str() );
}

}  // namespace otk
