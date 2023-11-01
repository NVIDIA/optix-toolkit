//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
