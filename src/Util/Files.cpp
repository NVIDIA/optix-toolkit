//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "Util/Config.h"  // generated from Util/src/Config.h.in
#include "Util/Files.h"
#include "Util/Exception.h"

#include <fstream>

namespace demandLoading {

static bool fileExists( const char* path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

static bool fileExists( const std::string& path )
{
    return fileExists( path.c_str() );
}

static std::string existingFilePath( const char* directory, const char* relativeSubDir, const char* relativePath )
{
    std::string path = directory ? directory : "";
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
    return fileExists( path ) ? path : "";
}

const char* getRuntimeFilePath( const char* relativeSubDir, const char* relativePath )
{
    static std::string s;

    // Allow for overrides.
    static const char* directories[] =
    {
        OTK_RUNTIME_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( directory )
        {
            s = existingFilePath( directory, relativeSubDir, relativePath );
            if( !s.empty() )
            {
                return s.c_str();
            }
        }
    }
    throw Exception( ( std::string{"otk::getRuntimeFilePath couldn't locate "} + relativePath ).c_str() );
}

} // namespace demandLoading
