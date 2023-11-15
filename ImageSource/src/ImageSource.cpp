//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/ImageSource/ImageSource.h>

#include "Config.h"  // for OTK_USE_OIIO

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>
#include <OptiXToolkit/ImageSource/CoreEXRReader.h>
#if OTK_USE_OIIO
#include <OptiXToolkit/ImageSource/OIIOReader.h>
#endif

#include <cstddef>  // for size_t
#include <fstream>
#include <memory>
#include <string>

namespace {

bool fileExists( const std::string& path )
{
    return std::ifstream( path ).good();
}

}  // namespace


namespace imageSource {

bool ImageSourceBase::readMipTail( char*        dest,
                                   unsigned int mipTailFirstLevel,
                                   unsigned int numMipLevels,
                                   const uint2* mipLevelDims,
                                   unsigned int pixelSizeInBytes,
                                   CUstream     stream )
{
    size_t offset = 0;
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims = mipLevelDims[mipLevel];
        readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream );

        // Increment offset.
        offset += levelDims.x * levelDims.y * pixelSizeInBytes;
    }

    return true;
}

std::shared_ptr<ImageSource> createImageSource( const std::string& filename, const std::string& directory )
{
    // Special cases
    if( filename == "checkerboard" )
    {
        return std::make_shared<CheckerBoardImage>( 2048, 2048, /*squaresPerSide=*/32, /*useMipmaps=*/true );
    }

    // Construct ImageSource based on filename extension.
    const size_t      dot       = filename.find_last_of( '.' );
    const std::string extension = dot == std::string::npos ? "" : filename.substr( dot );

    // Attempt relative path first, then absolute path.
    const std::string path = fileExists( filename ) ? filename : directory + '/' + filename;

    if( extension == ".exr" )
    {
        return std::make_shared<CoreEXRReader>( path );
    }

#if OTK_USE_OIIO
    return std::make_shared<OIIOReader>( path );
#else
    throw std::runtime_error( "Image file not supported: " + filename );
#endif
}

}  // namespace imageSource
