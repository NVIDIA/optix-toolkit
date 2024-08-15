// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    const std::string path = directory.empty() ? filename : ( fileExists( filename ) ? filename : directory + '/' + filename );

#if OTK_USE_OPENEXR    
    if( extension == ".exr" )
    {
        return std::make_shared<CoreEXRReader>( path );
    }
#endif
#if OTK_USE_OIIO
    return std::make_shared<OIIOReader>( path );
#else
    throw std::runtime_error( "Image file not supported: " + filename );
#endif
}

}  // namespace imageSource
