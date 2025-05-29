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
#include <OptiXToolkit/ImageSource/DDSImageReader.h>

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

unsigned long long ImageSource::getHash( CUstream stream )
{
    TextureInfo info;
    open( &info );

    // FIXME: Handle non-host fulfilled image types
    if( getFillType() != CU_MEMORYTYPE_HOST )
        return 0ULL;

    // Start hash as checksum of info
    const unsigned long long m = 1013904223ULL;
    unsigned long long hash = info.width;
    hash = hash * m + info.height;
    hash = hash * m + info.format;
    hash = hash * m + info.numChannels;
    hash = hash * m + info.numMipLevels;

    // Continue with checksum of a coarse mip level.
    int mipLevel = static_cast<int>( log2f( (info.width + info.height) / 64.0f ) );
    mipLevel = std::max( std::min( mipLevel, (int)info.numMipLevels-1 ), 0 );

    unsigned int levelWidth = info.width >> mipLevel;
    unsigned int levelHeight = info.height >> mipLevel;
    unsigned int mipLevelSize = ( levelWidth * levelHeight * getBitsPerPixel( info ) ) / BITS_PER_BYTE;
    std::vector<unsigned long long> buffer( ( mipLevelSize + 7 ) / 8 );
    buffer.back() = 0;
    readMipLevel( (char*)buffer.data(), mipLevel, levelWidth, levelHeight, stream );

    for( unsigned int i = 0; i < buffer.size(); ++i )
        hash = hash * m + buffer[i];

    return hash;
}

bool ImageSourceBase::readMipTail( char*        dest,
                                   unsigned int mipTailFirstLevel,
                                   unsigned int numMipLevels,
                                   const uint2* mipLevelDims,
                                   CUstream     stream )
{
    int bitsPerPixel = getBitsPerPixel( getInfo() );
    size_t offset = 0;
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims = mipLevelDims[mipLevel];
        readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream );

        // Increment offset.
        offset += ( levelDims.x * levelDims.y * bitsPerPixel ) / BITS_PER_BYTE;
    }

    return true;
}

std::shared_ptr<ImageSource> createImageSource( const std::string& filename, const std::string& directory )
{
    // Special cases
    if( filename == "checkerboard" )
    {
        return std::make_shared<CheckerBoardImage>( 8192, 8192, /*squaresPerSide=*/16, /*useMipmaps=*/true );
    }

    // Construct ImageSource based on filename extension.
    const size_t      dot       = filename.find_last_of( '.' );
    const std::string extension = ( dot == std::string::npos ) ? "" : filename.substr( dot );

    // Attempt relative path first, then absolute path.
    const std::string path = directory.empty() ? filename : ( fileExists( filename ) ? filename : directory + '/' + filename );

    if( extension == ".dds" )
    {
        return std::make_shared<DDSImageReader>( path, false );
    }

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
