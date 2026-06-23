// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Util/MipmappedArrayCheck.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <algorithm>
#include <sstream>

namespace demandLoading {

// Bits per pixel, matching imageSource::getBitsPerPixel() semantics (including 3 -> 4 channel promotion).
// Returned in bits (not bytes) so that sub-byte-per-pixel formats such as BC1/BC4 (4 bits/pixel) are not
// truncated to zero; the byte conversion happens once, after multiplying by the texel count.
static size_t bitsPerPixel( const CUDA_ARRAY3D_DESCRIPTOR& desc )
{
    const unsigned int numChannels = ( desc.NumChannels != 3 ) ? desc.NumChannels : 4;
    return static_cast<size_t>( imageSource::getBitsPerChannel( desc.Format ) ) * numChannels;
}

size_t getMipmappedArrayPackedBytes( const CUDA_ARRAY3D_DESCRIPTOR& desc, unsigned int numMipLevels )
{
    const size_t bpp = bitsPerPixel( desc );

    size_t totalBits = 0;
    for( unsigned int level = 0; level < numMipLevels; ++level )
    {
        const size_t levelWidth  = std::max( size_t{ 1 }, desc.Width >> level );
        const size_t levelHeight = std::max( size_t{ 1 }, desc.Height >> level );
        totalBits += bpp * levelWidth * levelHeight;
    }
    return totalBits / imageSource::BITS_PER_BYTE;
}

bool isMipmappedArraySizeSupported( const CUDA_ARRAY3D_DESCRIPTOR& desc, unsigned int numMipLevels, std::string* reason )
{
    const size_t packedBytes = getMipmappedArrayPackedBytes( desc, numMipLevels );
    if( packedBytes <= MAX_MIPMAPPED_ARRAY_BYTES )
        return true;

    if( reason != nullptr )
    {
        std::ostringstream ss;
        ss << "Mipmapped array of " << desc.Width << "x" << desc.Height << " (format 0x" << std::hex << desc.Format
           << std::dec << ", " << desc.NumChannels << " channels, " << numMipLevels << " mip levels) has a packed size of "
           << packedBytes << " bytes, which exceeds the " << MAX_MIPMAPPED_ARRAY_BYTES
           << " byte (4 GiB) device sampling limit. Coarse mip levels of such an array sample as zero (black) on the device.";
        *reason = ss.str();
    }
    return false;
}

void createMipmappedArray( CUmipmappedArray* array, const CUDA_ARRAY3D_DESCRIPTOR* desc, unsigned int numMipLevels )
{
    std::string reason;
    if( !isMipmappedArraySizeSupported( *desc, numMipLevels, &reason ) )
        OTK_ERROR_CHECK_MSG( CUDA_ERROR_INVALID_VALUE, reason.c_str() );

    OTK_ERROR_CHECK( cuMipmappedArrayCreate( array, desc, numMipLevels ) );
}

}  // namespace demandLoading
