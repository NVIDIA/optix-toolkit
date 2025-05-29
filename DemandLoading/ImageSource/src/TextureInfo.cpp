// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/EXRReader.h>

#include <cuda.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace imageSource {

unsigned int getBitsPerChannel( const CUarray_format format )
{
    switch( format )
    {
        case CU_AD_FORMAT_SIGNED_INT8:
        case CU_AD_FORMAT_UNSIGNED_INT8:
            return BITS_PER_BYTE * sizeof( std::int8_t );

        case CU_AD_FORMAT_SIGNED_INT16:
        case CU_AD_FORMAT_UNSIGNED_INT16:
            return BITS_PER_BYTE * sizeof( std::int16_t );

        case CU_AD_FORMAT_SIGNED_INT32:
        case CU_AD_FORMAT_UNSIGNED_INT32:
            return BITS_PER_BYTE * sizeof( std::int32_t );

        case CU_AD_FORMAT_HALF:
            return BITS_PER_BYTE * sizeof( half );

        case CU_AD_FORMAT_FLOAT:
            return BITS_PER_BYTE * sizeof( float );

        case CU_AD_FORMAT_BC1_UNORM:
        case CU_AD_FORMAT_BC1_UNORM_SRGB:
            return 1;

        case CU_AD_FORMAT_BC2_UNORM:
        case CU_AD_FORMAT_BC2_UNORM_SRGB:
        case CU_AD_FORMAT_BC3_UNORM:
        case CU_AD_FORMAT_BC3_UNORM_SRGB:
        case CU_AD_FORMAT_BC6H_UF16:
        case CU_AD_FORMAT_BC6H_SF16:
        case CU_AD_FORMAT_BC7_UNORM:
        case CU_AD_FORMAT_BC7_UNORM_SRGB:
            return 2;

        case CU_AD_FORMAT_BC4_UNORM:
        case CU_AD_FORMAT_BC4_SNORM:
        case CU_AD_FORMAT_BC5_UNORM:
        case CU_AD_FORMAT_BC5_SNORM:
            return 4;

        default:
            OTK_ASSERT_MSG( false, "Invalid CUDA array format" );
            return 0;
    }
}

unsigned int getBitsPerPixel( const TextureInfo& info )
{
    unsigned int numChannels = ( info.numChannels != 3 ) ? info.numChannels : 4;
    return getBitsPerChannel( info.format ) * numChannels;
}

size_t getTextureSizeInBytes( const TextureInfo& info )
{
    size_t texSizeInBytes = static_cast<size_t>( ( getBitsPerPixel( info ) * info.width * info.height ) / BITS_PER_BYTE );
    if( info.numMipLevels > 1 )
        texSizeInBytes = texSizeInBytes * 4ULL / 3ULL;
    return texSizeInBytes;
}

}  // namespace imageSource
