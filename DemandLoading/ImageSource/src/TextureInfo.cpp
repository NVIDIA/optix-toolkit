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

unsigned int getBytesPerChannel( const CUarray_format format )
{
    switch( format )
    {
        case CU_AD_FORMAT_SIGNED_INT8:
        case CU_AD_FORMAT_UNSIGNED_INT8:
            return sizeof( std::int8_t );

        case CU_AD_FORMAT_SIGNED_INT16:
        case CU_AD_FORMAT_UNSIGNED_INT16:
            return sizeof( std::int16_t );

        case CU_AD_FORMAT_SIGNED_INT32:
        case CU_AD_FORMAT_UNSIGNED_INT32:
            return sizeof( std::int32_t );

        case CU_AD_FORMAT_HALF:
            return sizeof( half );

        case CU_AD_FORMAT_FLOAT:
            return sizeof( float );

        default:
            OTK_ASSERT_MSG( false, "Invalid CUDA array format" );
            return 0;
    }
}

size_t getTextureSizeInBytes( const TextureInfo& info )
{
    size_t texSize = static_cast<size_t>( getBytesPerChannel( info.format ) ) * info.numChannels * info.width * info.height;
    if( info.numMipLevels > 1 )
        texSize = texSize * 4ULL / 3ULL;
    return texSize;
}

}  // namespace imageSource
