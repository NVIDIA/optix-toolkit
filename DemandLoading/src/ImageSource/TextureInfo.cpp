
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

#include "Exception.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <ImageSource/TextureInfo.h>

namespace imageSource {

unsigned int getBytesPerChannel( const CUarray_format format )
{
    switch( format )
    {
        case CU_AD_FORMAT_SIGNED_INT8:
        case CU_AD_FORMAT_UNSIGNED_INT8:
            return 1;

        case CU_AD_FORMAT_SIGNED_INT16:
        case CU_AD_FORMAT_UNSIGNED_INT16:
            return 2;

        case CU_AD_FORMAT_SIGNED_INT32:
        case CU_AD_FORMAT_UNSIGNED_INT32:
            return 4;

        case CU_AD_FORMAT_HALF:
            return sizeof( half );

        case CU_AD_FORMAT_FLOAT:
            return sizeof( float );

        default:
            DEMAND_ASSERT_MSG( false, "Invalid CUDA array format" );
            return 0;
    }

    DEMAND_ASSERT_MSG( false, "Invalid CUDA array format" );
    return 0;
}

size_t getTextureSizeInBytes( const TextureInfo& info )
{
    size_t texSize = static_cast<size_t>( getBytesPerChannel( info.format ) ) * info.numChannels * info.width * info.height;
    if( info.numMipLevels > 1 )
        texSize = texSize * 4ULL / 3ULL;
    return texSize;
}

}  // namespace imageSource
