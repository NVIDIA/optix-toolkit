// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

namespace imageSource {

/// Image info, including dimensions and format.
struct TextureInfo
{
    unsigned int   width;
    unsigned int   height;
    CUarray_format format;
    unsigned int   numChannels;
    unsigned int   numMipLevels;
    bool           isValid;
    bool           isTiled;
};

inline bool isBcFormat( CUarray_format format )
{
    return (format >= CU_AD_FORMAT_BC1_UNORM && format <= CU_AD_FORMAT_BC7_UNORM_SRGB );
}

/// Get sizes of things
const unsigned int BITS_PER_BYTE = 8;
unsigned int getBitsPerChannel( CUarray_format format );
unsigned int getBitsPerPixel( const TextureInfo& info );
size_t getTextureSizeInBytes( const TextureInfo& info );

/// Check equality
inline bool operator==( const TextureInfo& ainfo, const TextureInfo& binfo )
{
    return ainfo.width == binfo.width                   //
           && ainfo.height == binfo.height              //
           && ainfo.format == binfo.format              //
           && ainfo.numChannels == binfo.numChannels    //
           && ainfo.numMipLevels == binfo.numMipLevels  //
           && ainfo.isValid == binfo.isValid            //
           && ainfo.isTiled == binfo.isTiled;
}

/// Check inequeality
inline bool operator!=( const TextureInfo& ainfo, const TextureInfo& binfo )
{
    return !( ainfo == binfo );
}

}  // namespace imageSource
