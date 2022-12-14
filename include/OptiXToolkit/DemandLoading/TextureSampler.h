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

#pragma once

#include <vector_types.h>

#ifndef __CUDACC_RTC__
#include <cuda.h>
#else
using CUtexObject = unsigned long long;
#endif

namespace demandLoading {

const unsigned int MAX_TILE_LEVELS = 9;

/// Device-side texture info.
struct TextureSampler
{
    CUtexObject texture;

    // Description for the sampler.  This struct must agree with the
    // corresponding struct in optix, TextureInfo in TextureFootprint.h
    struct Description
    {
        unsigned int isInitialized : 1;
        unsigned int reserved1 : 2;
        unsigned int numMipLevels : 5;
        unsigned int logTileWidth : 4;
        unsigned int logTileHeight : 4;
        unsigned int reserved2 : 4;
        unsigned int isSparseTexture : 1;
        unsigned int isUdimBaseTexture : 1;
        unsigned int wrapMode0 : 2;
        unsigned int wrapMode1 : 2;
        unsigned int mipmapFilterMode : 1;
        unsigned int maxAnisotropy : 5;
    } desc;

    // Texture dimensions
    unsigned int width;
    unsigned int height;
    unsigned int mipTailFirstLevel;

    // Virtual addressing
    unsigned int startPage;
    unsigned int numPages;

    // Precomputed values for each level
    struct MipLevelSizes
    {
        unsigned int   mipLevelStart;
        unsigned short levelWidthInTiles;
        unsigned short levelHeightInTiles;
    };
    MipLevelSizes mipLevelSizes[MAX_TILE_LEVELS];

    // Udim textures
    unsigned int udimStartPage;
    unsigned short udim;
    unsigned short vdim;
};

}  // namespace demandLoading
