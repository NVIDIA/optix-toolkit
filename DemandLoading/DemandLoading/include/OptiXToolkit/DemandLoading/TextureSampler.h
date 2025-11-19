// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector_types.h>

#ifndef __CUDACC_RTC__
#include <cuda.h>
#else
using CUtexObject = unsigned long long;
#endif

#include "TextureDescriptor.h"

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

    // Cascaded textures
    unsigned int cascadeLevel : 4;
    unsigned int hasCascade   : 1;
    unsigned int filterMode   : 2;
    unsigned int numChannelTextures : 5;
    unsigned int conservativeFilter : 1;
    unsigned int pad          : 19;

    // Extra data
    CUdeviceptr extraData;
};

// Indexing related to base colors
const unsigned int NUM_PAGES_PER_TEXTURE = 2;
const unsigned long long NO_BASE_COLOR = 0xFFFFFFFFFFFFFFFFULL;

inline bool isBaseColorId( unsigned int pageId, unsigned int maxTextures )
{
    return pageId >= maxTextures;
}
inline unsigned int samplerIdToBaseColorId( unsigned int samplerId, unsigned int maxTextures )
{
    return samplerId + maxTextures;
}
inline unsigned int pageIdToSamplerId( unsigned int pageId, unsigned int maxTextures )
{
    return ( pageId >= maxTextures ) ? pageId - maxTextures : pageId;
}

}  // namespace demandLoading
