// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/TextureSampler.h>

#include <cuda.h>
#include <cuda_runtime.h>  // for uchar4

const unsigned int MIN_TILE_WIDTH               = 64;
const unsigned int MAX_MIP_LEVEL_WIDTH_IN_TILES = 64;
const unsigned int MAX_PAGES_PER_MIP_LEVEL      = MAX_MIP_LEVEL_WIDTH_IN_TILES * MAX_MIP_LEVEL_WIDTH_IN_TILES;
const unsigned int MAX_TEXTURE_DIM              = MIN_TILE_WIDTH * MAX_MIP_LEVEL_WIDTH_IN_TILES;

const unsigned int REQUEST_FINE   = 0;
const unsigned int REQUEST_COARSE = 1;
const unsigned int REQUEST_BOTH   = 2;

struct FootprintInputs
{
    float x;
    float y;
    float level;   // for fooprint2DLod
    float dPdx_x;  // for footprint2DGrad
    float dPdx_y;
    float dPdy_x;
    float dPdy_y;

    FootprintInputs( float _x, float _y, float _level = 0.f, float _dPdx_x = 0.f, float _dPdx_y = 0.f, float _dPdy_x = 0.f, float _dPdy_y = 0.f )
        : x( _x )
        , y( _y )
        , level( _level )
        , dPdx_x( _dPdx_x )
        , dPdx_y( _dPdx_y )
        , dPdy_x( _dPdy_x )
        , dPdy_y( _dPdy_y )
    {
    }
};

struct MipLevelSizes
{
    unsigned int   mipLevelStart;
    unsigned short levelWidthInTiles;
    unsigned short levelHeightInTiles;
};

struct Params
{
    cudaTextureObject_t           texture;
    demandLoading::TextureSampler sampler;
    unsigned int*                 referenceBits;
    unsigned int*                 residenceBits;

    FootprintInputs* inputs;   // One per launch index.
    uint4*           outputs;  // One per launch index.
};

struct RayGenData
{
    float r, g, b;
};
