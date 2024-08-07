//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
