//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <optix.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

#define VISIBILITY_MASK_OMM_ENABLED  1
#define VISIBILITY_MASK_OMM_DISABLED 2

struct RenderOptions
{
    float2   windowMin, windowMax;
    bool     validate_opacity;
    bool     opacity_shading;
    uint32_t textureLayer;
    bool     force2state;
};

struct Params
{
    uchar3*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    OptixTraversableHandle handle;
    RenderOptions          options;
    uint32_t*              error_count;
};

struct HitTextureData
{
    cudaTextureObject_t               tex = {};
    cudaTextureReadMode               readMode;
    cuOmmBaking::CudaTextureAlphaMode alphaMode;
    cudaChannelFormatDesc             chanDesc;
    float                             transparencyCutoff;
    float                             opacityCutoff;
};

struct HitSbtData
{
    HitTextureData texture;

    cuOmmBaking::BakeInputDesc desc;
};
