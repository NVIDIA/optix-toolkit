// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

#include <stdint.h>

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
