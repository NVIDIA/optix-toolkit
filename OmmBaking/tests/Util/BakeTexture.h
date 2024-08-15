// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <cstdint>

struct TextureToStateParams
{
    cudaTextureObject_t tex;
    bool                isRgba;
    bool                isNormalizedCoords;

    float               opacityCutoff;
    float               transparencyCutoff;

    unsigned int        width;
    unsigned int        height;
    unsigned int        pitchInBits;

    uint32_t*           buffer;
};

cudaError_t launchTextureToState( const TextureToStateParams params, cudaStream_t stream );
