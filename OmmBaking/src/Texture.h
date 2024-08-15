// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

struct StateTextureConfig
{
    uint32_t width;
    uint32_t height;

    uint32_t pitchInBits;   
};

cudaError_t launchSummedAreaTable(
    void* d_temp_storage,
    size_t& temp_storage_bytes,    
    StateTextureConfig config,
    const uint8_t* input,
    uint2* outputSat,    
    cudaStream_t stream );

struct CudaTextureConfig
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;

    float    opacityCutoff;
    float    transparencyCutoff;

    cuOmmBaking::CudaTextureAlphaMode  alphaMode;

    cudaChannelFormatDesc chanDesc;
    cudaTextureDesc       texDesc;
};

cudaError_t launchSummedAreaTable(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    CudaTextureConfig config,
    cudaTextureObject_t inputTexture,
    uint2* outputSat,    
    cudaStream_t stream );
