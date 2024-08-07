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
