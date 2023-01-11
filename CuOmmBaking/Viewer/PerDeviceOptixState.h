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

#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>

namespace ommBakingApp
{

// Store the parts of the OptiX pipeline for one device
struct PerDeviceOptixState
{
    int                         device_idx               = -1;
    OptixDeviceContext          context                  = 0;   // OptiX context for this device
    
    OptixModule                 ptx_module               = 0;   // OptiX module stores device code from a file
    OptixPipelineCompileOptions pipeline_compile_options = {};  // Determines how to compile a pipeline
    OptixPipeline               pipeline                 = 0;   // One or more modules linked together for device side programs of the app
    OptixProgramGroup           raygen_prog_group        = 0;   // OptiX raygen programs for the pipeline
    OptixProgramGroup           miss_prog_group          = 0;   // OptiX miss programs for the pipeline
    OptixProgramGroup           hitgroup_prog_group      = 0;   // OptiX hitgroup programs for the pipeline
    CUstream                    stream                   = 0;   // Cuda stream where OptiX launches will occur
};


} // namespace ommBakingApp
