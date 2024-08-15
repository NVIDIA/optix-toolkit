// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>

#include <optix.h>

namespace ommBakingApp
{

// Store the parts of the OptiX pipeline for one device
struct PerDeviceOptixState
{
    int                         device_idx               = -1;
    OptixDeviceContext          context                  = 0;   // OptiX context for this device
    
    OptixModule                 optixir_module           = 0;   // OptiX module stores device code from a file
    OptixPipelineCompileOptions pipeline_compile_options = {};  // Determines how to compile a pipeline
    OptixPipeline               pipeline                 = 0;   // One or more modules linked together for device side programs of the app
    OptixProgramGroup           raygen_prog_group        = 0;   // OptiX raygen programs for the pipeline
    OptixProgramGroup           miss_prog_group          = 0;   // OptiX miss programs for the pipeline
    OptixProgramGroup           hitgroup_prog_group      = 0;   // OptiX hitgroup programs for the pipeline
    CUstream                    stream                   = 0;   // Cuda stream where OptiX launches will occur
};


} // namespace ommBakingApp
