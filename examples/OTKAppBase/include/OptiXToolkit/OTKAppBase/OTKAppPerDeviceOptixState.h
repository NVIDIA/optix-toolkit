// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandLoading
{
    class DemandLoader;
}

namespace otkApp
{

// Store the parts of the OptiX pipeline for one device
struct OTKAppPerDeviceOptixState
{
    int                         device_idx               = -1;
    OptixDeviceContext          context                  = 0;   // OptiX context for this device
    OptixTraversableHandle      gas_handle               = 0;   // Traversable handle for geometry acceleration structure (gas)
    CUdeviceptr                 d_gas_output_buffer      = 0;   // device side buffer for gas
    CUdeviceptr                 d_vertices               = 0;   // vertices
    float3*                     d_normals                = 0;   // normals
    float2*                     d_tex_coords             = 0;   // texture coordinates
    
    OptixModule                 optixir_module           = 0;   // OptiX module stores device code from a file
    OptixPipelineCompileOptions pipeline_compile_options = {};  // Determines how to compile a pipeline
    OptixPipeline               pipeline                 = 0;   // One or more modules linked together for device side programs of the app

    OptixProgramGroup           raygen_prog_group        = 0;   // OptiX raygen programs for the pipeline
    OptixProgramGroup           miss_prog_group          = 0;   // OptiX miss programs for the pipeline
    OptixProgramGroup           hitgroup_prog_group      = 0;   // OptiX hitgroup programs for the pipeline
    
    OptixShaderBindingTable     sbt                      = {};  // Shader binding table
    OTKAppLaunchParams          params                   = {};  // Host-side copy of parameters for OptiX launch
    OTKAppLaunchParams*         d_params                 = nullptr;  // Device-side copy of params
    CUstream                    stream                   = 0;   // Cuda stream where OptiX launches will occur

    std::shared_ptr<demandLoading::DemandLoader> demandLoader;  // Manages demand load requests
    demandLoading::Ticket       ticket;                         // Tracks demand load requests for last OptiX launch
};

// Shader binding table records
template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<OTKAppRayGenData>           RayGenSbtRecord;
typedef SbtRecord<OTKAppMissData>             MissSbtRecord;
typedef SbtRecord<OTKAppHitGroupData>         HitGroupSbtRecord;
typedef SbtRecord<OTKAppTriangleHitGroupData> TriangleHitGroupSbtRecord;

} // namespace otkApp
