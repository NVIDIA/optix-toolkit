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

namespace demandLoading
{
    class DemandLoader;
}

namespace demandTextureApp
{

// Store the parts of the OptiX pipeline for one device
struct PerDeviceOptixState
{
    int                         device_idx               = -1;
    OptixDeviceContext          context                  = 0;   // OptiX context for this device
    OptixTraversableHandle      gas_handle               = 0;   // Traversable handle for geometry acceleration structure (gas)
    CUdeviceptr                 d_gas_output_buffer      = 0;   // device side buffer for gas
    CUdeviceptr                 d_vertices               = 0;   // vertices
    float3*                     d_normals                = 0;   // normals
    float2*                     d_tex_coords             = 0;   // texture coordinates
    
    OptixModule                 ptx_module               = 0;   // OptiX module stores device code from a file
    OptixPipelineCompileOptions pipeline_compile_options = {};  // Determines how to compile a pipeline
    OptixPipeline               pipeline                 = 0;   // One or more modules linked together for device side programs of the app

    OptixProgramGroup           raygen_prog_group        = 0;   // OptiX raygen programs for the pipeline
    OptixProgramGroup           miss_prog_group          = 0;   // OptiX miss programs for the pipeline
    OptixProgramGroup           hitgroup_prog_group      = 0;   // OptiX hitgroup programs for the pipeline
    
    OptixShaderBindingTable     sbt                      = {};  // Shader binding table
    Params                      params                   = {};  // Host-side copy of parameters for OptiX launch
    Params*                     d_params                 = nullptr;  // Device-side copy of params
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

typedef SbtRecord<RayGenData>           RayGenSbtRecord;
typedef SbtRecord<MissData>             MissSbtRecord;
typedef SbtRecord<HitGroupData>         HitGroupSbtRecord;
typedef SbtRecord<TriangleHitGroupData> TriangleHitGroupSbtRecord;

} // namespace demandTextureApp
