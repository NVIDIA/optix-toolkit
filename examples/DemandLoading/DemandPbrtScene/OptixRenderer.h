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

// optix.h uses std::min/std::max
#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "Accumulator.h"
#include "Dependencies.h"
#include "Params.h"
#include "Renderer.h"
#include "Traversable.h"

#include <OptiXToolkit/DemandGeometry/ProxyInstances.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/OptiXMemory/SyncRecord.h>

#include <optix.h>

#include <cuda.h>

#include <memory>
#include <vector>

namespace demandPbrtScene {

struct Options;
class Scene;

class OptixRenderer : public Renderer
{
  public:
    OptixRenderer( const Options& options, int numAttributes );
    ~OptixRenderer() override = default;

    void initialize( CUstream stream ) override;
    void cleanup() override;

    const otk::DebugLocation&          getDebugLocation() const override { return m_params[0].debug; }
    const LookAtParams&                getLookAt() const override { return m_params[0].lookAt; }
    const PerspectiveCamera&           getCamera() const override { return m_params[0].camera; }
    Params&                            getParams() override { return m_params[0]; }
    OptixDeviceContext                 getDeviceContext() const override { return m_context; }
    const OptixPipelineCompileOptions& getPipelineCompileOptions() const override { return m_pipelineCompileOptions; }

    void setDebugLocation( const otk::DebugLocation& value ) override { m_params[0].debug = value; }
    void setCamera( const PerspectiveCamera& value ) override
    {
        setClearAccumulator();
        m_params[0].camera = value;
    }
    void setLookAt( const LookAtParams& value ) override
    {
        setClearAccumulator();
        m_params[0].lookAt = value;
    }
    void setProgramGroups( const std::vector<OptixProgramGroup>& value ) override;

    void beforeLaunch( CUstream stream ) override;
    void launch( CUstream stream, uchar4* image ) override;
    void afterLaunch() override;
    void fireOneDebugDump() override;
    void setClearAccumulator() override { m_clearAccumulator = true; }

  private:
    using uint_t = unsigned int;

    void createOptixContext();
    void initPipelineOpts();
    void initializeParamsFromOptions();
    void createPipeline();
    void writeRayGenRecords( CUstream stream );
    void writeMissRecords( CUstream stream );
    void writeHitGroupRecords( CUstream stream );
    void writeSbt();
    void buildShaderBindingTable( CUstream stream );

    const Options&                  m_options;
    int                             m_numAttributes{};
    OptixDeviceContext              m_context{};
    OptixPipelineCompileOptions     m_pipelineCompileOptions{};
    std::vector<OptixProgramGroup>  m_programGroups;
    OptixPipeline                   m_pipeline{};
    otk::SyncRecord<otk::EmptyData> m_rayGenRecord{ 1 };
    otk::SyncRecord<otk::EmptyData> m_missRecord{ 1 };
    otk::SyncRecord<otk::EmptyData> m_hitGroupRecords{ +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS };
    OptixShaderBindingTable         m_sbt{};
    otk::SyncVector<Params>         m_params{ 1 };
    bool                            m_pipelineChanged{ true };
    bool                            m_sbtChanged{ true };
    bool                            m_fireOneDebugDump{};

    Accumulator                     m_accumulator;
    bool                            m_clearAccumulator;
};

}  // namespace demandPbrtScene
