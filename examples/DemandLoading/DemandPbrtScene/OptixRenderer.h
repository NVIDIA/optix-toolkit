// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    bool                            m_clearAccumulator{};
    Accumulator                     m_accumulator;
};

}  // namespace demandPbrtScene
