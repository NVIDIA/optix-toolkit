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

#include "DemandPbrtScene/Accumulator.h"
#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Renderer.h"

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
    void setCallableProgramGroups( const std::vector<OptixProgramGroup>& value ) override;
#ifdef OTK_USE_MDL
    void setPipelineState( OptixPipeline                         pipeline,
                           const std::vector<OptixProgramGroup>& programGroups,
                           const std::vector<OptixProgramGroup>& callableProgramGroups ) override;
#endif

    void beforeLaunch( CUstream stream ) override;
    void launch( CUstream stream, uchar4* image ) override;
    void afterLaunch() override;
    void fireOneDebugDump() override;
    void setClearAccumulator() override { m_clearAccumulator = true; }

  private:
    using uint_t = unsigned int;

    struct PipelineState
    {
        PipelineState( OptixPipeline pipeline,
                       std::vector<OptixProgramGroup> programGroups,
                       std::vector<OptixProgramGroup> callableProgramGroups );
        ~PipelineState();

        PipelineState( const PipelineState& )            = delete;
        PipelineState& operator=( const PipelineState& ) = delete;

        void recordLaunchCompleteEvent( CUstream stream );
        bool launchComplete() const;
        void waitForLaunchComplete() const;

        OptixPipeline                   pipeline{};
        CUevent                         launchCompleteEvent{};
        std::vector<OptixProgramGroup>  programGroups;
        std::vector<OptixProgramGroup>  callableProgramGroups;
        otk::SyncRecord<otk::EmptyData> rayGenRecord;
        otk::SyncRecord<otk::EmptyData> missRecord;
        otk::SyncRecord<otk::EmptyData> hitGroupRecords;
        otk::SyncRecord<otk::EmptyData> callableRecords;
        OptixShaderBindingTable         sbt{};
    };

    using PipelineStatePtr = std::shared_ptr<PipelineState>;

    void createOptixContext();
    void initPipelineOpts();
    void initializeParamsFromOptions();
    PipelineStatePtr createPipelineState();
    void writeRayGenRecords( CUstream stream, PipelineState& state );
    void writeMissRecords( CUstream stream, PipelineState& state );
    void writeHitGroupRecords( CUstream stream, PipelineState& state );
    void writeCallableRecords( CUstream stream, PipelineState& state );
    void writeSbt( PipelineState& state );
    void buildShaderBindingTable( CUstream stream, PipelineState& state );
    void activatePendingPipelineState();
    void retirePipelineState( PipelineStatePtr state );
    void collectRetiredPipelineStates();
    void waitForPipelineStates();

    const Options&                  m_options;
    int                             m_numAttributes{};
    OptixDeviceContext              m_context{};
    OptixPipelineCompileOptions     m_pipelineCompileOptions{};
    std::vector<OptixProgramGroup>  m_programGroups;
    std::vector<OptixProgramGroup>  m_callableProgramGroups;
    PipelineStatePtr                m_activeState;
    PipelineStatePtr                m_pendingState;
    std::vector<PipelineStatePtr>   m_retiredStates;
    otk::SyncVector<Params>         m_params{ 1 };
    bool                            m_pipelineChanged{ true };
    bool                            m_sbtChanged{ true };
    bool                            m_fireOneDebugDump{};
    bool                            m_clearAccumulator{};
    Accumulator                     m_accumulator;
};

}  // namespace demandPbrtScene
