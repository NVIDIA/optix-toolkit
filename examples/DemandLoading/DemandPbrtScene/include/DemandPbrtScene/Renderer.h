// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Dependencies.h"

#include <optix.h>

#include <cuda.h>

#include <memory>
#include <vector>

namespace otk {

struct DebugLocation;

}  // namespace otk

namespace demandPbrtScene {

struct LookAtParams;
struct Options;
struct PerspectiveCamera;

class Scene;

class Renderer
{
  public:
    virtual ~Renderer() = default;

    virtual void initialize( CUstream stream ) = 0;
    virtual void cleanup()                     = 0;

    virtual const otk::DebugLocation&          getDebugLocation() const          = 0;
    virtual const LookAtParams&                getLookAt() const                 = 0;
    virtual const PerspectiveCamera&           getCamera() const                 = 0;
    virtual Params&                            getParams()                       = 0;
    virtual OptixDeviceContext                 getDeviceContext() const          = 0;
    virtual const OptixPipelineCompileOptions& getPipelineCompileOptions() const = 0;

    virtual void setDebugLocation( const otk::DebugLocation& data )              = 0;
    virtual void setCamera( const PerspectiveCamera& definition )                = 0;
    virtual void setLookAt( const LookAtParams& lookAt )                         = 0;
    virtual void setProgramGroups( const std::vector<OptixProgramGroup>& value ) = 0;

    virtual void beforeLaunch( CUstream stream )           = 0;
    virtual void launch( CUstream stream, uchar4* pixels ) = 0;
    virtual void afterLaunch()                             = 0;
    virtual void fireOneDebugDump()                        = 0;
    virtual void setClearAccumulator()                     = 0;
};

RendererPtr createRenderer( const Options& options, int numAttributes );

}  // namespace demandPbrtScene
