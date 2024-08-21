// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/GeometryResolverStatistics.h"

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>

#include <cuda.h>

#include <memory>
#include <optix_types.h>

namespace demandPbrtScene {

class FrameStopwatch;
struct SceneSyncState;

class GeometryResolver
{
  public:
    virtual ~GeometryResolver() = default;

    virtual void initialize( CUstream stream, OptixDeviceContext context, const SceneDescriptionPtr& scene, SceneSyncState& sync ) = 0;

    virtual demandGeometry::Context getContext() const = 0;

    virtual void resolveOneGeometry() = 0;

    virtual bool resolveRequestedProxyGeometries( CUstream              stream,
                                                  OptixDeviceContext    context,
                                                  const FrameStopwatch& frameTime,
                                                  SceneSyncState&       sync ) = 0;

    virtual GeometryResolverStatistics getStatistics() const = 0;
};

GeometryResolverPtr createGeometryResolver( const Options&        options,
                                            ProgramGroupsPtr      programGroups,
                                            GeometryLoaderPtr     geometryLoader,
                                            ProxyFactoryPtr       proxyFactory,
                                            DemandTextureCachePtr demandTextureCache,
                                            MaterialResolverPtr   materialResolver );

}  // namespace demandPbrtScene
