// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/SceneStatistics.h"

#include <cuda.h>

namespace demandPbrtScene {

struct Options;
struct Params;

class Scene
{
  public:
    virtual ~Scene() = default;

    virtual void initialize( CUstream stream ) = 0;

    virtual bool beforeLaunch( CUstream stream, Params& params )      = 0;
    virtual void afterLaunch( CUstream stream, const Params& params ) = 0;

    // one-shot support
    virtual void resolveOneGeometry() = 0;
    virtual void resolveOneMaterial() = 0;

    // stats for nerds
    virtual SceneStatistics getStatistics() const = 0;
};

ScenePtr createScene( const Options& options,
                      PbrtSceneLoaderPtr sceneLoader,
                      DemandTextureCachePtr demandTextureCache,
                      DemandLoaderPtr demandLoader,
                      MaterialResolverPtr materialResolver,
                      GeometryResolverPtr geometryResolver, RendererPtr renderer );

}  // namespace demandPbrtScene
