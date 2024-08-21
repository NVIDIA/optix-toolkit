// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/ProxyGranularity.h"
#include "DemandPbrtScene/RenderMode.h"

#include <functional>
#include <string>

#include <OptiXToolkit/ShaderUtil/vec_math.h>

namespace demandPbrtScene {

struct Options
{
    std::string      program;
    std::string      sceneFile;
    std::string      outFile;
    int              width{ 768 };
    int              height{ 512 };
    float3           background{};
    int              warmupFrames{ 0 };
    bool             oneShotGeometry{};
    bool             oneShotMaterial{};
    bool             verboseLoading{};
    bool             verboseProxyGeometryResolution{};
    bool             verboseProxyMaterialResolution{};
    bool             verboseSceneDecomposition{};
    bool             verboseTextureCreation{};
    bool             sortProxies{};
    bool             sync{};
    bool             usePinholeCamera{ true };
    bool             faceForward{};
    bool             debug{};
    bool             oneShotDebug{};
    int2             debugPixel{};
    RenderMode       renderMode{};
    ProxyGranularity proxyGranularity{};
};

using UsageFn = void( const char* program, const char* message );
Options parseOptions( int argc, char* argv[], const std::function<UsageFn>& usage );
Options parseOptions( int argc, char* argv[] );

}  // namespace demandPbrtScene
