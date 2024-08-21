// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/GeometryCacheStatistics.h"
#include "DemandPbrtScene/GeometryResolverStatistics.h"
#include "DemandPbrtScene/ImageSourceFactoryStatistics.h"
#include "DemandPbrtScene/MaterialResolverStatistics.h"
#include "DemandPbrtScene/ProxyFactoryStatistics.h"
#include "DemandPbrtScene/SceneStatistics.h"

namespace demandPbrtScene {

struct UserInterfaceStatistics
{
    unsigned int                 numFramesRendered;
    GeometryCacheStatistics      geometryCache;
    ImageSourceFactoryStatistics imageSourceFactory;
    ProxyFactoryStatistics       proxyFactory;
    GeometryResolverStatistics   geometry;
    MaterialResolverStats        materials;
    SceneStatistics              scene;
};

}  // namespace demandPbrtScene
