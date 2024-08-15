// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "GeometryCacheStatistics.h"
#include "GeometryResolverStatistics.h"
#include "ImageSourceFactoryStatistics.h"
#include "MaterialResolverStatistics.h"
#include "ProxyFactoryStatistics.h"
#include "SceneStatistics.h"

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
