// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

struct ProxyFactoryStatistics
{
    unsigned int numSceneProxiesCreated;
    unsigned int numShapeProxiesCreated;
    unsigned int numInstanceProxiesCreated;
    unsigned int numInstanceShapeProxiesCreated;
    unsigned int numInstancePrimitiveProxiesCreated;
    unsigned int numGeometryProxiesCreated;
};

}  // namespace demandPbrtScene
