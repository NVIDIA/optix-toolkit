// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/SceneProxy.h"

#include <vector>

namespace demandPbrtScene {

struct SceneGeometry
{
    GeometryInstance    instance;
    std::vector<uint_t> materialIds;  // one per MaterialGroup
    uint_t              instanceIndex;
};

}  // namespace demandPbrtScene
