// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <string>

namespace demandPbrtScene {

struct SceneStatistics
{
    std::string  fileName;
    double       parseTime;
    unsigned int numFreeShapes;
    unsigned int numObjects;
    unsigned int numObjectShapes;
    unsigned int numObjectInstances;
};

}  // namespace demandPbrtScene
