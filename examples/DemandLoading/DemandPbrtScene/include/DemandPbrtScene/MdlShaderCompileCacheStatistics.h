// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL

namespace demandPbrtScene {

struct MdlShaderCompileCacheStatistics
{
    unsigned int numMissingShaders{};
    unsigned int numQueuedShaders{};
    unsigned int numCompilingShaders{};
    unsigned int numReadyShaders{};
    unsigned int numFailedShaders{};
};

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
