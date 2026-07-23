// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL
#include "DemandPbrtScene/MdlShaderCompileCacheStatistics.h"
#endif

namespace demandPbrtScene {

struct MaterialResolverStats
{
    unsigned int                    numPartialMaterialsRealized;
    unsigned int                    numMaterialsRealized;
    unsigned int                    numMaterialsReused;
    unsigned int                    numProxyMaterialsCreated;
    unsigned int                    numRequestedMaterialPages;
#ifdef OTK_USE_MDL
    unsigned int                    numMdlFallbackShaders;
    unsigned int                    numGeneratedMdlMaterialCompileRequests;
    MdlShaderCompileCacheStatistics mdlShaders;
#endif
};

}  // namespace demandPbrtScene
