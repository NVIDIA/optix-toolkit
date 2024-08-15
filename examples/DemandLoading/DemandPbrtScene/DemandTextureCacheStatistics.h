// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

struct DemandTextureCacheStatistics
{
    unsigned int numDiffuseTexturesCreated;
    unsigned int numAlphaTexturesCreated;
    unsigned int numSkyboxTexturesCreated;
};

}  // namespace demandPbrtScene
