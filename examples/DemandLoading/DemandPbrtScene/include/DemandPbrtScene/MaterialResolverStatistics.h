// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

struct MaterialResolverStats
{
    unsigned int numPartialMaterialsRealized;
    unsigned int numMaterialsRealized;
    unsigned int numMaterialsReused;
    unsigned int numProxyMaterialsCreated;
};

}
