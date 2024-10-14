// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/Paging.h>

#include <cassert>

namespace demandMaterial {

namespace app {

__device__ unsigned int getMaterialId();
__device__ void         reportClosestHit( unsigned int pageId, bool isResident );
__device__ const demandLoading::DeviceContext& getDeviceContext();

}  // namespace app

extern "C" __global__ void __closesthit__proxyMaterial()
{
    const unsigned int pageId = app::getMaterialId();
    bool               isResident{};
#ifndef NDEBUG
    assert( pageId > 0 );
#endif
    const unsigned long long pageTableEntry = demandLoading::pagingMapOrRequest( app::getDeviceContext(), pageId, &isResident );
    app::reportClosestHit( pageId, isResident );
}

}  // namespace demandMaterial
