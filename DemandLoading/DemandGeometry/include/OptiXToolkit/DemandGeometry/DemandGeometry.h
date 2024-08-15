// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>

#include <optix.h>

#include <vector_types.h>

namespace demandGeometry {

using uint_t = unsigned int;

struct Context
{
    const OptixAabb* proxies;
};

namespace app {

__device__ void     reportClosestHitNormal( float3 ffNormal );
__device__ Context& getContext();
__device__ const demandLoading::DeviceContext& getDeviceContext();

}  // namespace app
}  // namespace demandGeometry
