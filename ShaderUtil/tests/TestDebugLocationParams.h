// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ShaderUtil/DebugLocation.h>

#include <vector_types.h>

using uint_t = unsigned int;

namespace otk {
namespace shaderUtil {
namespace testing {

struct Params
{
    uint_t        width;
    uint_t        height;
    float3*       image;
    float3        miss;
    uint_t*       dumpIndicator;
    DebugLocation debug;
};

}  // namespace testing
}  // namespace shaderUtil
}  // namespace otk
