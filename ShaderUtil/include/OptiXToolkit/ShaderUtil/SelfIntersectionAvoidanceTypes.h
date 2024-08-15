// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file SelfIntersectionAvoidanceTypes.h
///
///

#include <OptiXToolkit/ShaderUtil/Preprocessor.h>

#include <optix.h>

namespace SelfIntersectionAvoidance {

struct Matrix3x4
{
    float4 row0;
    float4 row1;
    float4 row2;
};

} // SelfIntersectionAvoidance
