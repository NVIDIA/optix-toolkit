// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

enum class GeometryPrimitive : int
{
    NONE     = 0,
    TRIANGLE = 1,
    SPHERE   = 2,
};

inline int operator+( GeometryPrimitive value )
{
    return static_cast<int>( value );
}

}  // namespace demandPbrtScene
