// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

enum class RenderMode
{
    PRIMARY_RAY = 0,
    NEAR_AO,
    DISTANT_AO,
    PATH_TRACING
};

inline int operator+( RenderMode value )
{
    return static_cast<int>( value );
}

}  // namespace demandPbrtScene
