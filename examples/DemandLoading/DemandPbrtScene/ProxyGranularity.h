// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

enum class ProxyGranularity
{
    NONE   = 0,
    FINE   = 1,
    COARSE = 2
};

inline int operator+( ProxyGranularity value )
{
    return static_cast<int>( value );
}

}  // namespace demandPbrtScene
