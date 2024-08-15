// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstddef>

namespace demandLoading {

/// Return ceil(x/y) for integers x and y
template <typename IntT, typename IntT2>
inline IntT idivCeil( IntT x, IntT2 y )
{
    return ( x + y - 1 ) / y;
}


/// Pad the given size to the specified alignment.
inline size_t align( size_t size, size_t alignment )
{
    if( size % alignment != 0 )
        size += alignment - ( size % alignment );
    return size;
}

}  // namespace demandLoading
