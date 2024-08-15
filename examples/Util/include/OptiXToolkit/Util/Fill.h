// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <algorithm>
#include <vector>

namespace otk {

/// Fill a fixed-length array of Ts with a value of type U that can be converted to T.
template <typename T, size_t N, typename U>
void fill( T ( &ary )[N], U value )
{
    std::fill( std::begin( ary ), std::end( ary ), static_cast<T>( value ) );
}

/// Fill a std::vector<T> with a value of type U that can be converted to T.
template <typename T, typename U>
void fill( std::vector<T>& vec, U value )
{
    std::fill( std::begin( vec ), std::end( vec ), static_cast<T>( value ) );
}

}  // namespace otk
