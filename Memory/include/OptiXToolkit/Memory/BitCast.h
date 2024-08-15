// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstring>

namespace otk {

/// Safely convert from one type to another
/// Like reinterpret_cast, but conforms to the requirements of the C++ standard.
/// @tparam T       The destination type.
/// @tparam U       The source type.  Deduced from the type of the argument.
/// @param value    The value of type U to be converted to type T.
template <typename T, typename U>
T bit_cast( U value )
{
    static_assert( sizeof( T ) == sizeof( U ), "Source and destination types are not the same size" );
    T result;
    std::memcpy( &result, &value, sizeof( T ) );
    return result;
}

}  // namespace otk
