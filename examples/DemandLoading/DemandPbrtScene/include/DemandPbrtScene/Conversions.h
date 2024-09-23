// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstdlib>

namespace demandPbrtScene {

using uint_t = unsigned int;

inline uint_t toUInt( std::size_t size )
{
    return static_cast<uint_t>( size );
}

template <typename Container>
uint_t containerSize( const Container& container )
{
    return toUInt( container.size() );
}

}  // namespace demandPbrtScene
