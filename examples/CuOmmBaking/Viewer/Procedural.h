// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "Util/interval_math.h"

#include <OptiXToolkit/ShaderUtil/vec_math.h>

template <typename T>
__device__ __forceinline__ T eval_procedural( vec2<T> uv )
{
    using namespace otk;
    return clamp( 2.f * cosf( length( uv - make_float2( 0.5f, 0.5f ) ) * 4.f * M_PIf ), 0.f, 1.f );
}
