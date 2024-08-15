// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cuda_runtime.h>

namespace demandPbrtScene {

inline void toInstanceTransform( float ( &dest )[12], const ::pbrt::Transform& source )
{
    const float (&m)[4][4] = source.GetMatrix().m;

    for( int row = 0; row < 3; ++row )
    {
        for( int col = 0; col < 4; ++col )
        {
            dest[row * 4 + col] = m[row][col];
        }
    }
}

inline void toFloat4Transform( float4 ( &dest )[4], const ::pbrt::Transform& source )
{
    const float( &m )[4][4] = source.GetMatrix().m;

    for( int row = 0; row < 4; ++row )
    {
        dest[row] = make_float4( m[row][0], m[row][1], m[row][2], m[row][3] );
    }
}

}
