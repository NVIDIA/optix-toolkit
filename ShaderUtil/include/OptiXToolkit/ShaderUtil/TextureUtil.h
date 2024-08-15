// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file TextureUtil.h
#include <OptiXToolkit/ShaderUtil/Preprocessor.h>

enum FilterMode { FILTER_POINT=0, FILTER_BILINEAR, FILTER_BICUBIC, FILTER_SMARTBICUBIC };

/// Compute mip level from the texture gradients.
#ifdef __CUDACC__
OTK_INLINE OTK_DEVICE float getMipLevel( float2 ddx, float2 ddy, int texWidth, int texHeight, float invAnisotropy )
{
    ddx = float2{ddx.x * texWidth, ddx.y * texHeight};
    ddy = float2{ddy.x * texWidth, ddy.y * texHeight};

    // Trying to follow CUDA. CUDA performs a low precision EWA filter
    // correction on the texture gradients to determine the mip level.
    // This calculation is described in the Siggraph 1999 paper:
    // Feline: Fast Elliptical Lines for Anisotropic Texture Mapping

    const float A    = ddy.x * ddy.x + ddy.y * ddy.y;
    const float B    = -2.0f * ( ddx.x * ddy.x + ddx.y * ddy.y );
    const float C    = ddx.x * ddx.x + ddx.y * ddx.y;
    const float root = sqrtf( fmaxf( A * A - 2.0f * A * C + C * C + B * B, 0.0f ) );

    // Compute the square of the major and minor ellipse radius lengths to avoid sqrts.
    // Then compensate by taking half the log to get the mip level.

    const float minorRadius2 = ( A + C - root ) * 0.5f;
    const float majorRadius2 = ( A + C + root ) * 0.5f;
    const float filterWidth2 = fmaxf( minorRadius2, majorRadius2 * invAnisotropy * invAnisotropy );
    const float mipLevel     = 0.5f * log2f( filterWidth2 );
    return mipLevel;
}
#endif