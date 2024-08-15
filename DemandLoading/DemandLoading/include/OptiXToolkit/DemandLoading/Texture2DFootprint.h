// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandLoading {

/// Texture2DFootprint is binary compatible with the uint4 returned by the texture footprint intrinsics.
///
/// See optixTexFootprint2DGrad (etc.) in the OptiX API documentation.
/// (https://raytracing-docs.nvidia.com/optix7/api/html/index.html)
// clang-format off
struct Texture2DFootprint
{
    unsigned long long mask;             ///< Toroidally rotated 8x8 texel group mask to store footprint coverage
    unsigned int       tileY : 12;       ///< Y position of anchor tile. Tiles are 8x8 blocks of texel groups.
    unsigned int       reserved1 : 4;    ///< not used
    unsigned int       dx : 3;           ///< X rotation of mask relative to anchor tile. Mask starts at 8*tileX-dx in texel group coordinates.
    unsigned int       dy : 3;           ///< Y rotation of mask relative to anchor tile. Mask starts at 8*tileY-dy in texel group coordinates.
    unsigned int       reserved2 : 2;    ///< not used
    unsigned int       granularity : 4;  ///< enum giving texel group size. 0 indicates "same size as requested"
    unsigned int       reserved3 : 4;    ///< not used
    unsigned int       tileX : 12;       ///< X position of anchor tile
    unsigned int       level : 4;        ///< mip level
    unsigned int       reserved4 : 16;   ///< not used
};
// clang-format on

} // namespace demandLoading
