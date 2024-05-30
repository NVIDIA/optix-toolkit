
//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
