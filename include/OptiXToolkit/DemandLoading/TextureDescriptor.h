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

/// \file TextureDescriptor.h
/// TextureDescriptor specifies address mode, filter mode, etc.

#include <cuda.h>

namespace demandLoading {

/// TextureDescriptor specifies the address mode (e.g. wrap vs. clamp), filter mode (point vs. linear), etc.
struct TextureDescriptor
{
    /// Address mode (e.g. wrap)
    CUaddress_mode addressMode[2] = {CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP};

    /// Filter mode (e.g. linear vs. point)
    CUfilter_mode filterMode = CU_TR_FILTER_MODE_LINEAR;

    /// Filter mode between miplevels (e.g. linear vs. point)
    CUfilter_mode mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    /// Maximum anisotropy.   A value of 1 disables anisotropic filtering.
    unsigned int maxAnisotropy = 16;

    /// CUDA texture flags.  Use 0 to enable trilinear optimization (off by default).
    unsigned int flags = CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION;
};

}  // namespace demandLoading
