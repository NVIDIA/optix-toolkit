// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file TextureDescriptor.h
/// TextureDescriptor specifies address mode, filter mode, etc.

#include <cuda.h>
#include <OptiXToolkit/ShaderUtil/TextureUtil.h>

namespace demandLoading {

/// TextureDescriptor specifies the address mode (e.g. wrap vs. clamp), filter mode (point vs. linear), etc.
struct TextureDescriptor
{
    /// Address mode (e.g. wrap)
    CUaddress_mode addressMode[2]{ CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP };

    /// Filter mode (point, bilinear, bicubic, smartbicubic)
    unsigned int filterMode = FILTER_SMARTBICUBIC;

    /// Filter mode between miplevels (linear or point)
    CUfilter_mode mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    /// Maximum anisotropy.  A value of 1 disables anisotropic filtering.
    unsigned int maxAnisotropy = 16;

    /// CUDA texture flags.  Use 0 to enable trilinear optimization (off by default).
    unsigned int flags = CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION;

    // Apply conservative filtering (overblur to prevent aliasing for very anisotropic samples).
    bool conservativeFilter = true;
};

inline CUfilter_mode toCudaFilterMode( unsigned int mode )
{
    return ( mode == FILTER_POINT ) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
}

inline bool operator==( const TextureDescriptor& adesc, const TextureDescriptor& bdesc )
{
    return adesc.addressMode[0] == bdesc.addressMode[0]         //
           && adesc.addressMode[1] == bdesc.addressMode[1]      //
           && adesc.filterMode == bdesc.filterMode              //
           && adesc.mipmapFilterMode == bdesc.mipmapFilterMode  //
           && adesc.maxAnisotropy == bdesc.maxAnisotropy        //
           && adesc.flags == bdesc.flags                        //
           && adesc.conservativeFilter == bdesc.conservativeFilter;
}

inline bool operator!=( const TextureDescriptor& lhs, const TextureDescriptor& rhs )
{
    return !( lhs == rhs );
}

}  // namespace demandLoading
