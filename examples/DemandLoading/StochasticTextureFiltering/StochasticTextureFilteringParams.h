// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

// clang-format off
enum        PixelFilterMode                   {pfBOX=0, pfTENT, pfGAUSSIAN, pfPIXELCENTER, pfSIZE};
const char* PIXEL_FILTER_MODE_NAMES[pfSIZE] = {"Box",   "Tent", "Gaussian", "Pixel Center"};

enum        TextureFilterMode                   {fmPOINT=0, fmLINEAR, fmCUBIC,   fmLANCZOS, fmMITCHELL, fmSIZE};
const char* TEXTURE_FILTER_MODE_NAMES[fmSIZE] = {"Point",   "Linear", "Bicubic", "Lanczos", "Mitchell"};

enum        TextureJitterMode                   {jmNONE=0, jmBOX, jmTENT, jmGAUSSIAN, jmEWA0, jmEXT_ANISOTROPY,
                                                 jmUNSHARPMASK, jmLANCZOS, jmMITCHELL, jmCLANCZOS, jmSIZE};
const char* TEXTURE_JITTER_MODE_NAMES[jmSIZE] = {"None",   "Box", "Tent", "Gaussian", "EWA 0", "Extend Anisotropy",
                                                 "Unsharp Mask", "Lanczos", "Mitchell", "Cylindrical Lanczos"};
// clang-format on

struct StochasticTextureFilteringParams
{
    unsigned int pixelFilterMode;
    unsigned int textureFilterMode;
    unsigned int textureJitterMode;
    float mipScale;
    float textureFilterWidth;
    float textureFilterStrength;
};
