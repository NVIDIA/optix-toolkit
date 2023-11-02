//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

const int SUBFRAME_ID  = 0;
const int PIXEL_FILTER_ID = 1;
const int TEXTURE_FILTER_ID = 2;
const int MOUSEX_ID = 3;
const int MOUSEY_ID = 4;

const int MIP_SCALE_ID = 0;
const int TEXTURE_FILTER_WIDTH_ID = 1;
const int TEXTURE_FILTER_STRENGTH_ID = 2;

enum FilteModes
{
    POINT = 0,
    EWA,
    EWA_JITTER,
    BICUBIC,
    STOCHASTIC_EWA,
    EXTENDED_ANISOTROPY
};

// clang-format off
enum        PixelFilterMode                   {pfBOX=0, pfTENT, pfGAUSSIAN, pfPIXELCENTER, pfSIZE};
const char* PIXEL_FILTER_MODE_NAMES[pfSIZE] = {"Box",   "Tent", "Gaussian", "Pixel Center"};

enum TextureFilterMode {
    tfPOINT=0, tfBOX_POINT, tfTENT_POINT, tfGAUSSIAN_POINT,
    tfTRILINEAR, tfCUBIC, tfBOX_LINEAR, tfTENT_LINEAR, tfGAUSSIAN_LINEAR, tfUNSHARP_LINEAR, tfSIZE};

const char* TEXTURE_FILTER_MODE_NAMES[tfSIZE] = {
    "Point", "Box Point", "Tent Point", "Gaussian Point",
    "Trilinear", "Cubic", "Box Linear", "Tent Linear", "Gaussian Linear", "Unsharp Mask Linear"};
// clang-format on