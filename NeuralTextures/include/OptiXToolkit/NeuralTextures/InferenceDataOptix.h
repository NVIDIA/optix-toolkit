/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <stdint.h>
#include "InferenceConstants.h"

const uint32_t MAX_NTC_SUBTEXTURES = 4;
enum ColorSpaces { CS_LINEAR = 0, CS_SRGB = 1, CS_HLG = 2 };

struct InferenceDataOptix
{
    NtcTextureSetConstants constants;

    // Data needed to decode the latent texture in Optix
    int latentFeatures;
    int latentWidth;
    int latentHeight;
    int numLatentMips;

    // Subtexture info
    int numTextures;
    int numChannels;
    uint8_t texFirstChannel[MAX_NTC_SUBTEXTURES];
    uint8_t texNumChannels[MAX_NTC_SUBTEXTURES];

    // Device-specific data
    CUtexObject latentTexture;
    CUdeviceptr d_mlpWeights;
};
