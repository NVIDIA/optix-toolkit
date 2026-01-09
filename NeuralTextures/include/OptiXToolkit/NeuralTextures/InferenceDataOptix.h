/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
