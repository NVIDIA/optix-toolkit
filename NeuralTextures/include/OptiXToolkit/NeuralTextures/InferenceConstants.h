/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

 #ifndef NTC_INFERENCE_CONSTANTS_H
 #define NTC_INFERENCE_CONSTANTS_H
 
 #ifdef __cplusplus
 #define NTC_UINT unsigned int
 #else
 #define NTC_UINT uint
 #endif
 
 #define NTC_MAX_MIPS        16
 #define NTC_MAX_CHANNELS    16
 #define NTC_MAX_NEURAL_MIPS 8
 
 #define NTC_FEATURES_PER_LAYER 4 // Number of features stored in each layer of the latent texture array (BGRA4 has 4 channels, thus 4 features)
 
 #define NTC_MLP_LAYERS                4  // Total number of layers (matrix multiplication ops) in the MLP. Supported values: 3 or 4
 #define NTC_MLP_FEATURES              16 // Maximum number of features per texel in the latent representation
 #define NTC_MLP_POS_ENC_WAVES         3  // Number of positional encoding waves (frequencies) used for the input coordinates
 #define NTC_MLP_SUPPLEMENTAL_INPUTS   14 // Positional encoding (12 inputs) and the mip level (twice)
 #define NTC_MLP_INPUT_CHANNELS        48 // Size of the MLP input vector = roundup(NTC_MLP_SUPPLEMENTAL_INPUTS + NTC_MLP_FEATURES * 2, 16)
 #define NTC_MLP_HIDDEN0_CHANNELS      64 // Size of the first hidden layer, must be a multiple of 16
 #define NTC_MLP_HIDDEN1_CHANNELS      48 // Size of the second hidden layer, must be a multiple of 16
 #if NTC_MLP_LAYERS >= 4
 #define NTC_MLP_HIDDEN2_CHANNELS      32 // Size of the third hidden layer, must be a multiple of 16
 #else
 #define NTC_MLP_HIDDEN2_CHANNELS      0  // For 3-layer MLPs, there is no third hidden layer
 #endif
 #define NTC_MLP_OUTPUT_CHANNELS       16 // Size of the MLP output vector, must be a multiple of 16
 
 struct NtcColorMipConstants
 {
     int neuralMip;
     float positionLod;
     float positionScale;
     int pad;
 };
 
 struct NtcTextureSetConstants
 {
 #ifdef __cplusplus
     NtcColorMipConstants colorMips[NTC_MAX_MIPS];
     int networkWeightOffsets[4];
     int networkBiasOffsets[4];
     int networkScaleOffsets[4];
 #else
     // These structures are packed as int4 for compatibility with engines
     // that don't support structs in constant buffers.
     int4 colorMips[NTC_MAX_MIPS];
     
     // This maps to the int[] array on the host side
     // but doesn't require 16-byte alignment for each element.
     int4 networkWeightOffsets;
     int4 networkBiasOffsets;
     int4 networkScaleOffsets;
 #endif
 
     int imageWidth;
     int imageHeight;
     int imageMips;    
     int pad0;
 
     NTC_UINT validChannelMask;
     NTC_UINT channelColorSpaces; // Packed with 2 bits for each channel
     int pad1;
     int pad2;
 };
 
 #endif
 