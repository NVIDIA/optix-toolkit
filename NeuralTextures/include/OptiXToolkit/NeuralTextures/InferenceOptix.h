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

#include <cuda.h>
#include <optix.h>
#include <optix_device.h>

#include "InferenceConstants.h"

constexpr OptixCoopVecMatrixLayout MAT_LAYOUT = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;

__device__ __forceinline__ float fracf( float x ) { return x - floorf( x ); }

// Half bytes
__device__ __forceinline__ int hb0( int a ) { return a & 0xf; }
__device__ __forceinline__ int hb1( int a ) { return (a >> 4) & 0xf; }
__device__ __forceinline__ int hb2( int a ) { return (a >> 8) & 0xf; }
__device__ __forceinline__ int hb3( int a ) { return (a >> 12) & 0xf; }

template<class vecT>
__device__ __forceinline__ 
void lerp4latents( vecT& out, int idx, uint16_t s00, uint16_t s10, uint16_t s01, uint16_t s11, float w00, float w10, float w01, float w11 )
{
    // Scale and bias to convert half bytes with range [0 .. 15] to [-1.0 .. 1.0]
    const float scale = 2.0f / 15.0f;
    const float bias = -1.0f;

    out[idx]     = scale * ( w00 * hb0(s00) + w10 * hb0(s10) + w01 * hb0(s01) + w11 * hb0(s11) ) + bias;
    out[idx + 1] = scale * ( w00 * hb1(s00) + w10 * hb1(s10) + w01 * hb1(s01) + w11 * hb1(s11) ) + bias;
    out[idx + 2] = scale * ( w00 * hb2(s00) + w10 * hb2(s10) + w01 * hb2(s01) + w11 * hb2(s11) ) + bias;
    out[idx + 3] = scale * ( w00 * hb3(s00) + w10 * hb3(s10) + w01 * hb3(s01) + w11 * hb3(s11) ) + bias;
}


template<class vecT>
__device__ __forceinline__
void SampleLatentGrid(
    CUtexObject latentTexture,
    float2 uv,
    float neuralLod,
    float2 neuralMipSize,
    int outputOffset,
    int numLatentFeatures,
    vecT& outputArray )
{
    // Sample and interpolate half bytes stored in ushort4 textures, since CUDA
    // does not currently support bgra4 textures.

    // Move samples from pixel centers to pixel corners
    float dx = 1.0f / neuralMipSize.x;
    float dy = 1.0f / neuralMipSize.y;
    uv.x -= dx * 0.5f;
    uv.y -= dy * 0.5f;

    // Separate coordinates into pixel corner (x,y), and fractional part (wx, wy)
    float x = ( floorf( uv.x * neuralMipSize.x ) ) * dx;
    float y = ( floorf( uv.y * neuralMipSize.y ) ) * dy;
    float wx = ( uv.x - x ) * neuralMipSize.x;
    float wy = ( uv.y - y ) * neuralMipSize.y;

    // Compute bilinear weights
    float w00 = (1.0f - wx) * (1.0f - wy);
    float w10 = wx          * (1.0f - wy);
    float w01 = (1.0f - wx) * wy;
    float w11 = wx          * wy;

    // Move (x,y) from pixel corner to pixel center
    x += dx * 0.5f;
    y += dy * 0.5f;
    
    // Read 4 latents, and interpolate the features held in them
    ushort4 s00 = tex2DLod<ushort4>( latentTexture, x,    y,    neuralLod );
    ushort4 s10 = tex2DLod<ushort4>( latentTexture, x+dx, y,    neuralLod );
    ushort4 s01 = tex2DLod<ushort4>( latentTexture, x,    y+dy, neuralLod );
    ushort4 s11 = tex2DLod<ushort4>( latentTexture, x+dx, y+dy, neuralLod );

    lerp4latents( outputArray, outputOffset, s00.x, s10.x, s01.x, s11.x, w00, w10, w01, w11 );

    if( numLatentFeatures > 4 )
        lerp4latents( outputArray, outputOffset + 4, s00.y, s10.y, s01.y, s11.y, w00, w10, w01, w11 );

    if( numLatentFeatures > 8 )
        lerp4latents( outputArray, outputOffset + 8, s00.z, s10.z, s01.z, s11.z, w00, w10, w01, w11 );

    if( numLatentFeatures > 12 )
        lerp4latents( outputArray, outputOffset + 12, s00.w, s10.w, s01.w, s11.w, w00, w10, w01, w11 );
}


template<class vecT>
__device__ __forceinline__
void EncodeSamplePosition( float2 posf, float lod, int offset, vecT& outputArray )
{
    int idx = offset;

#pragma unroll
    for (int wave = 0; wave < NTC_MLP_POS_ENC_WAVES; ++wave)
    {
        outputArray[idx + 0] = half( fracf(posf.x) * 2 - 1 );
        outputArray[idx + 1] = half( fracf(posf.y) * 2 - 1 );
        outputArray[idx + 2] = half( fracf(posf.x + 0.25f) * 2 - 1 );
        outputArray[idx + 3] = half( fracf(posf.y + 0.25f) * 2 - 1 );

        idx += 4;
        posf.x *= 2.f;
        posf.y *= 2.f;
    }

    outputArray[idx + 0] = half( lod );
    outputArray[idx + 1] = half( lod );
}


template<class vecT>
__device__ __forceinline__
void PrepareNetworkInputs(
    const NtcTextureSetConstants& constants,
    CUtexObject latentTexture,
    int numLatentFeatures,
    int latentWidth,
    int latentHeight,
    float2 uv,
    int2 texel,
    int mipLevel,
    vecT& networkInputs)
{
    // Zero init the array - in some cases, OUTPUT_SIZE is rounded up from the actual used size.
    networkInputs = vecT(0);
    
    // Find the neural mip level and sample the latent grids
    const NtcColorMipConstants& colorMip = constants.colorMips[mipLevel];
    const int neuralMip = colorMip.neuralMip;

    float2 neuralMipSize = make_float2( latentWidth >> neuralMip, latentHeight >> neuralMip );
    SampleLatentGrid<vecT>( latentTexture, uv, neuralMip + 0, neuralMipSize, 0, numLatentFeatures, networkInputs );

    neuralMipSize = make_float2( latentWidth >> ( neuralMip + 1 ), latentHeight >> ( neuralMip + 1 ) );
    SampleLatentGrid<vecT>( latentTexture, uv, neuralMip + 1, neuralMipSize, NTC_MLP_FEATURES, numLatentFeatures, networkInputs );
    
    // Encode the sample position
    float pscale = colorMip.positionScale;
    float2 posf = make_float2(texel.x * pscale, texel.y * pscale);
    EncodeSamplePosition<vecT>(posf, colorMip.positionLod, NTC_MLP_FEATURES * 2, networkInputs);
}


namespace HGELUClamp
{
    static __device__ constexpr float minval  = -3.f / 16.f;
    static __device__ constexpr float maxval  = 3.f;
    static __device__ constexpr int   bins    = 256;
    static __device__ constexpr float step    = ( maxval - minval ) / float( bins - 1 );
    static __device__ constexpr float invStep = 1.f / step;
    static __device__ constexpr int   qmax    = int( maxval / step );
    static __device__ constexpr int   qmin    = qmax - bins + 1;
    static __device__ constexpr int bias = -( bins / 2 ) - qmin;
};


template<class VecT>
__device__ __forceinline__
VecT activate(const VecT& x, bool scaleActivation)
{
    VecT tmp    = optixCoopVecFFMA( x, VecT( 1.0f / 3.0f ), VecT( 0.5f ) );
    tmp         = optixCoopVecMin( optixCoopVecMax( tmp, 0.0f ), 1.f );  // clamp(0,1)
    VecT result = optixCoopVecMin( x, 3.0f );
    result      = optixCoopVecMul( result, tmp );
    if( scaleActivation )
        result = optixCoopVecFFMA( result, VecT( (float)HGELUClamp::invStep ), VecT( (float)HGELUClamp::bias ) );
    return result;
}


template <class T_VEC_IN, class T_VEC_OUT>
__device__ __forceinline__
void EvaluateLayer_CoopVec_FP8(
    const T_VEC_IN& inputArray,
    CUdeviceptr     weights,
    uint32_t        weightsOffsetInBytes,
    uint32_t        biasOffsetInBytes,
    bool            scaleActivation,
    T_VEC_OUT&      outputArray )
{
    constexpr int N_IN  = T_VEC_IN::size;
    constexpr int N_OUT = T_VEC_OUT::size;

    outputArray =
        optixCoopVecMatMul
        <
        T_VEC_OUT,                            // VecTOut
        T_VEC_IN,                             // VecTIn
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3, // inputInterpretation
        MAT_LAYOUT,                           // matrixLayout
        false,                                // transpose
        N_OUT,                                // N
        N_IN,                                 // K
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3, // matrixElementType
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16      // biasElementType
        >(
            inputArray,                       // inputVector
            weights,                          // matrix
            weightsOffsetInBytes,
            weights,                          // bias array
            biasOffsetInBytes
        );

    outputArray = activate<T_VEC_OUT>( outputArray, scaleActivation );
}


template <class T_VEC_IN, class T_VEC_OUT>
__device__ __forceinline__
void EvaluateOutputLayer_CoopVec_FP8(
    const T_VEC_IN& inputArray,
    CUdeviceptr     weights,
    uint32_t        weightsOffsetInBytes,
    uint32_t        scaleOffsetInBytes,
    uint32_t        biasOffsetInBytes,
    T_VEC_OUT&      outputArray )
{
    constexpr int N_IN  = T_VEC_IN::size;
    constexpr int N_OUT = T_VEC_OUT::size;
    using T_VEC_MAT_OUT = OptixCoopVec<int32_t, N_OUT>;

    T_VEC_MAT_OUT z_i1 =
        optixCoopVecMatMul
        <
        T_VEC_MAT_OUT,      // VecTOut
        T_VEC_IN,           // VecTIn
        OPTIX_COOP_VEC_ELEM_TYPE_INT8,     // inputInterpretation
        MAT_LAYOUT,                        // matrixLayout
        false,                             // transpose
        N_OUT,                             // N
        N_IN,                              // K
        OPTIX_COOP_VEC_ELEM_TYPE_INT8,     // matrixElementType
        OPTIX_COOP_VEC_ELEM_TYPE_INT32     // biasElementType
        >(
            inputArray,                    // inputVector
            weights,                       // matrix
            weightsOffsetInBytes,
            weights,                       // bias array
            biasOffsetInBytes
        );

    outputArray = optixCoopVecCvt<T_VEC_OUT>( z_i1 );
    T_VEC_OUT scale = optixCoopVecLoad<T_VEC_OUT>( weights + scaleOffsetInBytes );
    outputArray = optixCoopVecMul( outputArray, scale );
}


template <class T_VEC_OUTPUT>
__device__ __forceinline__
bool SampleTextureSet( T_VEC_OUTPUT& outputLayer, NtcTextureSetConstants& tsc, 
    CUtexObject latentTexture, int numLatentFeatures, int latentWidth, int latentHeight,
    CUdeviceptr mlpWeights, int x, int y, int mipLevel )
{
    const int2   imageSize = make_int2( max( tsc.imageWidth >> mipLevel, 1 ), max( tsc.imageHeight >> mipLevel, 1 ) );
    const int2   texel     = make_int2( x, imageSize.y - 1 - y );
    const float2 uv        = make_float2( (texel.x+0.5f) / imageSize.x, (texel.y+0.5f) / imageSize.y );

    const int* weightOffsets = tsc.networkWeightOffsets;
    const int* scaleOffsets  = tsc.networkScaleOffsets;
    const int* biasOffsets   = tsc.networkBiasOffsets;

    // Define network layer types
    using T_VEC_INPUT   = OptixCoopVec<half, NTC_MLP_INPUT_CHANNELS>;
    using T_VEC_HIDDEN0 = OptixCoopVec<half, NTC_MLP_HIDDEN0_CHANNELS>;
    using T_VEC_HIDDEN1 = OptixCoopVec<half, NTC_MLP_HIDDEN1_CHANNELS>;
    using T_VEC_HIDDEN2 = OptixCoopVec<half, NTC_MLP_HIDDEN2_CHANNELS>;

    // Run network
    T_VEC_INPUT networkInputs;
    PrepareNetworkInputs<T_VEC_INPUT>( tsc, latentTexture, numLatentFeatures, latentWidth, latentHeight, uv, texel, mipLevel, networkInputs );
    T_VEC_HIDDEN0 hiddenOutput0;
    EvaluateLayer_CoopVec_FP8<T_VEC_INPUT, T_VEC_HIDDEN0>( networkInputs, mlpWeights, weightOffsets[0], biasOffsets[0], false, hiddenOutput0 );
    T_VEC_HIDDEN1 hiddenOutput1;
    EvaluateLayer_CoopVec_FP8<T_VEC_HIDDEN0, T_VEC_HIDDEN1>( hiddenOutput0, mlpWeights, weightOffsets[1], biasOffsets[1], false, hiddenOutput1 );

#if NTC_MLP_LAYERS == 4
    T_VEC_HIDDEN2 hiddenOutput2;
    EvaluateLayer_CoopVec_FP8<T_VEC_HIDDEN1, T_VEC_HIDDEN2>( hiddenOutput1, mlpWeights, weightOffsets[2], biasOffsets[2], true, hiddenOutput2 );
    EvaluateOutputLayer_CoopVec_FP8<T_VEC_HIDDEN2, T_VEC_OUTPUT>( hiddenOutput2, mlpWeights, weightOffsets[3], scaleOffsets[3], biasOffsets[3], outputLayer );
#else
    EvaluateOutputLayer_CoopVec_FP8<T_VEC_HIDDEN1, T_VEC_OUTPUT>( hiddenOutput1, mlpWeights, weightOffsets[2], scaleOffsets[2], biasOffsets[2], outputLayer );
#endif

    return true;
}
