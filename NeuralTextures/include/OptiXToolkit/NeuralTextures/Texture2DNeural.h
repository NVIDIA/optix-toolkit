// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file Texture2DNeural.h 
/// Device-side functions for sampling neural textures.

#include <OptiXToolkit/NeuralTextures/InferenceDataOptix.h>
#include <OptiXToolkit/NeuralTextures/Texture2DNeural.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>

namespace neuralTextures {

// Valid output types for the neural texture functions
using T_VEC_OUT_FLOAT = OptixCoopVec<float, NTC_MLP_OUTPUT_CHANNELS>;
using T_VEC_OUT_HALF = OptixCoopVec<half, NTC_MLP_OUTPUT_CHANNELS>;

D_INLINE float2 texelJitter( float2 xi, int mipLevel, int filterMode )
{
    if( filterMode == FILTER_BILINEAR || ( mipLevel > 0 && filterMode == FILTER_SMARTBICUBIC ) )
        return boxFilter( xi );
    else if ( filterMode >= FILTER_BICUBIC )
        return boxMuller( xi ) * GAUSSIAN_STANDARD_WIDTH;
    return float2{0.0f, 0.0f};
}

template <class T_VEC_OUT> D_INLINE bool
ntcTex2DGrad( T_VEC_OUT& out, InferenceDataOptix* &infData, const DeviceContext& context, TextureSampler* sampler, float u, float v, float2 ddx, float2 ddy, float2 xi )
{
    // Get the ntc inference data
    if( !sampler || !sampler->extraData )
        return false;

    infData = reinterpret_cast<InferenceDataOptix*>( sampler->extraData );

    // Get mip level
    float mipLevel = getMipLevel( ddx, ddy, sampler->width, sampler->height, 1.0f / sampler->desc.maxAnisotropy );
    float mipJitter = ( sampler->desc.mipmapFilterMode ) ? xi.x : 0.5f;
    int mip = clamp( (int)( mipLevel + mipJitter ), 0, sampler->desc.numMipLevels - 1 );
    float2 uv = float2{u, v};

    // Request texture footprint for neural texture mips
#ifdef SPARSE_TEX_SUPPORT
    if( sampler->desc.isSparseTexture )
    {
        // FIXME: Handle cascades, which require to translate between specified neural mip and
        // actual neural mip based on cascade. Also need to request the cascade.
        const NtcColorMipConstants& colorMip = infData->constants.colorMips[mip];
        float neuralMip = colorMip.neuralMip + 0.5f;
        if( !requestTexFootprint2DLod( *sampler, context.referenceBits, context.residenceBits, uv.x, uv.y, neuralMip ) )
            return false;
    }
#endif

    float2 jitter = texelJitter( xi, mip, sampler->filterMode );
    const int mipWidth = infData->constants.imageWidth >> mip;
    const int mipHeight = infData->constants.imageHeight >> mip;
    const int x = clamp( (int)(uv.x * mipWidth + jitter.x), 0, mipWidth - 1 );
    const int y = clamp( (int)(uv.y * mipHeight + jitter.y), 0, mipHeight - 1 );

    return SampleTextureSet<T_VEC_OUT>( out, infData->constants, sampler->texture, infData->latentFeatures, 
        infData->latentWidth, infData->latentHeight, infData->d_mlpWeights, x, y, mip );
}


template <class T_VEC_OUT> D_INLINE bool
ntcTex2DGrad( T_VEC_OUT& out, InferenceDataOptix* &infData, const DeviceContext& context, unsigned int textureId, float u, float v, float2 ddx, float2 ddy, float2 xi )
{
    bool resident;
    TextureSampler* sampler = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, &resident ) );
    return ntcTex2DGrad<T_VEC_OUT>( out, infData, context, sampler, u, v, ddx, ddy, xi );
}

template <class T_VEC_OUT> D_INLINE bool
ntcTex2DGrad( T_VEC_OUT& out, const DeviceContext& context, unsigned int textureId, float u, float v, float2 ddx, float2 ddy, float2 xi )
{
    InferenceDataOptix* infData = nullptr;
    return ntcTex2DGrad<T_VEC_OUT>( out, infData, context, textureId, u, v, ddx, ddy, xi );
}


template <class T_VEC_OUT> D_INLINE bool
ntcTex2DGradUdim( T_VEC_OUT& out, InferenceDataOptix* &infData, const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, float2 xi )
{
    bool isResident = true;
    TextureSampler* bsmp = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, &isResident ) ); // base sampler

    // Use base texture if it's not a udim texture, if the mip level fits in the base texture, or if the base texture has a cascade.
    bool useBaseTexture = ( !bsmp ) || ( bsmp && bsmp->udim == 0 );
    if( !useBaseTexture && bsmp->desc.isUdimBaseTexture )
    {
        float mipLevel = getMipLevel( ddx, ddy, bsmp->width * bsmp->udim, bsmp->height * bsmp->vdim, 1.0f / bsmp->desc.maxAnisotropy );
        useBaseTexture = ( mipLevel >= 0.0f ) || bsmp->hasCascade;
    }

    // Sample a subtexture
    if( !useBaseTexture )
    {
        float        sx, sy;
        unsigned int xidx, yidx;
        separateUdimCoord( x, CU_TR_ADDRESS_MODE_WRAP, bsmp->udim, sx, xidx );
        separateUdimCoord( y, CU_TR_ADDRESS_MODE_WRAP, bsmp->vdim, sy, yidx );

        unsigned int subTexId = bsmp->udimStartPage + ( yidx * bsmp->udim + xidx ) * bsmp->numChannelTextures;
        isResident = ntcTex2DGrad<T_VEC_OUT>( out, infData, context, subTexId, sx, sy, ddx, ddy, xi );

        if( isResident || !bsmp->desc.isUdimBaseTexture )
            return isResident;
    }

    // Scale the gradients and sample base texture
    if( bsmp && bsmp->udim != 0 )
    {
        ddx /= bsmp->udim;
        ddy /= bsmp->vdim;
    }
    return ntcTex2DGrad<T_VEC_OUT>( out, infData, context, bsmp, x, y, ddx, ddy, xi );
}

template <class T_VEC_OUT> D_INLINE bool
ntcTex2DGradUdim( T_VEC_OUT& out, const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, float2 xi )
{
    InfereceDataOptix* infData;
    return ntcTex2DGradUdim<T_VEC_OUT>( out, infData, context, textureId, x, y, ddx, ddy, xi );
}

} // namespace neuralTextures
