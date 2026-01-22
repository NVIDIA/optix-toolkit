// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file Texture2DCubic.h 
/// Device-side functions for cubic filtering and sampling derivatives from a texture.

#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/CubicFiltering.h>

namespace demandLoading {

/// Fetch from a demand-loaded texture with the specified textureId. Results are computed and stored
/// in the non-null locations pointed to by result, dresultds, and dresultdt. Filter based on the 
/// filterMode in the texture sampler. Return true if the requested texture data is resident.
template <class TYPE> D_INLINE bool
tex2DCubic( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
            TYPE* result, TYPE* dresultds, TYPE* dresultdt, float2 texelJitter = float2{} )
{
    // Get the sampler if the gradients are small enough. If |grad|>=1, use base color
    bool resident = true; 
    float minGradSquared = minf( dot( ddx, ddx ), dot( ddy, ddy ) );
    TextureSampler* sampler = nullptr;
    if( minGradSquared < 1.0f )
        sampler = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, &resident ) );

#ifdef REQUEST_CASCADE
    // skip the request for 1x1 textures and for |grad|>1
    if( !( resident && !sampler ) )
        resident &= !requestCascade( context, textureId, sampler, ddx, ddy );
#endif

    // Zero out results
    if( result ) *result = TYPE{};
    if( dresultds ) *dresultds = TYPE{};
    if( dresultdt ) *dresultdt = TYPE{};

    // Sample base color if |grad|>=1 or a 1x1 texture
    if( minGradSquared >= 1.0f || ( resident && !sampler ) )
    {
        bool baseColorPageResident = true;
        bool baseColorResident = true;
        if( result )
            baseColorResident &= getBaseColor<TYPE>( context, textureId, *result, &baseColorPageResident );

        if( !baseColorPageResident ) // The base color page is not resident, wait for next launch.
            return false;
        if( baseColorResident ) // The base color is resident, return it.
            return true;

        // The texture does not supply a base color. Get the sampler if it wasn't requested before.
        if( minGradSquared >= 1.0f )
            sampler = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, &resident ) );
    }
    if( !sampler )
        return false;

    const CUtexObject texture = sampler->texture;
    const unsigned int texWidth = sampler->width;
    const unsigned int texHeight = sampler->height;
    const unsigned int filterMode = sampler->filterMode;
    const unsigned int mipmapFilterMode = sampler->desc.mipmapFilterMode;
    const unsigned int maxAnisotropy = sampler->desc.maxAnisotropy;
 
    // Jitter the texture coordinate for stochastic filtering
    s = s + (texelJitter.x / texWidth);
    t = t + (texelJitter.y / texHeight);

    // Prevent footprint from exceeding min tile width for non-mipmapped textures.
    if( sampler->desc.numMipLevels == 1 && sampler->desc.isSparseTexture )
    {
        const float halfMinTileWidth = 32.0f;  // half min tile width for sparse textures
        float pixelSpan = getPixelSpan( ddx, ddy, texWidth, texHeight );
        float scale = minf( halfMinTileWidth / pixelSpan, 1.0f );
        ddx *= scale;
        ddy *= scale;
    }

#ifdef SPARSE_TEX_SUPPORT
    if( sampler->desc.isSparseTexture )
    {
        float pixelSpan = getPixelSpan( ddx, ddy, texWidth, texHeight );
        float expandX = 0.0f;
        float expandY = 0.0f;
        if( filterMode == FILTER_BICUBIC || ( filterMode == FILTER_SMARTBICUBIC && pixelSpan < 0.5f ) )
        {
            // expand by 2 pixels for cubic sampling
            expandX = 2.0f * maxf( pixelSpan, 1.0f ) / texWidth;
            expandY = 2.0f * maxf( pixelSpan, 1.0f ) / texHeight;
        }
        else if( dresultds != nullptr || dresultdt != nullptr )
        {
            // expand by half footprint extent for linear sampling because of finite differencing
            expandX = 0.5f * maxf( fabsf( ddx.x ), fabsf( ddy.x ) );
            expandY = 0.5f * maxf( fabsf( ddx.y ), fabsf( ddy.y ) );
        }
        resident &= requestTexFootprint2DGrad( *sampler, context.referenceBits, context.residenceBits, 
                                               s, t, ddx.x, ddx.y, ddy.x, ddy.y, expandX, expandY );
        if( !resident )
            return false;
    }
#endif

    textureCubic( texture, texWidth, texHeight, filterMode, mipmapFilterMode, maxAnisotropy, sampler->conservativeFilter,
                  s, t, ddx, ddy, result, dresultds, dresultdt );
    return resident;
}


/// Fetch from a demand-loaded texture with the specified textureId. Simplified entry point without derivatives.
template <class TYPE> D_INLINE bool
tex2DCubic( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
            TYPE* result, float2 texelJitter = float2{} )
{
    return tex2DCubic( context, textureId, s, t, ddx, ddy, result, nullptr, nullptr, texelJitter );
}


/// Fetch from a demand-loaded udim (or non-udim) texture with the specified textureId. Results are 
/// computed and stored in the non-null locations pointed to by result, dresultds, and dresultdt. 
/// Filter based on the filterMode in the texture sampler. Return true if the requested texture data is resident.
template <class TYPE> D_INLINE bool
tex2DCubicUdim( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
                TYPE* result, TYPE* dresultds, TYPE* dresultdt, float2 texelJitter = float2{} )
{
    bool resident;
    TextureSampler* bsmp = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, &resident ) ); // base sampler
    if( !resident )
    {
        if( result ) *result = TYPE{};
        if( dresultds ) *dresultds = TYPE{};
        if( dresultdt ) *dresultdt = TYPE{};
        return false;
    }

    // Use base texture if it's not a udim texture, if the mip level fits in the base texture, or if the base texture has a cascade.
    bool useBaseTexture = ( !bsmp ) || ( bsmp && bsmp->udim == 0 );
    if( !useBaseTexture && bsmp->desc.isUdimBaseTexture )
    {
        float mipLevel = getMipLevel( ddx, ddy, bsmp->width * bsmp->udim, bsmp->height * bsmp->vdim, 1.0f / bsmp->desc.maxAnisotropy );
        useBaseTexture = ( mipLevel >= 0.0f ) || bsmp->hasCascade;
        if( useBaseTexture )
            texelJitter = float2{0.0f};
    }

    // Sampling the subtexture. Change textureId, texture coords, and texture gradients.
    if( !useBaseTexture )
    {
        float        subs, subt;
        unsigned int sidx, tidx;
        separateUdimCoord( s, CU_TR_ADDRESS_MODE_WRAP, bsmp->udim, subs, sidx );
        separateUdimCoord( t, CU_TR_ADDRESS_MODE_WRAP, bsmp->vdim, subt, tidx );
        textureId = bsmp->udimStartPage + ( tidx * bsmp->udim + sidx ) * bsmp->numChannelTextures;
        s = subs;
        t = subt;
    }
    else if( bsmp && bsmp->udim != 0 ) // Scale the gradients for the base texture
    {
        ddx /= bsmp->udim;
        ddy /= bsmp->vdim;
    }

    // Call sampling routine
    return tex2DCubic( context, textureId, s, t, ddx, ddy, result, dresultds, dresultdt, texelJitter );
}

/// Fetch from a demand-loaded udim texture with the specified textureId. Simplified entry point without derivatives.
template <class TYPE> D_INLINE bool
tex2DCubicUdim( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
                TYPE* result, float2 texelJitter = float2{} )
{
    return tex2DCubicUdim<TYPE>( context, textureId, s, t, ddx, ddy, result, nullptr, nullptr, texelJitter );
}

} // namespace demandloading
