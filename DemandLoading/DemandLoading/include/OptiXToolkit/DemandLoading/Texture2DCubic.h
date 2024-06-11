//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

/// \file Texture2DCubic.h 
/// Device-side functions for cubic filtering and sampling derivatives from a texture.

#include <OptiXToolkit/DemandLoading/Texture2DExtended.h>
#include <OptiXToolkit/ShaderUtil/CubicFiltering.h>

namespace demandLoading {

/// Fetch from a demand-loaded texture with the specified textureId. Results are computed and stored
/// in the non-null locations pointed to by result, dresultds, and dresultdt. Filter based on the 
/// filterMode in the texture sampler. Return true if the requested texture data is resident.
template <class TYPE> D_INLINE bool
textureCubic( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
              TYPE* result, TYPE* dresultds, TYPE* dresultdt, float2 texelJitter = float2{} )
{
    // Get the sampler if the gradients are small enough. If |grad|>1, use base color
    bool resident = true; 
    bool baseColorResident = true;
    float minGradSquared = minf( dot( ddx, ddx ), dot( ddy, ddy ) );
    TextureSampler* sampler = nullptr; 
    if( minGradSquared < 1.0f )
        sampler = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, &resident ) );

#ifdef REQUEST_CASCADE
    resident &= !requestCascade( context, textureId, sampler, ddx, ddy );
#endif

    // Zero out results
    if( result ) *result = TYPE{};
    if( dresultds ) *dresultds = TYPE{};
    if( dresultdt ) *dresultdt = TYPE{};

    // Sample base color
    if( !sampler && result )
        resident &= getBaseColor<TYPE>( context, textureId, *result, &baseColorResident );
    if( !sampler )
        return resident && baseColorResident;

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

    textureCubic( texture, texWidth, texHeight, filterMode, mipmapFilterMode, maxAnisotropy,
                  s, t, ddx, ddy, result, dresultds, dresultdt );
    return resident;
}


/// Fetch from a demand-loaded texture with the specified textureId. Simplified entry point without derivatives.
template <class TYPE> D_INLINE bool
textureCubic( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
              TYPE* result, float2 texelJitter = float2{} )
{
    return textureCubic( context, textureId, s, t, ddx, ddy, result, nullptr, nullptr, texelJitter );
}


/// Fetch from a demand-loaded udim (or non-udim) texture with the specified textureId. Results are 
/// computed and stored in the non-null locations pointed to by result, dresultds, and dresultdt. 
/// Filter based on the filterMode in the texture sampler. Return true if the requested texture data is resident.
template <class TYPE> D_INLINE bool
textureUdim( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
             TYPE* result, TYPE* dresultds, TYPE* dresultdt, float2 texelJitter = float2{} )
{
    float minGradSquared = minf( dot( ddx, ddx ), dot( ddy, ddy ) );
    if( minGradSquared < 1.0f )
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
            float mipLevel = getMipLevel( ddx, ddy, bsmp->width, bsmp->height, 1.0f / bsmp->desc.maxAnisotropy );
            useBaseTexture = ( mipLevel >= 0.0f ) || bsmp->hasCascade;
            if( useBaseTexture )
                texelJitter = float2{0.0f};
        }

        // Sampling the subtexture. Change textureId, texture coords, and texture gradients.
        if( !useBaseTexture )
        {
            float        subs, subt;
            unsigned int sidx, tidx;
            wrapAndSeparateUdimCoord( s, CU_TR_ADDRESS_MODE_WRAP, bsmp->udim, subs, sidx );
            wrapAndSeparateUdimCoord( t, CU_TR_ADDRESS_MODE_WRAP, bsmp->vdim, subt, tidx );

            textureId = bsmp->udimStartPage + ( tidx * bsmp->udim + sidx ) * bsmp->numChannelTextures;
            s = subs;
            t = subt;
            ddx = float2{ ddx.x * bsmp->udim, ddx.y * bsmp->vdim };
            ddy = float2{ ddy.x * bsmp->udim, ddy.y * bsmp->vdim };
        }
    }

    // Call sampling routine
    return textureCubic( context, textureId, s, t, ddx, ddy, result, dresultds, dresultdt, texelJitter );
}

/// Fetch from a demand-loaded udim texture with the specified textureId. Simplified entry point without derivatives.
template <class TYPE> D_INLINE bool
textureUdim( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
             TYPE* result, float2 texelJitter = float2{} )
{
    return textureUdim<TYPE>( context, textureId, s, t, ddx, ddy, result, nullptr, nullptr, texelJitter );
}


D_INLINE void copyResult( float* result, float4 res, int startIdx, int nchannels )
{
    if( !result )
        return;
    result[startIdx] = res.x;
    if( startIdx + 1 < nchannels ) result[startIdx + 1] = res.y;
    if( startIdx + 2 < nchannels ) result[startIdx + 2] = res.z;
    if( startIdx + 3 < nchannels ) result[startIdx + 3] = res.w;
}

/// Fetch from a demand-loaded udim (or non-udim) texture with the specified starting textureId. 
/// Results are computed and stored as floats in the non-null locations pointed to by result, dresultds, and dresultdt. 
/// nchannels specifies how many channels to fetch from the texture.  The function assumes that channels
/// are stored in textures with consecutive ids, and that each texture except for the last has 4 channels.
/// Filtering is based on the filterMode in the texture sampler. Returns true if the requested texture data is resident.
D_INLINE bool
texture( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
         int nchannels, float* result, float* dresultds, float* dresultdt, float2 texelJitter = float2{} )
{
    bool resident = true;
    float4 res, drds, drdt;
    float4* resPtr = (result) ? &res : nullptr;
    float4* drdsPtr = (dresultds) ? &drds : nullptr; 
    float4* drdtPtr = (dresultdt) ? &drdt : nullptr;

    int idx = 0;
    do
    {
        resident &= textureUdim<float4>( context, textureId + (idx>>2), s, t, ddx, ddy, resPtr, drdsPtr, drdtPtr, texelJitter );
        copyResult( result, res, idx, nchannels );
        copyResult( dresultds, drds, idx, nchannels );
        copyResult( dresultdt, drdt, idx, nchannels );
        idx += 4;
    } while( idx < nchannels );

    return resident;
}

/// Fetch from a demand-loaded udim texture as floats with the specified starting textureId. 
/// Simplified entry point without derivatives.
D_INLINE bool
texture( const DeviceContext& context, unsigned int textureId, float s, float t, float2 ddx, float2 ddy, 
         int nchannels, float* result, float2 texelJitter = float2{} )
{
    return texture( context, textureId, s, t, ddx, ddy, nchannels, result, nullptr, nullptr, texelJitter );
}

} // namespace demandloading
