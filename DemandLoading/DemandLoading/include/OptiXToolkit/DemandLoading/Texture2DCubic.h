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

namespace demandLoading {

D_INLINE float4 cubicWeights( float x )
{
    float4 weight;
    weight.x = -x*x*x      + 3.0f*x*x - 3.0f*x + 1.0f;
    weight.y = 3.0f*x*x*x  - 6.0f*x*x          + 4.0f;
    weight.z = -3.0f*x*x*x + 3.0f*x*x + 3.0f*x + 1.0f;
    weight.w = x*x*x;
    return weight * (1.0f / 6.0f);
}

D_INLINE float4 cubicDerivativeWeights( float x )
{
    float4 weight;
    weight.x = -0.5f*x*x + x       - 0.5f;
    weight.y = 1.5f*x*x  - 2.0f*x;
    weight.z = -1.5f*x*x + x       + 0.5f;
    weight.w = 0.5f*x*x;
    return weight;
}

D_INLINE float4 linearWeights( float x )
{
    return float4{ 0.0f, 1.0f-x, x, 0.0f };
}

D_INLINE float4 linearDerivativeWeights( float x )
{
    return float4{ x-1.0f, -x, 1.0f-x, x } * 0.5f; // finite difference
    //return float4{ 0.0f, -0.5f, 0.5f, 0.0f }; // actual linear derivative
}

D_INLINE float getPixelSpan( float2 ddx, float2 ddy, float width, float height )
{
    float pixelSpanX = maxf( fabsf( ddx.x ), fabsf( ddy.x ) ) * width;
    float pixelSpanY = maxf( fabsf( ddx.y ), fabsf( ddy.y ) ) * height;
    return maxf( pixelSpanX, pixelSpanY ) + 1.0e-8f;
}

D_INLINE void fixGradients( float2& ddx, float2& ddy, TextureSampler* sampler, int filterMode )
{
    const float halfMinTileWidth = 32.0f;  // half min tile width for sparse textures
    const float invAnisotropy    = 1.0f / 16.0f;
    const float minCubicVal      = 1.0f / 131072.0f;

    // Fix large gradients. Prevent footprint from exceeding min tile width for non-mipmapped textures
    if( sampler->desc.numMipLevels == 1 && sampler->desc.isSparseTexture )
    {
        float pixelSpan = getPixelSpan( ddx, ddy, sampler->width, sampler->height );
        float scale = minf( halfMinTileWidth / pixelSpan, 1.0f );
        ddx *= scale;
        ddy *= scale;
    }

    float minVal = ( filterMode <= FILTER_BILINEAR ) ? 0.5f / maxf(sampler->width, sampler->height) : minCubicVal;

    // Check if gradients are large enough
    float ddx2 = dot( ddx, ddx );
    float ddy2 = dot( ddy, ddy );
    if( ddx2 > minVal*minVal && ddy2 > minVal*minVal )
        return;

    // Fix zero gradients
    if( ddx2 + ddy2 == 0.0f )
        ddx = float2{ minVal, 0.0f };
    else if( ddx2 == 0.0f )
        ddx = float2{ ddy.y, -ddy.x } * invAnisotropy;

    if( ddy2 == 0.0f )
        ddy = float2{ -ddx.y, ddx.x } * invAnisotropy;

    // Fix short gradients
    if( dot( ddx, ddx ) < minVal*minVal )
        ddx *= minVal / length( ddx );
    if( dot( ddy, ddy ) < minVal*minVal )
        ddy *= minVal / length( ddy );
}

template <class TYPE> D_INLINE TYPE
textureWeighted( TextureSampler* sampler, float i, float j, float4 wx, float4 wy, int mipLevel )
 {
    int width = ( sampler->width >> mipLevel );
    int height = ( sampler->height >> mipLevel );

    // Get x,y coordinates to sample
    float x0 = (i - 0.5f + wx.y / (wx.x + wx.y) ) / width;
    float x1 = (i + 1.5f + wx.w / (wx.z + wx.w) ) / width;
    float y0 = (j - 0.5f + wy.y / (wy.x + wy.y) ) / height;
    float y1 = (j + 1.5f + wy.w / (wy.z + wy.w) ) / height;

    // Sum four weighted samples
    TYPE t0 = (wx.x + wx.y) * (wy.x + wy.y) * ::tex2DLod<TYPE>( sampler->texture, x0, y0, mipLevel );
    TYPE t1 = (wx.z + wx.w) * (wy.x + wy.y) * ::tex2DLod<TYPE>( sampler->texture, x1, y0, mipLevel );
    TYPE t2 = (wx.x + wx.y) * (wy.z + wy.w) * ::tex2DLod<TYPE>( sampler->texture, x0, y1, mipLevel );
    TYPE t3 = (wx.z + wx.w) * (wy.z + wy.w) * ::tex2DLod<TYPE>( sampler->texture, x1, y1, mipLevel );
    return t0 + t1 + t2 + t3;
}


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

    // Fix gradient sizes
    const unsigned int filterMode = sampler->filterMode;
    fixGradients( ddx, ddy, sampler, filterMode );

    // Jitter the texture coordinate for stochastic filtering
    s = s + (texelJitter.x / sampler->width);
    t = t + (texelJitter.y / sampler->height);

    // Determine the blend between linear and cubic filtering based on the filterolation mode
    float pixelSpan = getPixelSpan( ddx, ddy, sampler->width, sampler->height );
    float cubicBlend = 0.0f;
    float ml = 0.0f;
    int mipLevel = 0;

    if( filterMode == FILTER_BICUBIC ) 
    {
        cubicBlend = 1.0f;
        ml = getMipLevel( ddx, ddy, sampler->width, sampler->height, 1.0f / sampler->desc.maxAnisotropy );
        if( sampler->desc.mipmapFilterMode == CU_TR_FILTER_MODE_POINT )
            ml = maxf( 0.0f, ceilf( ml - 0.5f ) );
        mipLevel = ceilf( ml );
        mipLevel = max( mipLevel, 0 );
    }
    else if( filterMode == FILTER_SMARTBICUBIC && pixelSpan <= 0.5f ) 
    {
        cubicBlend = minf( -1.0f - log2f( pixelSpan ), 1.0f );
        if( sampler->desc.mipmapFilterMode == CU_TR_FILTER_MODE_POINT )
            cubicBlend = floorf( cubicBlend + 0.5f );
    }

    int mipLevelWidth = max(sampler->width >> mipLevel, 1);
    int mipLevelHeight = max(sampler->height >> mipLevel, 1);

#ifdef SPARSE_TEX_SUPPORT
    if( sampler->desc.isSparseTexture )
    {
        float expandX = 0.0f;
        float expandY = 0.0f;
        if( cubicBlend > 0.0f )
        {
            // expand by 2 pixels for cubic sampling
            expandX = 2.0f / mipLevelWidth;
            expandY = 2.0f / mipLevelHeight;
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

    // Linear Sampling
    if( cubicBlend < 1.0f )
    {
        // Don't do bilinear sample for result unless cubicBlend is 0.
        if( cubicBlend <= 0.0f && result )
        {
            *result = ::tex2DGrad<TYPE>( sampler->texture, s, t, ddx, ddy );
        }

        // Do a central difference along ddx, ddy
        if( dresultds )
        {
            TYPE t1 = ::tex2DGrad<TYPE>( sampler->texture, s + ddx.x, t + ddx.y, ddx, ddy );
            TYPE t2 = ::tex2DGrad<TYPE>( sampler->texture, s - ddx.x, t - ddx.y, ddx, ddy );
            *dresultds = ( t1 - t2 ) / ( 2.0f * length( ddx ) );
        }
        if( dresultdt )
        {
            TYPE t1 = ::tex2DGrad<TYPE>( sampler->texture, s + ddy.x, t + ddy.y, ddx, ddy );
            TYPE t2 = ::tex2DGrad<TYPE>( sampler->texture, s - ddy.x, t - ddy.y, ddx, ddy );
            *dresultdt = ( t1 - t2 ) / ( 2.0f * length( ddy ) );
        }

        if( cubicBlend <= 0.0f )
            return true;
    }

    // Get unnormalized texture coordinates
    float ts = s * mipLevelWidth - 0.5f;
    float tt = t * mipLevelHeight - 0.5f;
    float i = floorf( ts );
    float j = floorf( tt );

    // Do cubic sampling
    if( result )
    {
        // Blend between cubic and linear weights
        float4 wx = cubicBlend * cubicWeights(ts - i) + (1.0f - cubicBlend) * linearWeights(ts - i);
        float4 wy = cubicBlend * cubicWeights(tt - j) + (1.0f - cubicBlend) * linearWeights(tt - j);
        *result = textureWeighted<TYPE>( sampler, i, j, wx, wy, mipLevel );
    }
    if( dresultds || dresultdt )
    {
        // Get axis aligned cubic derivatives
        float4 wx = cubicDerivativeWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        TYPE drds = textureWeighted<TYPE>( sampler, i, j, wx, wy, mipLevel ) * sampler->width;
    
        wx = cubicWeights(ts - i);
        wy = cubicDerivativeWeights(tt - j);
        TYPE drdt = textureWeighted<TYPE>( sampler, i, j, wx, wy, mipLevel ) * sampler->height;

        // Rotate cubic derivatives to align with ddx and ddy, and blend with linear derivatives computed earlier
        float a = atan2f( ddx.y, ddx.x );
        if( dresultds )
            *dresultds = cubicBlend * ( drds * cosf( a ) + drdt * sinf( a ) ) + (1.0f - cubicBlend) * *dresultds;

        float b = atan2f( ddy.y, ddy.x );
        if( dresultdt )
            *dresultdt = cubicBlend * ( drds * cosf( b ) + drdt * sinf( b ) ) + (1.0f - cubicBlend) * *dresultdt;
    }

    // Return unless we have to blend between levels
    if( filterMode != FILTER_BICUBIC || ml == mipLevel || ml < 0.0f )
        return resident;

    //-------------------------------------------------------------------------------
    // Sample second level for blending between levels in FILTER_BICUBIC mode

    // Get unnormalized texture coordinates
    mipLevel--;
    mipLevelWidth = max(sampler->width >> mipLevel, 1);
    mipLevelHeight = max(sampler->height >> mipLevel, 1);
    ts = s * mipLevelWidth - 0.5f;
    tt = t * mipLevelHeight - 0.5f;
    i = floorf( ts );
    j = floorf( tt );

    float levelBlend = 1.0f - (ml - mipLevel);

    // Do cubic sampling
    if( result )
    {
        // Blend between cubic and linear weights
        float4 wx = cubicWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        *result = levelBlend * textureWeighted<TYPE>( sampler, i, j, wx, wy, mipLevel ) + (1.0f - levelBlend) * *result;
    }
    if( dresultds || dresultdt )
    {
        // Get axis aligned cubic derivatives
        float4 wx = cubicDerivativeWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        TYPE drds = textureWeighted<TYPE>( sampler, i, j, wx, wy, mipLevel ) * sampler->width;

        wx = cubicWeights(ts - i);
        wy = cubicDerivativeWeights(tt - j);
        TYPE drdt = textureWeighted<TYPE>( sampler, i, j, wx, wy, mipLevel ) * sampler->height;

        // Rotate cubic derivatives to align with ddx and ddy, and blend with linear derivatives computed earlier
        float a = atan2f( ddx.y, ddx.x );
        if( dresultds )
            *dresultds = levelBlend * ( drds * cosf( a ) + drdt * sinf( a ) ) + (1.0f - levelBlend) * *dresultds;

        float b = atan2f( ddy.y, ddy.x );
        if( dresultdt )
            *dresultdt = levelBlend * ( drds * cosf( b ) + drdt * sinf( b ) ) + (1.0f - levelBlend) * *dresultdt;
    }

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
