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

/// \file CubicFiltering.h
/// Device-side functions for cubic filtering and sampling derivatives from a texture.

#include <OptiXToolkit/ShaderUtil/TextureUtil.h>

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
    float pixelSpanX = fmaxf( fabsf( ddx.x ), fabsf( ddy.x ) ) * width;
    float pixelSpanY = fmaxf( fabsf( ddx.y ), fabsf( ddy.y ) ) * height;
    return fmaxf( pixelSpanX, pixelSpanY ) + 1.0e-8f;
}

D_INLINE void fixGradients( float2& ddx, float2& ddy, int width, int height, int filterMode )
{
    const float invAnisotropy = 1.0f / 16.0f;
    const float minCubicVal   = 1.0f / 131072.0f;
    float minVal = ( filterMode <= FILTER_BILINEAR ) ? 0.5f / fmaxf(width, height) : minCubicVal;

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
textureWeighted( CUtexObject texture, float i, float j, float4 wx, float4 wy, int mipLevel, int mipLevelWidth, int mipLevelHeight )
{
    // Get x,y coordinates to sample
    float x0 = (i - 0.5f + wx.y / (wx.x + wx.y) ) / mipLevelWidth;
    float x1 = (i + 1.5f + wx.w / (wx.z + wx.w) ) / mipLevelWidth;
    float y0 = (j - 0.5f + wy.y / (wy.x + wy.y) ) / mipLevelHeight;
    float y1 = (j + 1.5f + wy.w / (wy.z + wy.w) ) / mipLevelHeight;

    // Sum four weighted samples
    TYPE t0 = (wx.x + wx.y) * (wy.x + wy.y) * ::tex2DLod<TYPE>( texture, x0, y0, mipLevel );
    TYPE t1 = (wx.z + wx.w) * (wy.x + wy.y) * ::tex2DLod<TYPE>( texture, x1, y0, mipLevel );
    TYPE t2 = (wx.x + wx.y) * (wy.z + wy.w) * ::tex2DLod<TYPE>( texture, x0, y1, mipLevel );
    TYPE t3 = (wx.z + wx.w) * (wy.z + wy.w) * ::tex2DLod<TYPE>( texture, x1, y1, mipLevel );
    return t0 + t1 + t2 + t3;
}

/// Compute cubic filtered texture sample based on filterMode.
template <class TYPE> D_INLINE bool
textureCubic( CUtexObject texture, int texWidth, int texHeight,
              unsigned int filterMode, unsigned int mipmapFilterMode, unsigned int maxAnisotropy,
              float s, float t, float2 ddx, float2 ddy, TYPE* result, TYPE* dresultds, TYPE* dresultdt )
{
    // General gradient size fixes
    fixGradients( ddx, ddy, texWidth, texHeight, filterMode );

    // Determine the blend between linear and cubic filtering based on the filter mode
    float pixelSpan = getPixelSpan( ddx, ddy, texWidth, texHeight );
    float cubicBlend = 0.0f;
    float ml = 0.0f;
    int mipLevel = 0;

    if( filterMode == FILTER_BICUBIC ) 
    {
        cubicBlend = 1.0f;
        ml = getMipLevel( ddx, ddy, texWidth, texHeight, 1.0f / maxAnisotropy );
        if( mipmapFilterMode == CU_TR_FILTER_MODE_POINT )
            ml = fmaxf( 0.0f, ceilf( ml - 0.5f ) );
        mipLevel = ceilf( ml );
        mipLevel = max( mipLevel, 0 );
    }
    else if( filterMode == FILTER_SMARTBICUBIC && pixelSpan <= 0.5f ) 
    {
        cubicBlend = fminf( -1.0f - log2f( pixelSpan ), 1.0f );
        if( mipmapFilterMode == CU_TR_FILTER_MODE_POINT )
            cubicBlend = floorf( cubicBlend + 0.5f );
    }

    int mipLevelWidth = max(texWidth >> mipLevel, 1);
    int mipLevelHeight = max(texHeight >> mipLevel, 1);

    // Linear Sampling
    if( cubicBlend < 1.0f )
    {
        // Don't do bilinear sample for result unless cubicBlend is 0.
        if( cubicBlend <= 0.0f && result )
        {
            *result = ::tex2DGrad<TYPE>( texture, s, t, ddx, ddy );
        }

        // Do a central difference along ddx, ddy
        if( dresultds )
        {
            TYPE t1 = ::tex2DGrad<TYPE>( texture, s + ddx.x, t + ddx.y, ddx, ddy );
            TYPE t2 = ::tex2DGrad<TYPE>( texture, s - ddx.x, t - ddx.y, ddx, ddy );
            *dresultds = ( t1 - t2 ) / ( 2.0f * length( ddx ) );
        }
        if( dresultdt )
        {
            TYPE t1 = ::tex2DGrad<TYPE>( texture, s + ddy.x, t + ddy.y, ddx, ddy );
            TYPE t2 = ::tex2DGrad<TYPE>( texture, s - ddy.x, t - ddy.y, ddx, ddy );
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
        *result = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight );
    }
    if( dresultds || dresultdt )
    {
        // Get axis aligned cubic derivatives
        float4 wx = cubicDerivativeWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        TYPE drds = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelWidth;
    
        wx = cubicWeights(ts - i);
        wy = cubicDerivativeWeights(tt - j);
        TYPE drdt = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelHeight;

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
        return;

    //-------------------------------------------------------------------------------
    // Sample second level for blending between levels in FILTER_BICUBIC mode

    // Get unnormalized texture coordinates
    mipLevel--;
    mipLevelWidth = max(texWidth >> mipLevel, 1);
    mipLevelHeight = max(texHeight >> mipLevel, 1);
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
        *result = levelBlend * textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) + (1.0f - levelBlend) * *result;
    }
    if( dresultds || dresultdt )
    {
        // Get axis aligned cubic derivatives
        float4 wx = cubicDerivativeWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        TYPE drds = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelWidth;

        wx = cubicWeights(ts - i);
        wy = cubicDerivativeWeights(tt - j);
        TYPE drdt = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelHeight;

        // Rotate cubic derivatives to align with ddx and ddy, and blend with linear derivatives computed earlier
        float a = atan2f( ddx.y, ddx.x );
        if( dresultds )
            *dresultds = levelBlend * ( drds * cosf( a ) + drdt * sinf( a ) ) + (1.0f - levelBlend) * *dresultds;

        float b = atan2f( ddy.y, ddy.x );
        if( dresultdt )
            *dresultdt = levelBlend * ( drds * cosf( b ) + drdt * sinf( b ) ) + (1.0f - levelBlend) * *dresultdt;
    }
}
