// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file CubicFiltering.h
/// Device-side functions for cubic filtering and sampling derivatives from a texture.

using namespace otk;
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

D_INLINE float getPixelSpan( float2 ddx, float2 ddy, float width, float height )
{
    float pixelSpanX = fmaxf( fabsf( ddx.x ), fabsf( ddy.x ) ) * width;
    float pixelSpanY = fmaxf( fabsf( ddx.y ), fabsf( ddy.y ) ) * height;
    return fmaxf( pixelSpanX, pixelSpanY ) + 1.0e-8f;
}

D_INLINE void fixGradients( float2& ddx, float2& ddy, int width, int height, int filterMode, bool conservative )
{
    const float invAnisotropy = 1.0f / 16.0f; // FIXME: This should come from the texture
    const float invAniso2 = invAnisotropy * invAnisotropy;
    const float minVal = 0.99f / fmaxf(width, height);

    // Check if gradients are large enough
    float ddx2 = dot( ddx, ddx );
    float ddy2 = dot( ddy, ddy );
    if( ddx2 > minVal*minVal && ddy2 > minVal*minVal )
    {
        if( conservative || ( ddx2 >= invAniso2 * ddy2 && ddy2 >= invAniso2 * ddx2 ) )
            return;
    }

    // Fix zero gradients
    if( ddx2 == 0.0f )
        ddx = float2{ minVal, 0.0f };
    if( ddy2 == 0.0f )
        ddy = float2{ 0.0f, minVal };

    // Fix short gradients
    if( dot( ddx, ddx ) < minVal*minVal )
        ddx *= minVal / length( ddx );
    if( dot( ddy, ddy ) < minVal*minVal )
        ddy *= minVal / length( ddy );

    // Fix anisotropy for non-conservative filtering
    if( conservative )
        return;

    ddx2 = dot( ddx, ddx );
    ddy2 = dot( ddy, ddy );
    if( ddx2 * invAniso2 > ddy2 )
        ddx *= sqrtf( ddy2 / ( ddx2 * invAniso2 ) );
    if( ddy2 * invAniso2 > ddx2 )
        ddy *= sqrtf( ddx2 / ( ddy2 * invAniso2 ) );
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
template <class TYPE> D_INLINE void
textureCubic( CUtexObject texture, int texWidth, int texHeight,
              unsigned int filterMode, unsigned int mipmapFilterMode, unsigned int maxAnisotropy, bool conservativeFiltering,
              float s, float t, float2 ddx, float2 ddy, TYPE* result, TYPE* dresultds, TYPE* dresultdt )
{
    float p1 = (ddx.x * texWidth) * (ddx.x * texWidth) + (ddx.y * texHeight) * (ddx.y * texHeight);
    float p2 = (ddy.x * texWidth) * (ddy.x * texWidth) + (ddy.y * texHeight) * (ddy.y * texHeight);
    float maxGradientLengthInPixels = sqrtf( maxf( p1, p2 ) );

    fixGradients( ddx, ddy, texWidth, texHeight, filterMode, conservativeFiltering );

    // Determine the blend between linear and cubic filtering
    float cubicBlend = ( filterMode == FILTER_BICUBIC ) ? 1.0f : 0.0f;
    if( filterMode == FILTER_SMARTBICUBIC && maxGradientLengthInPixels < 2.0f )
    {
        cubicBlend = fminf( 2.0f - maxGradientLengthInPixels, 1.0f );
    }

    // Get mip level if needed
    float ml = 0.0f;
    int mipLevel = 0;
    if( dresultds || dresultdt || filterMode == FILTER_BICUBIC )
    {
        ml = getMipLevel( ddx, ddy, texWidth, texHeight, 1.0f / maxAnisotropy );
        if( mipmapFilterMode == CU_TR_FILTER_MODE_POINT )
            ml = fmaxf( 0.0f, ceilf( ml - 0.5f ) );
        mipLevel = floorf( ml );
        mipLevel = max( mipLevel, 0 );
    }
    int mipLevelWidth = max( texWidth >> mipLevel , 1);
    int mipLevelHeight = max( texHeight >> mipLevel , 1);

    // Get unnormalized texture coordinates
    float ts = s * mipLevelWidth - 0.5f;
    float tt = t * mipLevelHeight - 0.5f;
    float i = floorf( ts );
    float j = floorf( tt );

    // Bilinear and point sampling
    if( cubicBlend < 1.0f )
    {
        // Get bilinear sample
        if( result )
        {
            *result = ::tex2DGrad<TYPE>( texture, s, t, ddx, ddy );
        }

        // Do software interpolation on half pixel width grid (to line up with both coarse and fine mip levels)
        if( filterMode != FILTER_POINT && ( dresultds || dresultdt ) )
        {
            float ii = (ts-i > 0.5f) ? i+0.5f : i;
            float jj = (tt-j > 0.5f) ? j+0.5f : j;
            float s0 = (ii+0.5f) / mipLevelWidth;
            float s1 = (ii+1.0f) / mipLevelWidth;
            float t0 = (jj+0.5f) / mipLevelHeight;
            float t1 = (jj+1.0f) / mipLevelHeight;

            TYPE t00 = ::tex2DGrad<TYPE>( texture, s0, t0, ddx, ddy );
            TYPE t10 = ::tex2DGrad<TYPE>( texture, s1, t0, ddx, ddy );
            TYPE t01 = ::tex2DGrad<TYPE>( texture, s0, t1, ddx, ddy );
            TYPE t11 = ::tex2DGrad<TYPE>( texture, s1, t1, ddx, ddy );
            if( dresultds )
                *dresultds = lerp(t10-t00, t11-t01, 2.0f*(tt-jj)) * mipLevelWidth * 2.0f;
            if( dresultdt )
                *dresultdt = lerp(t01-t00, t11-t10, 2.0f*(ts-ii)) * mipLevelHeight * 2.0f;
        }

        if( cubicBlend <= 0.0f )
            return;
    }

    // Cubic filtering for result
    if( result )
    {
        float4 wx = cubicWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        TYPE res = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight );
        *result = lerp( res, *result, 1.0f-cubicBlend );
    }
    // Cubic filtering for derivatives
    if( dresultds || dresultdt )
    {
        float4 wx = cubicDerivativeWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        TYPE drds = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelWidth;
    
        wx = cubicWeights(ts - i);
        wy = cubicDerivativeWeights(tt - j);
        TYPE drdt = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelHeight;

        if( dresultds )
            *dresultds = lerp( *dresultds, drds, cubicBlend );

        if( dresultdt )
            *dresultdt = lerp( *dresultdt, drdt, cubicBlend );
    }

    // Return unless we have to blend between levels
    if( filterMode != FILTER_BICUBIC || mipmapFilterMode == FILTER_POINT || ml == mipLevel || ml <= 0.0f )
        return;

    //-------------------------------------------------------------------------------
    // Sample second mip level for FILTER_BICUBIC mode

    // Get unnormalized texture coordinates
    mipLevel++;
    mipLevelWidth = max(texWidth >> mipLevel, 1);
    mipLevelHeight = max(texHeight >> mipLevel, 1);
    ts = s * mipLevelWidth - 0.5f;
    tt = t * mipLevelHeight - 0.5f;
    i = floorf( ts );
    j = floorf( tt );

    float levelBlend = (mipLevel-ml);

    // Cubic filtering for result
    if( result )
    {
        float4 wx = cubicWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        *result = lerp( textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ), *result, levelBlend );
    }
    // Cubic filtering for derivatives
    if( dresultds || dresultdt )
    {
        float4 wx = cubicDerivativeWeights(ts - i);
        float4 wy = cubicWeights(tt - j);
        TYPE drds = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelWidth;

        wx = cubicWeights(ts - i);
        wy = cubicDerivativeWeights(tt - j);
        TYPE drdt = textureWeighted<TYPE>( texture, i, j, wx, wy, mipLevel, mipLevelWidth, mipLevelHeight ) * mipLevelHeight;

        if( dresultds )
            *dresultds = lerp( drds, *dresultds, levelBlend );

        if( dresultdt )
            *dresultdt = lerp( drdt, *dresultdt, levelBlend );
    }
}
