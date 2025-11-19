// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file Texture2DExtended.h 
/// Extended device-side entry points for fetching from demand-loaded sparse textures.

#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
using namespace otk;

namespace demandLoading {

/// Fetch from demand-loaded udim texture.  A "udim" texture is an array of texture images that are treated as a single texture
/// object (with an optional base texture).  This entry point does not combine multiple samples to blend across subtexture boundaries.
/// Use CU_TR_ADDRESS_MODE_CLAMP when defining all subtextures. Other blending modes will show lines between subtextures.
template <class TYPE> D_INLINE TYPE
tex2DGradUdim( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident, float2 texelJitter )
{
    TYPE rval{};
    TextureSampler* bsmp = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) ); // base sampler

    // Use base texture if it's not a udim texture, if the mip level fits in the base texture, or if the base texture has a cascade.
    bool useBaseTexture = ( !bsmp ) || ( bsmp && bsmp->udim == 0 );
    if( !useBaseTexture && bsmp->desc.isUdimBaseTexture )
    {
        float mipLevel = getMipLevel( ddx, ddy, bsmp->width * bsmp->udim, bsmp->height * bsmp->vdim, 1.0f / bsmp->desc.maxAnisotropy );
        useBaseTexture = ( mipLevel >= 0.0f ) || bsmp->hasCascade;
        if( useBaseTexture )
            texelJitter = float2{0.0f};
    }

    // Sample the subtexture
    if( !useBaseTexture )
    {
        float        sx, sy;
        unsigned int xidx, yidx;
        separateUdimCoord( x, CU_TR_ADDRESS_MODE_WRAP, bsmp->udim, sx, xidx );
        separateUdimCoord( y, CU_TR_ADDRESS_MODE_WRAP, bsmp->vdim, sy, yidx );

        unsigned int subTexId = bsmp->udimStartPage + ( yidx * bsmp->udim + xidx ) * bsmp->numChannelTextures;
        rval = tex2DGrad<TYPE>( context, subTexId, sx, sy, ddx, ddy, isResident, texelJitter );

        if( *isResident || !bsmp->desc.isUdimBaseTexture )
            return rval;
    }

    // Scale the gradients for the base texture
    if( bsmp && bsmp->udim != 0 )
    {
        ddx /= bsmp->udim;
        ddy /= bsmp->vdim;
    }

    // Sample the base texture
    rval = tex2DGrad<TYPE>( context, textureId, x, y, ddx, ddy, isResident, texelJitter );
    *isResident = *isResident && useBaseTexture;
    return rval;
}

template <class TYPE> D_INLINE TYPE
tex2DGradUdim( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident )
{
    return tex2DGradUdim<TYPE>( context, textureId, x, y, ddx, ddy, isResident, float2{0.0f} );
}


/// Fetch from demand-loaded udim texture.  A "udim" texture is an array of texture images that are treated as a single texture
/// object (with an optional base texture).  This entry point will combine multiple samples to blend across subtexture boundaries.
/// Use CU_TR_ADDRESS_MODE_BORDER when defining all subtextures. Other blending modes will show lines between subtextures.
template <class TYPE> D_INLINE TYPE
tex2DGradUdimBlend( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident, float2 texelJitter )
{
    TYPE rval{};
    TextureSampler* bsmp = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) ); // base sampler

    // Use base texture if it's not a udim texture, if the mip level fits in the base texture, or if the base texture has a cascade.
    bool useBaseTexture = ( !bsmp ) || ( bsmp && bsmp->udim == 0 );
    if( !useBaseTexture && bsmp->desc.isUdimBaseTexture )
    {
        float mipLevel = getMipLevel( ddx, ddy, bsmp->width * bsmp->udim, bsmp->height * bsmp->vdim, 1.0f / bsmp->desc.maxAnisotropy );
        useBaseTexture = ( mipLevel >= 0.0f ) || bsmp->hasCascade;
        if( useBaseTexture )
            texelJitter = float2{0.0f};
    }

    // Sample subtextures
    if( !useBaseTexture )
    {
        const unsigned int udim = bsmp->udim;
        const unsigned int vdim = bsmp->vdim;

        // Find the xy extents of the texture gradients
        float dx = maxf( fabsf( ddx.x ), fabsf( ddy.x ) );
        float dy = maxf( fabsf( ddx.y ), fabsf( ddy.y ) );

        // Clamp large gradients (that are < 1) to prevent the texture footprint from spanning
        // more than half a subtexture, which could cause artifacts at subtexture boundaries.
        float mx = maxf( dx, dy );
        if( mx > 0.25f && mx < 1.0f )
        {
            float scale = 0.25f / mx;
            ddx *= scale;
            ddy *= scale;
            dx *= scale;
            dy *= scale;
        }

        // Add in a fudge factor to the gradient extents to account for the black edge of a texture in border mode.
        // (The fudge factor must be at least half a texel width at the mip level being sampled.)
        // FIXME: this only works if both subtextures have the same dimensions.  Otherwise, the sample weight on the
        // border will end up != 1, resulting in a dark border.
        const float mipLevelCorrection = 0.5f * exp2f( getMipLevel( ddx, ddy, 1.0f, 1.0f, 1.0f / 16.0f ) );
        const float magnificationCorrection = 0.5f / 64.0f; // half pixel width in smallest tile size
        dx += ( mipLevelCorrection + magnificationCorrection );
        dy += ( mipLevelCorrection + magnificationCorrection );

        // Find subtexture for texture coordinate (x,y)
        float        sx, sy, sx0, sy0, sx1, sy1;
        unsigned int xidx, yidx, xidx0, yidx0, xidx1, yidx1;
        separateUdimCoord( x, CU_TR_ADDRESS_MODE_WRAP, udim, sx, xidx );
        separateUdimCoord( y, CU_TR_ADDRESS_MODE_WRAP, vdim, sy, yidx );

        // Find the bounds of sample footprint in udim coords
        separateUdimCoord( x - dx, CU_TR_ADDRESS_MODE_WRAP, udim, sx0, xidx0 );
        separateUdimCoord( y - dy, CU_TR_ADDRESS_MODE_WRAP, vdim, sy0, yidx0 );
        separateUdimCoord( x + dx, CU_TR_ADDRESS_MODE_WRAP, udim, sx1, xidx1 );
        separateUdimCoord( y + dy, CU_TR_ADDRESS_MODE_WRAP, vdim, sy1, yidx1 );

        // special case for base colors - force (xidx0,yidx0) to equal (xidx,yidx)
        xidx0 = ( mx >= 1.0f ) ? xidx : xidx0;
        yidx0 = ( mx >= 1.0f ) ? yidx : yidx0;

        // Try to sample up to 4 subtextures
        bool subTexResident;
        unsigned int subTexId = bsmp->udimStartPage + ( yidx0 * udim + xidx0 ) * bsmp->numChannelTextures;
        float xoff = ( xidx != xidx0 ) ? 1.0f : 0.0f;
        float yoff = ( yidx != yidx0 ) ? 1.0f : 0.0f;
        rval = tex2DGrad<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx, ddy, &subTexResident, texelJitter );
        *isResident = subTexResident;

        // Special case for base colors - don't blend with neighbors
        if( mx >= 1.0f )
            return rval;

        if( xidx1 != xidx0 )
        {
            subTexId = bsmp->udimStartPage + ( yidx0 * udim + xidx1 ) * bsmp->numChannelTextures;
            xoff = ( xidx != xidx1 ) ? -1.0f : 0.0f;
            yoff = ( yidx != yidx0 ) ? 1.0f : 0.0f;
            rval += tex2DGrad<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx, ddy, &subTexResident, texelJitter );
            *isResident = *isResident && subTexResident;
        }
        if( yidx1 != yidx0 )
        {
            subTexId = bsmp->udimStartPage + ( yidx1 * udim + xidx0 ) * bsmp->numChannelTextures;
            xoff = ( xidx != xidx0 ) ? 1.0f : 0.0f;
            yoff = ( yidx != yidx1 ) ? -1.0f : 0.0f;
            rval += tex2DGrad<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx, ddy, &subTexResident, texelJitter );
            *isResident = *isResident && subTexResident;
        }
        if( xidx1 != xidx0 && yidx1 != yidx0 )
        {
            subTexId = bsmp->udimStartPage + ( yidx1 * udim + xidx1 ) * bsmp->numChannelTextures;
            xoff = ( xidx != xidx1 ) ? -1.0f : 0.0f;
            yoff = ( yidx != yidx1 ) ? -1.0f : 0.0f;
            rval += tex2DGrad<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx, ddy, &subTexResident, texelJitter );
            *isResident = *isResident && subTexResident;
        }

        if( *isResident || !bsmp->desc.isUdimBaseTexture )
            return rval;
    }

    // Scale the gradients for the base texture
    if( bsmp && bsmp->udim != 0 )
    {
        ddx /= bsmp->udim;
        ddy /= bsmp->vdim;
    }

    // Sample the base texture
    rval = tex2DGrad<TYPE>( context, textureId, x, y, ddx, ddy, isResident, texelJitter );
    *isResident = *isResident && useBaseTexture;
    return rval;
}

template <class TYPE> D_INLINE TYPE
tex2DGradUdimBlend( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident )
{
    return tex2DGradUdimBlend<TYPE>( context, textureId, x, y, ddx, ddy, isResident, float2{0.0f} );
}

}  // namespace demandLoading
