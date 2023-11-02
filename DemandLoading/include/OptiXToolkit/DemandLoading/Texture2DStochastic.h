//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file Texture2DExtended.h 
/// Extended device-side entry points for fetching from demand-loaded sparse textures.

#include <OptiXToolkit/DemandLoading/Texture2DExtended.h>

namespace demandLoading {

/// Fetch from a demand-loaded texture with the specified identifer, obtained via DemandLoader::createTexture.
/// The given DeviceContext is typically a launch parameter, obtained via DemandLoader::launchPrepare,
/// that has been copied to device memory.
template <class TYPE> D_INLINE TYPE 
tex2DGradStochastic( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident, float2 xi )
{
    // Check for base color
    TYPE rval;
    bool baseColorResident;
    convertType( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );

    const float minGradSquared = minf( ddx.x * ddx.x + ddx.y * ddx.y, ddy.x * ddy.x + ddy.y * ddy.y );
    if( minGradSquared >= 1.0f )
    {
        *isResident = getBaseColor<TYPE>( context, textureId, rval, &baseColorResident );
        if( *isResident || !baseColorResident )
            return rval;
    }

    // Check whether the texture sampler is resident.
    const TextureSampler* sampler =
        reinterpret_cast<const TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    if( !sampler )
    {
        if( *isResident )
            *isResident = getBaseColor<TYPE>( context, textureId, rval, &baseColorResident );
#ifdef REQUEST_CASCADE
        *isResident = *isResident && !requestCascade( context, textureId, sampler, ddx, ddy );
#endif
        return rval;
    }

    // Prevent footprint from exceeding min tile width for non-mipmapped textures
    if( sampler->desc.numMipLevels == 1 && sampler->desc.isSparseTexture )
    {
        float       pixelSpanX       = maxf( fabsf( ddx.x ), fabsf( ddy.x ) ) * sampler->width;
        float       pixelSpanY       = maxf( fabsf( ddx.y ), fabsf( ddy.y ) ) * sampler->height;
        float       pixelSpan        = maxf( pixelSpanX, pixelSpanY );
        const float halfMinTileWidth = 32.0f;  // half min tile width for sparse textures

        if( pixelSpan > halfMinTileWidth )
        {
            float scale = halfMinTileWidth / pixelSpan;
            ddx         = make_float2( ddx.x * scale, ddx.y * scale );
            ddy         = make_float2( ddy.x * scale, ddy.y * scale );
        }
    }

    // Jitter
    x = x + (xi.x / sampler->width);
    y = y + (xi.y / sampler->height);

#ifdef SPARSE_TEX_SUPPORT
    // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
    *isResident = !sampler->desc.isSparseTexture;
    if( context.requestIfResident == false )
        rval = ::tex2DGrad<TYPE>( sampler->texture, x, y, ddx, ddy, isResident );

    // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
    if( *isResident == false && sampler->desc.isSparseTexture)
        *isResident = requestTexFootprint2DGrad( *sampler, context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y );

    // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
    if( *isResident && context.requestIfResident )
        rval = ::tex2DGrad<TYPE>( sampler->texture, x, y, ddx, ddy ); // non-pedicated texture fetch
#else
    *isResident = true;
    rval = ::tex2DGrad<TYPE>( sampler->texture, x, y, ddx, ddy );
#endif

#ifdef REQUEST_CASCADE
    *isResident = *isResident && !requestCascade( context, textureId, sampler, ddx, ddy );
#endif

    return rval;
}


/// Fetch from demand-loaded udim texture.  A "udim" texture is an array of texture images that are treated as a single texture
/// object (with an optional base texture).  This entry point does not combine multiple samples to blend across subtexture boundaries.
/// Use CU_TR_ADDRESS_MODE_CLAMP when defining all subtextures. Other blending modes will show lines between subtextures.
template <class TYPE> D_INLINE TYPE
tex2DGradUdimStochastic( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident, float2 xi )
{
    TYPE rval{};
    TextureSampler* bsmp = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) ); // base sampler

    // Use base texture if it's not a udim texture, if the mip level fits in the base texture, or if the base texture has a cascade.
    bool useBaseTexture = ( !bsmp ) || ( bsmp && bsmp->udim == 0 );
    if( !useBaseTexture && bsmp->desc.isUdimBaseTexture )
    {
        float mipLevel = getMipLevel( ddx, ddy, bsmp->width, bsmp->height, 1.0f / bsmp->desc.maxAnisotropy );
        useBaseTexture = ( mipLevel >= 0.0f ) || bsmp->hasCascade;
    }

    // Sample the subtexture
    if( !useBaseTexture )
    {
        float        sx, sy;
        unsigned int xidx, yidx;
        wrapAndSeparateUdimCoord( x, CU_TR_ADDRESS_MODE_WRAP, bsmp->udim, sx, xidx );
        wrapAndSeparateUdimCoord( y, CU_TR_ADDRESS_MODE_WRAP, bsmp->vdim, sy, yidx );

        unsigned int subTexId = bsmp->udimStartPage + ( yidx * bsmp->udim + xidx );
        const float2 ddx_dim = make_float2( ddx.x * bsmp->udim, ddx.y * bsmp->vdim );
        const float2 ddy_dim = make_float2( ddy.x * bsmp->udim, ddy.y * bsmp->vdim );
        rval = tex2DGradStochastic<TYPE>( context, subTexId, sx, sy, ddx_dim, ddy_dim, isResident, xi );

        if( *isResident || !bsmp->desc.isUdimBaseTexture )
            return rval;
    }

    // Sample the base texture
    rval = tex2DGradStochastic<TYPE>( context, textureId, x, y, ddx, ddy, isResident, xi );
    *isResident = *isResident && useBaseTexture;
    return rval;
}


/// Fetch from demand-loaded udim texture.  A "udim" texture is an array of texture images that are treated as a single texture
/// object (with an optional base texture).  This entry point will combine multiple samples to blend across subtexture boundaries.
/// Use CU_TR_ADDRESS_MODE_BORDER when defining all subtextures. Other blending modes will show lines between subtextures.
template <class TYPE> D_INLINE TYPE
tex2DGradUdimBlendStochastic( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident, float2 xi )
{
    TYPE rval{};
    TextureSampler* bsmp = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) ); // base sampler

    // Use base texture if it's not a udim texture, if the mip level fits in the base texture, or if the base texture has a cascade.
    bool useBaseTexture = ( !bsmp ) || ( bsmp && bsmp->udim == 0 );
    if( !useBaseTexture && bsmp->desc.isUdimBaseTexture )
    {
        float mipLevel = getMipLevel( ddx, ddy, bsmp->width, bsmp->height, 1.0f / bsmp->desc.maxAnisotropy );
        useBaseTexture = ( mipLevel >= 0.0f ) || bsmp->hasCascade;
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
        float mx = maxf( dx * udim, dy * vdim );
        if( mx > 0.25f && mx < 1.0f )
        {
            float scale = 0.25f / mx;
            ddx *= scale;
            ddy *= scale;
            dx *= scale;
            dy *= scale;
        }

        // Scale the gradients
        const float2 ddx_dim = ddx * float2{float(udim), float(vdim)};
        const float2 ddy_dim = ddy * float2{float(udim), float(vdim)};

        // Add in a fudge factor to the gradient extents to account for the black edge of a texture in border mode.
        // (The fudge factor must be at least half a texel width at the mip level being sampled.)
        // FIXME: this only works if both subtextures have the same dimensions.  Otherwise, the sample weight on the
        // border will end up != 1, resulting in a dark border.
        const float subtexProxySize = 64.0f;
        const float mipLevelCorrection = 0.5f * exp2f( getMipLevel( ddx, ddy, 1.0f, 1.0f, 1.0f / 16.0f ) );
        const float magnificationCorrection = 0.5f / static_cast<float>( min( udim, vdim ) * subtexProxySize );
        dx += ( mipLevelCorrection + magnificationCorrection );
        dy += ( mipLevelCorrection + magnificationCorrection );

        // Find subtexture for texture coordinate (x,y)
        float        sx, sy, sx0, sy0, sx1, sy1;
        unsigned int xidx, yidx, xidx0, yidx0, xidx1, yidx1;
        wrapAndSeparateUdimCoord( x, CU_TR_ADDRESS_MODE_WRAP, udim, sx, xidx );
        wrapAndSeparateUdimCoord( y, CU_TR_ADDRESS_MODE_WRAP, vdim, sy, yidx );

        // Find the bounds of sample footprint in udim coords
        wrapAndSeparateUdimCoord( x - dx, CU_TR_ADDRESS_MODE_WRAP, udim, sx0, xidx0 );
        wrapAndSeparateUdimCoord( y - dy, CU_TR_ADDRESS_MODE_WRAP, vdim, sy0, yidx0 );
        wrapAndSeparateUdimCoord( x + dx, CU_TR_ADDRESS_MODE_WRAP, udim, sx1, xidx1 );
        wrapAndSeparateUdimCoord( y + dy, CU_TR_ADDRESS_MODE_WRAP, vdim, sy1, yidx1 );

        // special case for base colors - force (xidx0,yidx0) to equal (xidx,yidx)
        xidx0 = ( mx >= 1.0f ) ? xidx : xidx0;
        yidx0 = ( mx >= 1.0f ) ? yidx : yidx0;

        // Try to sample up to 4 subtextures
        bool subTexResident;
        unsigned int subTexId = bsmp->udimStartPage + ( yidx0 * udim + xidx0 );
        float xoff = ( xidx != xidx0 ) ? 1.0f : 0.0f;
        float yoff = ( yidx != yidx0 ) ? 1.0f : 0.0f;
        rval = tex2DGrad<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx_dim, ddy_dim, &subTexResident );
        *isResident = subTexResident;

        // Special case for base colors - don't blend with neighbors
        if( mx >= 1.0f )
            return rval;

        if( xidx1 != xidx0 )
        {
            subTexId = bsmp->udimStartPage + ( yidx0 * udim + xidx1 );
            xoff = ( xidx != xidx1 ) ? -1.0f : 0.0f;
            yoff = ( yidx != yidx0 ) ? 1.0f : 0.0f;
            rval += tex2DGradStochastic<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx_dim, ddy_dim, &subTexResident, xi );
            *isResident = *isResident && subTexResident;
        }
        if( yidx1 != yidx0 )
        {
            subTexId = bsmp->udimStartPage + ( yidx1 * udim + xidx0 );
            xoff = ( xidx != xidx0 ) ? 1.0f : 0.0f;
            yoff = ( yidx != yidx1 ) ? -1.0f : 0.0f;
            rval += tex2DGradStochastic<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx_dim, ddy_dim, &subTexResident, xi );
            *isResident = *isResident && subTexResident;
        }
        if( xidx1 != xidx0 && yidx1 != yidx0 )
        {
            subTexId = bsmp->udimStartPage + ( yidx1 * udim + xidx1 );
            xoff = ( xidx != xidx1 ) ? -1.0f : 0.0f;
            yoff = ( yidx != yidx1 ) ? -1.0f : 0.0f;
            rval += tex2DGradStochastic<TYPE>( context, subTexId, sx + xoff, sy + yoff, ddx_dim, ddy_dim, &subTexResident, xi );
            *isResident = *isResident && subTexResident;
        }

        if( *isResident || !bsmp->desc.isUdimBaseTexture )
            return rval;
    }

    // Sample the base texture
    rval = tex2DGradStochastic<TYPE>( context, textureId, x, y, ddx, ddy, isResident, xi );
    *isResident = *isResident && useBaseTexture;
    return rval;
}

}  // namespace demandLoading
