//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <DemandLoading/Texture2D.h>

namespace demandLoading {

__device__ static __forceinline__ void wrapAndSeparateUdimCoord( float x, CUaddress_mode wrapMode, unsigned int udim, float& newx, unsigned int& xidx )
{
    newx = wrapTexCoord( x, wrapMode ) * udim;
    xidx = static_cast<unsigned int>( floorf( newx ) );
    xidx = (xidx < udim) ? xidx : 0; // fix problem that happens with -0.000
    newx -= floorf( newx );
}

/// Fetch from a demand-loaded udim texture.  A "udim" texture is an array of texture images that are treated as a single texture
/// object (with an optional base texture). This entry point is fast, but assumes that texture samples will not cross subtexture boundaries. 
/// When using this entry point, use CU_TR_ADDRESS_MODE_CLAMP when defining all subtextures to prevent dark lines between textures.
template <class TYPE>
__device__ static __forceinline__ TYPE
tex2DGradUdim( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident )
{
    // Check for base color
    TYPE rval;
    bool baseColorResident;

    const float minGradSquared = minf( ddx.x * ddx.x + ddx.y * ddx.y, ddy.x * ddy.x + ddy.y * ddy.y );
    if( minGradSquared >= 1.0f )
    {
        *isResident = true;
        if ( getBaseColor<TYPE>( context, textureId, rval, &baseColorResident ) )
        {
            return rval;
        }
        if( !baseColorResident ) // Don't request the sampler unless we really need to
        {
            convertColor( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );
            return rval;
        }
    }

    // Get the base texture
    TextureSampler* baseSampler = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    if( !baseSampler )
    {
        convertColor( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );
        if( *isResident )
            *isResident = getBaseColor<TYPE>( context, textureId, rval, &baseColorResident );
        return rval;
    }

    // Compute mip level in base texture
    const unsigned int udim              = baseSampler->udim;
    const unsigned int vdim              = baseSampler->vdim;
    const unsigned int isUdimBaseTexture = baseSampler->desc.isUdimBaseTexture;
    const unsigned int baseWidth         = isUdimBaseTexture ? baseSampler->width : 1u;
    const unsigned int baseHeight        = isUdimBaseTexture ? baseSampler->height : 1u;

    // Determine the mip level for the base texture
    float mipLevel = 0.0f;
    if( udim > 0 )  // udim is 0 for non-udim textures.
        mipLevel = getMipLevel( ddx, ddy, baseWidth, baseHeight, 1.0f / baseSampler->desc.maxAnisotropy );

    // Sample the subtexture
    if( mipLevel < 0.0f || ( !isUdimBaseTexture && udim > 0 ) )
    {
        float        sx, sy;
        unsigned int xidx, yidx;
        wrapAndSeparateUdimCoord( x, CU_TR_ADDRESS_MODE_WRAP, udim, sx, xidx );
        wrapAndSeparateUdimCoord( y, CU_TR_ADDRESS_MODE_WRAP, vdim, sy, yidx );

        unsigned int subTexId = baseSampler->udimStartPage + yidx * udim + xidx;
        const float2 ddx_dim = make_float2( ddx.x * udim, ddx.y * vdim );
        const float2 ddy_dim = make_float2( ddy.x * udim, ddy.y * vdim );
        rval = tex2DGrad<TYPE>( context, subTexId, sx, sy, ddx_dim, ddy_dim, isResident );
        if( *isResident )
            return rval;
    }

    // If the mip level was coarse enough (or not a udim texture), use the base texture if one exists.
    if( isUdimBaseTexture || udim == 0 )
    {
        // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
        *isResident = !baseSampler->desc.isSparseTexture;
        if( context.requestIfResident == false )
            rval = tex2DGrad<TYPE>( baseSampler->texture, x, y, ddx, ddy, isResident );

        // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
        if( *isResident == false  && baseSampler->desc.isSparseTexture)
            *isResident = requestTexFootprint2DGrad( *baseSampler, context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y );

        // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
        if( *isResident && context.requestIfResident )
            rval = tex2DGrad<TYPE>( baseSampler->texture, x, y, ddx, ddy ); // non-pedicated texture fetch
    }
    return rval;
}


/// Fetch from demand-loaded udim texture.  A "udim" texture is an array of texture images that are treated as a single texture
/// object (with an optional base texture).  This entry point will combine multiple samples to blend across subtexture boundaries.  
/// For proper blending, use CU_TR_ADDRESS_MODE_BORDER when defining all subtextures. Other blending modes will show lines 
/// between subtextures.
template <class TYPE>
__device__ static __forceinline__ TYPE
tex2DGradUdimBlend( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident )
{
    // Check for base color
    TYPE rval;
    bool baseColorResident;
    convertColor( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );

    float minGradSquared = minf( ddx.x * ddx.x + ddx.y * ddx.y, ddy.x * ddy.x + ddy.y * ddy.y );
    if( minGradSquared >= 1.0f )
    {
        *isResident = true;
        if ( getBaseColor<TYPE>( context, textureId, rval, &baseColorResident ) )
            return rval;
        if( !baseColorResident ) // Don't request the sampler unless we really need to
            return rval;
    }

    // Get the base texture
    TextureSampler* baseSampler = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    if( !baseSampler )
    {
        if( *isResident )
            *isResident = getBaseColor<TYPE>( context, textureId, rval, &baseColorResident );  
        return rval;
    }

    // Compute mip level in base texture
    const unsigned int udim              = baseSampler->udim;
    const unsigned int vdim              = baseSampler->vdim;
    const unsigned int isUdimBaseTexture = baseSampler->desc.isUdimBaseTexture;
    const unsigned int baseWidth         = isUdimBaseTexture ? baseSampler->width : 1u;
    const unsigned int baseHeight        = isUdimBaseTexture ? baseSampler->height : 1u;

    // Set up minGradSquared for subtextures. This needs to be done before clamping large gradients
    minGradSquared = minf( ddx.x * ddx.x * udim * udim + ddx.y * ddx.y * vdim * vdim,
                           ddy.x * ddy.x * udim * udim + ddy.y * ddy.y * vdim * vdim );

    // Find the xy extents of the texture gradients
    float dx = fmax( fabsf( ddx.x ), fabsf( ddy.x ) );
    float dy = fmax( fabsf( ddx.y ), fabsf( ddy.y ) );

    // Clamp large gradients to prevent the texture footprint from spanning
    // more than half a subtexture, which could cause artifacts at subtexture boundaries.
    float mx = maxf( dx * udim, dy * vdim );
    if( mx > 0.25f )
    {
        float scale = 0.25f / mx;
        ddx *= scale;
        ddy *= scale;
        dx *= scale;
        dy *= scale;
    }

    // Determine the mip level for the base texture
    float mipLevel = 0.0f;
    if( udim > 0 )
        mipLevel = getMipLevel( ddx, ddy, baseWidth, baseHeight, 1.0f / baseSampler->desc.maxAnisotropy );

    // If the mip level is coarse enough, use the base texture if one exists.
    if( mipLevel >= 0.0f && ( isUdimBaseTexture || udim == 0 ) )
    {
        // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
        *isResident = !baseSampler->desc.isSparseTexture;
        if( context.requestIfResident == false )
            rval = tex2DGrad<TYPE>( baseSampler->texture, x, y, ddx, ddy, isResident );

        // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
        if( *isResident == false  && baseSampler->desc.isSparseTexture)
            *isResident = requestTexFootprint2DGrad( *baseSampler, context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y );

        // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
        if( *isResident && context.requestIfResident )
            rval = tex2DGrad<TYPE>( baseSampler->texture, x, y, ddx, ddy ); // non-pedicated texture fetch

        return rval;
    }

    // sample from up to 4 subtextures and add the results. (this only works if the subtextures are defined
    // using CU_TR_ADDRESS_MODE_BORDER, which puts black on the edges. Adding multiple samples in border mode blends
    // across texture boundaries.)  If subtextures are not found, use the baseSampler instead, if available.
    TextureSampler* samplers[4] = {0, 0, 0, 0};

    // Find subtexture for texture coordinate (x,y)
    const CUaddress_mode wrapMode0 = CU_TR_ADDRESS_MODE_WRAP;  // always using wrap mode
    const CUaddress_mode wrapMode1 = CU_TR_ADDRESS_MODE_WRAP;

    float        sx, sy, sx0, sy0, sx1, sy1;
    unsigned int xidx, yidx, xidx0, yidx0, xidx1, yidx1;
    wrapAndSeparateUdimCoord( x, wrapMode0, udim, sx, xidx );
    wrapAndSeparateUdimCoord( y, wrapMode1, vdim, sy, yidx );

    bool         oneSampler     = true;
    bool         subTexResident = true;
    unsigned int subTexId       = baseSampler->udimStartPage + yidx * udim + xidx;

    // Check for base color
    if( minGradSquared >= 1.0f )
    {
        if( getBaseColor<TYPE>( context, subTexId, rval, &baseColorResident ) )
            return rval;
        if( !baseColorResident ) // don't request sampler unless we really need to
            return rval;
    }
    
    samplers[0] = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, subTexId, &subTexResident ) );
    *isResident = static_cast<bool>( samplers[0] );

    if( !samplers[0] )
    {
        if( !isUdimBaseTexture )
        {
            if( subTexResident )
            {
                if( getBaseColor<TYPE>( context, subTexId, rval, isResident ) )
                    return rval;
            }
            return rval;
        }
        samplers[0] = baseSampler;
    }
    else
    {
        // Add in a fudge factor to the gradient extents account for the black edge of a texture in border mode.
        // (The fudge factor must be at least half a texel width at the mip level being sampled to make sure
        // all needed udim textures are sampled.)
        // Note that this is calculated in the baseSampler texture coordinates.
        float xcorrect1 = 0.5f * exp2f( mipLevel ) / static_cast<float>( baseWidth );
        float ycorrect1 = 0.5f * exp2f( mipLevel ) / static_cast<float>( baseHeight );
        float xcorrect2 = 0.5f / static_cast<float>( min( udim, vdim ) * samplers[0]->width );
        float ycorrect2 = 0.5f / static_cast<float>( min( udim, vdim ) * samplers[0]->height );
        dx += xcorrect1 + xcorrect2;
        dy += ycorrect1 + ycorrect2;

        // Handle case of sampling multiple textures
        if( ( sx + dx * udim > 1.0f ) || ( sx - dx * udim < 0.0f ) || ( sy + dy * vdim > 1.0f ) || ( sy - dy * vdim < 0.0f ) )
        {
            // Get extent of texture sample in udim textures
            wrapAndSeparateUdimCoord( x - dx, wrapMode0, udim, sx0, xidx0 );
            wrapAndSeparateUdimCoord( y - dy, wrapMode1, vdim, sy0, yidx0 );
            wrapAndSeparateUdimCoord( x + dx, wrapMode0, udim, sx1, xidx1 );
            wrapAndSeparateUdimCoord( y + dy, wrapMode1, vdim, sy1, yidx1 );

            // Try to load each of the samplers
            subTexId    = baseSampler->udimStartPage + yidx0 * udim + xidx0;
            samplers[0] = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, subTexId, &subTexResident ) );
            *isResident = static_cast<bool>( samplers[0] );
            if( xidx1 != xidx0 )
            {
                subTexId = baseSampler->udimStartPage + yidx0 * udim + xidx1;
                samplers[1] = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, subTexId, &subTexResident ) );
                *isResident = *isResident && static_cast<bool>( samplers[1] );
                oneSampler = false;
            }
            if( yidx1 != yidx0 )
            {
                subTexId = baseSampler->udimStartPage + yidx1 * udim + xidx0;
                samplers[2] = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, subTexId, &subTexResident ) );
                *isResident = *isResident && static_cast<bool>( samplers[2] );
                oneSampler = false;
            }
            if( xidx1 != xidx0 && yidx1 != yidx0 )
            {
                subTexId = baseSampler->udimStartPage + yidx1 * udim + xidx1;
                samplers[3] = reinterpret_cast<TextureSampler*>( pagingMapOrRequest( context, subTexId, &subTexResident ) );
                *isResident = *isResident && static_cast<bool>( samplers[3] );
            }

            if( *isResident )
            {
                // Update texture coordinate to be coord for samplers[0]. Other coordinates are
                // offset by 1 in x and/or y.
                x = sx0 + dx * udim;
                y = sy0 + dy * vdim;
            }
            else
            {
                // If some of the samplers are not resident, revert to the base sampler
                if( !isUdimBaseTexture )
                {
                    *isResident = false;
                    return rval;
                }
                samplers[0] = baseSampler;
                samplers[1] = samplers[2] = samplers[3] = 0;
                oneSampler = true;
            }

            // Uncomment to show boundary between textures
            //convertColor( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );
            //return rval;
        }
        else
        {
            x = sx;
            y = sy;
        }
    }

    // Convert texture gradients to udim texture coordinates if all subtextures are resident
    if( *isResident )
    {
        ddx *= make_float2( udim, vdim );
        ddy *= make_float2( udim, vdim );
    }

    // Do the sampling and combine the results (just add them together)
    rval = TYPE();
    for( unsigned int i = 0; i < 4; ++i )
    {
        if( samplers[i] == 0 ) // skip null samplers
            continue;

        // Fix up texture coordinates based on which of the 4 samplers we are using
        float xx = x - static_cast<float>( i & 1 );
        float yy = y - static_cast<float>( i >> 1 );

        // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
        bool texResident = !samplers[i]->desc.isSparseTexture;
        if( context.requestIfResident == false )
            rval += tex2DGrad<TYPE>( samplers[i]->texture, xx, yy, ddx, ddy, &texResident );

        // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
        if( texResident == false && samplers[i]->desc.isSparseTexture)
            texResident = requestTexFootprint2DGrad( *samplers[i], context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y );

        // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
        if( texResident && context.requestIfResident )
            rval += tex2DGrad<TYPE>( samplers[i]->texture, xx, yy, ddx, ddy ); // non-pedicated texture fetch

        *isResident = *isResident && texResident;
        if( oneSampler )  // Early exit
            break;
    }

    return rval;
}

}  // namespace demandLoading
