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

/// \file Texture2D.h
/// Device code for fetching from demand-loaded sparse textures.

//#include <optix.h>

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Paging.h>
#include <OptiXToolkit/DemandLoading/Texture2DFootprint.h>
#include <OptiXToolkit/DemandLoading/TextureCascade.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include <OptiXToolkit/ImageSource/ImageHelpers.h>
using namespace imageSource;

#ifndef DOXYGEN_SKIP
#define FINE_MIP_LEVEL false
#define COARSE_MIP_LEVEL true
#endif

namespace demandLoading {

#if defined( __CUDACC__ ) || defined( OPTIX_PAGING_BIT_OPS )

#ifndef DOXYGEN_SKIP

/// Request a cascade (texture resolution) big enough handle a sample with texture derivatives ddx, ddy
D_INLINE bool requestCascade( const DeviceContext& context, unsigned int textureId, const TextureSampler* sampler, float2 ddx, float2 ddy )
{
    if( sampler && !sampler->hasCascade )
        return false;

    // Get the mip level
    float texWidth = ( sampler ) ? sampler->width : CASCADE_BASE;
    float texHeight = ( sampler ) ? sampler->height : CASCADE_BASE;
    float anisotropy = ( sampler ) ? sampler->desc.maxAnisotropy : 16.0f;
    float mipLevel = getMipLevel( ddx, ddy, texWidth, texHeight, 1.0f / anisotropy );

    // The current size is large enough, don't request a cascade
    if( sampler && mipLevel >= 0.0f )
        return false;
    if( mipLevel >= 0.0f )
        mipLevel = 0.0f;

    // The current size is too small, request a cascade
    unsigned int cascadeStartIndex = context.pageTable.capacity; // FIXME: Should this be an explicit variable?
    unsigned int requestCascadeLevel = ( sampler ) ? sampler->cascadeLevel : 0;
    requestCascadeLevel += (unsigned int)( ceilf( -mipLevel ) );
    requestCascadeLevel = max( 0, min( requestCascadeLevel, NUM_CASCADES - 1 ) );
    unsigned int cascadePage = cascadeStartIndex + textureId * NUM_CASCADES + requestCascadeLevel;
    pagingRequest( context.referenceBits, cascadePage );
    return true;
}


#if __CUDA_ARCH__ >= 600
#define SPARSE_TEX_SUPPORT true
#endif

#ifdef SPARSE_TEX_SUPPORT

D_INLINE bool requestAndCheckResidency( unsigned int* referenceBits, unsigned int* residenceBits, unsigned int pageId )
{
    pagingRequest( referenceBits, pageId );
    return checkBitSet( pageId, residenceBits );
}

// Request a rectangular region centered on (x,y) having width (2*dx, 2*dy).
D_INLINE bool requestTexFootprint2DRect( const TextureSampler& sampler,
                                         unsigned int*         referenceBits,
                                         unsigned int*         residenceBits,
                                         float                 x,
                                         float                 y,
                                         float                 dx,
                                         float                 dy,
                                         unsigned int          mipLevel,
                                         unsigned int          singleMipLevel )
{
    // Handle mip tail for fine level
    if( mipLevel >= sampler.mipTailFirstLevel )
    {
        pagingRequest( referenceBits, sampler.startPage );
        return checkBitSet( sampler.startPage, residenceBits );
    }

    // Get the extent of the rectangle
    const CUaddress_mode wrapMode0 = static_cast<CUaddress_mode>( sampler.desc.wrapMode0 );
    const CUaddress_mode wrapMode1 = static_cast<CUaddress_mode>( sampler.desc.wrapMode1 );

    float x0 = wrapTexCoord( x - dx, wrapMode0 );
    float y0 = wrapTexCoord( y - dy, wrapMode1 );
    float x1 = wrapTexCoord( x + dx, wrapMode0 );
    float y1 = wrapTexCoord( y + dy, wrapMode1 );

    // Request bits in fine mip level
    TextureSampler::MipLevelSizes sizes = sampler.mipLevelSizes[mipLevel];
    unsigned int mipLevelStart = sampler.startPage + sizes.mipLevelStart;
    int xx0 = static_cast<int>(x0 * sizes.levelWidthInTiles);
    int yy0 = static_cast<int>(y0 * sizes.levelHeightInTiles);
    int xx1 = static_cast<int>(x1 * sizes.levelWidthInTiles);
    int yy1 = static_cast<int>(y1 * sizes.levelHeightInTiles);

    bool isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx0, yy0, sizes.levelWidthInTiles ) );
    if( xx0 != xx1 )
        isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx1, yy0, sizes.levelWidthInTiles ) ) && isResident;
    if( yy0 != yy1 )
        isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx0, yy1, sizes.levelWidthInTiles ) ) && isResident;
    if( xx0 != xx1 && yy0 != yy1 )
        isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx1, yy1, sizes.levelWidthInTiles ) ) && isResident;

    if( singleMipLevel )
        return isResident;

    // Handle mip tail for coarse level
    if( mipLevel + 1 >= sampler.mipTailFirstLevel )
    {
        pagingRequest( referenceBits, sampler.startPage );
        return checkBitSet( sampler.startPage, residenceBits ) || isResident;
    }

    // Request bits in coarse mip level
    sizes = sampler.mipLevelSizes[mipLevel + 1];
    mipLevelStart = sampler.startPage + sizes.mipLevelStart;
    xx0 = xx0 >> 1;
    yy0 = yy0 >> 1;
    xx1 = xx1 >> 1;
    yy1 = yy1 >> 1;

    isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx0, yy0, sizes.levelWidthInTiles ) ) && isResident;
    if( xx0 != xx1 )
        isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx1, yy0, sizes.levelWidthInTiles ) ) && isResident;
    if( yy0 != yy1 )
        isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx0, yy1, sizes.levelWidthInTiles ) ) && isResident;
    if( xx0 != xx1 && yy0 != yy1 )
        isResident = requestAndCheckResidency( referenceBits, residenceBits, mipLevelStart + getPageOffsetFromTileCoords( xx1, yy1, sizes.levelWidthInTiles ) ) && isResident;

    return isResident;
}


/// Request the footprint for a texture sample at coords (x,y) with texture gradients (dPdx_x, dPdx_y), (dPdy_x, dPdy_y).
D_INLINE bool requestTexFootprint2DGrad( const TextureSampler& sampler,
                                         unsigned int*         referenceBits,
                                         unsigned int*         residenceBits,
                                         float                 x,
                                         float                 y,
                                         float                 dPdx_x,
                                         float                 dPdx_y,
                                         float                 dPdy_x,
                                         float                 dPdy_y,
                                         float                 expandX,
                                         float                 expandY )
{
    const float MAX_SW_MIPLEVEL_ERROR = 0.18f;

#if defined OPTIX_VERSION && !defined NO_FOOTPRINT
    // Get the footprint for the fine level, to find out which mip levels are sampled.
    unsigned int desc = *reinterpret_cast<const unsigned int*>( &sampler.desc );
    unsigned int singleMipLevel;
    uint4 fp = optixTexFootprint2DGrad( sampler.texture, desc, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y, FINE_MIP_LEVEL, &singleMipLevel );
    Texture2DFootprint* finefp = reinterpret_cast<Texture2DFootprint*>( &fp );
    unsigned int fineLevel = finefp->level;
    unsigned int swFootprint = finefp->reserved1;
    // In a SW footprint, the mip level is approximate, so assume multiple mip levels
    if( swFootprint && ( sampler.desc.numMipLevels > 0 ) )
        singleMipLevel = false;
#else
    // Figure out which mip levels are sampled.
    float ml = getMipLevel( float2{dPdx_x, dPdx_y}, float2{dPdy_x, dPdy_y}, sampler.width, sampler.height, 1.0f / sampler.desc.maxAnisotropy );
    unsigned int fineLevel = ( ml >= 0.0f ) ? static_cast<unsigned int>( ml ) : 0;
    unsigned int swFootprint = ( ml >= -MAX_SW_MIPLEVEL_ERROR );  // no need to correct if texture magnified enough
    unsigned int singleMipLevel = ( ml < -MAX_SW_MIPLEVEL_ERROR || sampler.desc.numMipLevels <= 1 ) ? 1 : 0;
#endif

    // Request footprint for rectangular extent. (Expand footprint by (0.5f / sampler.width) to make sure it's at least a texel wide.)
    float dxmax = 0.5f * fmaxf( fabs( dPdx_x ), fabs( dPdy_x ) ) + ( 0.5f / sampler.width )  + expandX;
    float dymax = 0.5f * fmaxf( fabs( dPdx_y ), fabs( dPdy_y ) ) + ( 0.5f / sampler.height ) + expandY;

    bool isResident = requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dxmax, dymax, fineLevel, singleMipLevel );
    if( !isResident || !swFootprint )
        return isResident;

    // Handle mip level discrepancy between SW and HW mip level calculations.
    // If the calculated mip level is near an integer boundary, request the mip level on the other side.

    float mipLevel = getMipLevel( float2{dPdx_x, dPdx_y}, float2{dPdy_x, dPdy_y}, sampler.width, sampler.height, 1.0f / sampler.desc.maxAnisotropy );
    float fracLevel = mipLevel - floorf( mipLevel );

    if( fracLevel < MAX_SW_MIPLEVEL_ERROR && fineLevel > 0 )
        isResident = requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dxmax, dymax, fineLevel - 1, true );
    else if( fracLevel > 1.0f - MAX_SW_MIPLEVEL_ERROR )
        isResident = requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dxmax, dymax, fineLevel + 2, true );
    return isResident;
}


/// Request the footprint for a texture sample at coords (x,y) with the specified lod value.
D_INLINE bool requestTexFootprint2DLod( const TextureSampler& sampler,
                                        unsigned int*         referenceBits,
                                        unsigned int*         residenceBits,
                                        float                 x,
                                        float                 y,
                                        float                 lod )
{
    // Figure out which mip levels are sampled.
    unsigned int fineLevel = static_cast<unsigned int>( fmaxf( lod, 0.0f ) );
    fineLevel = min( fineLevel, sampler.desc.numMipLevels - 1 );
    unsigned int singleMipLevel = static_cast<unsigned int>( lod == fineLevel || sampler.desc.numMipLevels <= 1 ); 

    // Request footprint for rectangular extent.
    float dx = exp2f( fineLevel * 2.0f ) / sampler.width;
    float dy = exp2f( fineLevel * 2.0f ) / sampler.height;
    return requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dx, dy, fineLevel, singleMipLevel );
}

#endif  // SPARSE_TEX_SUPPORT

#endif  // ndef DOXYGEN_SKIP


/// Fetch the base color of a texture stored in the demand loader page table as a half4
template <class Sample> 
D_INLINE bool 
getBaseColor( const DeviceContext& context, unsigned int textureId, Sample& rval, bool* baseColorResident )
{
    const unsigned long long baseVal = pagingMapOrRequest( context, textureId + context.maxTextures, baseColorResident );
    if( *baseColorResident && baseVal != NO_BASE_COLOR )
    {
        const half4* baseColor = reinterpret_cast<const half4*>( &baseVal );
        convertType( *baseColor, rval );
        return true;
    }
    return false;
}

#endif  // ndef DOXYGEN_SKIP

/// Fetch from a demand-loaded texture with the specified identifer, obtained via DemandLoader::createTexture.
/// The given DeviceContext is typically a launch parameter, obtained via DemandLoader::launchPrepare,
/// that has been copied to device memory.
template <class Sample> D_INLINE Sample 
tex2DGrad( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident, float2 texelJitter )
{
    // Check for base color
    Sample rval;
    bool baseColorResident;
    convertType( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );

    const float minGradSquared = minf( ddx.x * ddx.x + ddx.y * ddx.y, ddy.x * ddy.x + ddy.y * ddy.y );
    if( minGradSquared >= 1.0f )
    {
        *isResident = getBaseColor<Sample>( context, textureId, rval, &baseColorResident );
        if( *isResident || !baseColorResident )
            return rval;
    }

    // Check whether the texture sampler is resident.
    const TextureSampler* sampler =
        reinterpret_cast<const TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    if( !sampler )
    {
        if( *isResident )
            *isResident = getBaseColor<Sample>( context, textureId, rval, &baseColorResident );
#ifdef REQUEST_CASCADE
        *isResident = *isResident && !requestCascade( context, textureId, sampler, ddx, ddy );
#endif
        return rval;
    }

    // Prevent footprint from exceeding min tile width for non-mipmapped textures
    if( sampler->desc.numMipLevels == 1 && sampler->desc.isSparseTexture )
    {
        float       pixelSpanX       = fmaxf( fabsf( ddx.x ), fabsf( ddy.x ) ) * sampler->width;
        float       pixelSpanY       = fmaxf( fabsf( ddx.y ), fabsf( ddy.y ) ) * sampler->height;
        float       pixelSpan        = fmaxf( pixelSpanX, pixelSpanY );
        const float halfMinTileWidth = 32.0f;  // half min tile width for sparse textures

        if( pixelSpan > halfMinTileWidth )
        {
            float scale = halfMinTileWidth / pixelSpan;
            ddx         = make_float2( ddx.x * scale, ddx.y * scale );
            ddy         = make_float2( ddy.x * scale, ddy.y * scale );
        }
    }

    // Jitter the texture coordinate
    x = x + (texelJitter.x / sampler->width);
    y = y + (texelJitter.y / sampler->height);

#ifdef SPARSE_TEX_SUPPORT
    // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
    *isResident = !sampler->desc.isSparseTexture;
    if( !context.requestIfResident )
        rval = ::tex2DGrad<Sample>( sampler->texture, x, y, ddx, ddy, isResident );

    // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
    if( *isResident == false && sampler->desc.isSparseTexture)
        *isResident = requestTexFootprint2DGrad( *sampler, context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y, 0.0f, 0.0f );

    // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
    if( *isResident && context.requestIfResident )
        rval = ::tex2DGrad<Sample>( sampler->texture, x, y, ddx, ddy ); // non-pedicated texture fetch
#else
    *isResident = true;
    rval = ::tex2DGrad<Sample>( sampler->texture, x, y, ddx, ddy );
#endif

#ifdef REQUEST_CASCADE
    *isResident = *isResident && !requestCascade( context, textureId, sampler, ddx, ddy );
#endif

    return rval;
}

template <class TYPE> D_INLINE TYPE
tex2DGrad( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident )
{
    return tex2DGrad<TYPE>( context, textureId, x, y, ddx, ddy, isResident, float2{0.0f} );
}


/// Fetch from a demand-loaded texture with the specified identifier, obtained via DemandLoader::createTexture.
/// The given DeviceContext is typically a launch parameter, obtained via DemandLoader::launchPrepare,
/// that has been copied to device memory.
template <class Sample> D_INLINE Sample 
tex2DLod( const DeviceContext& context, unsigned int textureId, float x, float y, float lod, bool* isResident, float2 texelJitter )
{
    // Check whether the texture sampler is resident.
    const TextureSampler* sampler =
        reinterpret_cast<const TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    
    Sample rval;
    convertType( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );
    if( *isResident == false )
    {        
        return rval;
    }

    // Prevent footprint from exceeding min tile width for non-mipmapped textures
    if( sampler && sampler->desc.numMipLevels == 1 )
        lod = 0.0f;

    // Check for base color.
    // Note: It would be preferable to check for baseColor before the sampler is loaded, but 
    // texture width and height are needed to determine if we are in the base color case from lod.
    float exp2Lod = exp2f( lod );
    if( !sampler || exp2Lod >= max( sampler->width, sampler->height ) )
    {
        bool baseColorResident;
        if( getBaseColor<Sample>( context, textureId, rval, &baseColorResident ) )
            return rval;
        *isResident = false;
        return rval;
    }

    // Jitter the texture coordinate
    x = x + (texelJitter.x / sampler->width);
    y = y + (texelJitter.y / sampler->height);

#ifdef SPARSE_TEX_SUPPORT
    // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
    *isResident = false;
    if( !context.requestIfResident )
        rval = ::tex2DLod<Sample>( sampler->texture, x, y, lod, isResident );

    // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
    if( !*isResident && sampler->desc.isSparseTexture )
        *isResident = requestTexFootprint2DLod( *sampler, context.referenceBits, context.residenceBits, x, y, lod );

    // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
    if( *isResident && context.requestIfResident )
        rval = ::tex2DLod<Sample>( sampler->texture, x, y, lod ); // non-pedicated texture fetch
#else
    *isResident = true;
    rval = ::tex2DLod<Sample>( sampler->texture, x, y, lod );
#endif

#ifdef REQUEST_CASCADE
    float2 ddx  = make_float2( exp2Lod / sampler->width, 0.0f );
    float2 ddy  = make_float2( 0.0f, exp2Lod / sampler->height );
    *isResident = *isResident && !requestCascade( context, textureId, sampler, ddx, ddy );
#endif

    return rval;
}

template <class TYPE> D_INLINE TYPE
tex2DLod( const DeviceContext& context, unsigned int textureId, float x, float y, float lod, bool* isResident )
{
    return tex2DLod<TYPE>( context, textureId, x, y, lod, isResident, float2{0.0f} );
}


/// Fetch from a demand-loaded texture with the specified identifier, obtained via DemandLoader::createTexture.
/// The given DeviceContext is typically a launch parameter, obtained via DemandLoader::launchPrepare,
/// that has been copied to device memory.
template <class TYPE> D_INLINE TYPE
tex2D( const DeviceContext& context, unsigned int textureId, float x, float y, bool* isResident, float2 texelJitter )
{
    return tex2DLod<TYPE>( context, textureId, x, y, 0.0f, isResident, texelJitter );
}

template <class TYPE> D_INLINE TYPE
tex2D( const DeviceContext& context, unsigned int textureId, float x, float y, bool* isResident )
{
    return tex2D<TYPE>( context, textureId, x, y, isResident, float2{0.0f} );
}

}  // namespace demandLoading
