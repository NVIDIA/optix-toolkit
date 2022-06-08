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

/// \file Texture2D.h
/// Device code for fetching from demand-loaded sparse textures.

#include <optix.h>

#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/Paging.h>
#include <DemandLoading/Texture2DFootprint.h>
#include <DemandLoading/TextureSampler.h>
#include <DemandLoading/TileIndexing.h>

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include <cuda_fp16.h>  // to facilitate tex2D<half> in user code
struct half4
{
    half x, y, z, w;
};

#ifndef DOXYGEN_SKIP
#define FINE_MIP_LEVEL false
#define COARSE_MIP_LEVEL true
#endif

namespace demandLoading {

#if defined( __CUDACC__ ) || defined( OPTIX_PAGING_BIT_OPS )

#ifndef DOXYGEN_SKIP

// clang-format off
__device__ static __forceinline__ void convertColor( const float4& a, float4& b ) { b = a; }
__device__ static __forceinline__ void convertColor( const float4& a, float2& b ) { b = float2{a.x, a.y}; }
__device__ static __forceinline__ void convertColor( const float4& a, float& b ) { b = a.x; }
__device__ static __forceinline__ void convertColor( const float4& a, unsigned int& b ) { b = static_cast<unsigned int>( a.x ); }

__device__ static __forceinline__ void convertColor( const half4& a, float4& b ) { b = float4{a.x, a.y, a.z, a.w}; }
__device__ static __forceinline__ void convertColor( const half4& a, float2& b ) { b = float2{a.x, a.y}; }
__device__ static __forceinline__ void convertColor( const half4& a, float& b ) { b = static_cast<float>( a.x ); }
__device__ static __forceinline__ void convertColor( const half4& a, unsigned int& b ) { b = static_cast<unsigned int>( a.x ); }
// clang-format on

__device__ static __forceinline__ unsigned int isNonPowerOfTwo( unsigned int x )
{
    return ( x & ( x - 1 ) );
}

__device__ static __forceinline__ void fixOddSizeWrapFootprint( uint4& fp )
{
    // The odd size wrap problem occurs because of a shift in texture coordinates done in requestTexFootprint2DGrad.
    // The shift is done for non-power of 2 texture sizes to make sure all texture tiles needed by tex2DGrad are
    // requested. However, in rare cases, the shift will drop from the footprint one of the texture tiles needed by
    // tex2DGrad.  This only occurs in the case where 3 tiles are requested by the footprint instruction.  To fix
    // the problem, we duplicate the tile requests on the right side of the footprint to the left side, and vice-versa.
    // Similarly for top and bottom.

    // fix the wrapping footprint by making sure to always duplicate requested tiles on
    // both sides of the border (top-bottom or left-right)
    if( fp.x & ( fp.y >> 24 ) )
    {
        // OR top byte in mask to bottom byte, and vice-versa.
        fp.x = fp.x | ( fp.y >> 24 );
        fp.y = fp.y | ( fp.x << 24 );
    }
    else
    {
        // OR low order bits in each byte to high order bits, and vice-versa.
        fp.x = fp.x | ( ( fp.x & 0x01010101 ) << 7 ) | ( ( fp.x & 0x80808080 ) >> 7 );
        fp.y = fp.y | ( ( fp.y & 0x01010101 ) << 7 ) | ( ( fp.y & 0x80808080 ) >> 7 );
    }
}

__device__ static __forceinline__ bool requestTexFootprint2D( unsigned int*         referenceBits,
                                                              unsigned int*         residenceBits,
                                                              const TextureSampler& sampler,
                                                              unsigned int          fx,
                                                              unsigned int          fy,
                                                              unsigned int          fz,
                                                              unsigned int          fw )
{
    bool isResident = true;

    // Reconstitute the footprint
    uint4               result    = make_uint4( fx, fy, fz, fw );
    Texture2DFootprint* footprint = reinterpret_cast<Texture2DFootprint*>( &result );

    // Handle mip tail explicitly
    unsigned int mipLevel = footprint->level;
    if( mipLevel >= sampler.mipTailFirstLevel )
    {
        pagingRequest( referenceBits, sampler.startPage );
        isResident = checkBitSet( sampler.startPage, residenceBits ) && isResident;
        return isResident;
    }

    // Load MipLevelSizes as 64-bit int.
    TextureSampler::MipLevelSizes sizes = sampler.mipLevelSizes[mipLevel];

    unsigned int levelWidthInTiles   = sizes.levelWidthInTiles;
    unsigned int levelWidthInBlocks  = ( levelWidthInTiles + 7 ) >> 3;
    unsigned int levelHeightInTiles  = sizes.levelHeightInTiles;
    unsigned int levelHeightInBlocks = ( levelHeightInTiles + 7 ) >> 3;

    // Wrap tileX and tileY
    if( footprint->tileX * 8 >= levelWidthInTiles )
        footprint->tileX -= levelWidthInBlocks;
    if( footprint->tileY * 8 >= levelHeightInTiles )
        footprint->tileY -= levelHeightInBlocks;

    // Common, fast case, in which dx and dy are 0, and levelWidthInTiles and levelHeightInTiles are
    // multiples of 8.  Here, the mask lines up with the page table.
    const int fastCase = !( ( result.z & 0x3f0000 ) | ( ( levelWidthInTiles | levelHeightInTiles ) & 0x7 ) );
    if( fastCase )
    {
        unsigned int mipLevelWordStart = ( sampler.startPage + sizes.mipLevelStart ) / 32;
        unsigned int blockWordIndex = mipLevelWordStart + 2 * ( levelWidthInBlocks * footprint->tileY + footprint->tileX );
        pagingRequestWord( referenceBits + blockWordIndex, result.x );
        pagingRequestWord( referenceBits + blockWordIndex + 1, result.y );

        isResident = ( ( residenceBits[blockWordIndex] & result.x ) == result.x ) && isResident;
        isResident = ( ( residenceBits[blockWordIndex + 1] & result.y ) == result.y ) && isResident;
        return isResident;
    }

    // Uncommon case, in which dx or dy is non-zero (or level size in tiles is not evenly divisible by 8).
    // Here, the mask is toroidally rotated to span multiple 8x8 blocks.
    // Add each set bit to the page table separately.

    // The 8x8 bitmask is separated into result.x and result.y.
    // Process them separately to be faster.
    unsigned int mask   = ( result.x ) ? result.x : result.y;
    unsigned int offset = ( result.x ) ? 0 : 4;

    while( mask )
    {
        unsigned int idx = 31 - __clz( mask );

        int x = wrapFootprintTileCoord( idx % 8, footprint->dx, footprint->tileX, levelWidthInTiles );
        int y = wrapFootprintTileCoord( offset + idx / 8, footprint->dy, footprint->tileY, levelHeightInTiles );

        unsigned int pageOffset = getPageOffsetFromTileCoords( x, y, levelWidthInTiles );
        unsigned int pageId     = sampler.startPage + sizes.mipLevelStart + pageOffset;

        pagingRequest( referenceBits, pageId );
        isResident = checkBitSet( pageId, residenceBits ) && isResident;

        mask ^= ( 1 << idx );
        if( ( mask | offset ) == 0 )
        {
            mask   = result.y;
            offset = 4;
        }
    }

    return isResident;
}


/// Compute mip level from the texture gradients.
__device__ __forceinline__ float getMipLevel( float2 ddx, float2 ddy, int texWidth, int texHeight, float invAnisotropy )
{
    ddx = float2{ddx.x * texWidth, ddx.y * texHeight};
    ddy = float2{ddy.x * texWidth, ddy.y * texHeight};

    // Trying to follow CUDA. CUDA performs a low precision EWA filter
    // correction on the texture gradients to determine the mip level.
    // This calculation is described in the Siggraph 1999 paper:
    // Feline: Fast Elliptical Lines for Anisotropic Texture Mapping

    const float A    = ddy.x * ddy.x + ddy.y * ddy.y;
    const float B    = -2.0f * ( ddx.x * ddy.x + ddx.y * ddy.y );
    const float C    = ddx.x * ddx.x + ddx.y * ddx.y;
    const float root = sqrtf( fmaxf( A * A - 2.0f * A * C + C * C + B * B, 0.0f ) );

    // Compute the square of the major and minor ellipse radius lengths to avoid sqrts.
    // Then compensate by taking half the log to get the mip level.

    const float minorRadius2 = ( A + C - root ) * 0.5f;
    const float majorRadius2 = ( A + C + root ) * 0.5f;
    const float filterWidth2 = fmaxf( minorRadius2, majorRadius2 * invAnisotropy * invAnisotropy );
    const float mipLevel     = 0.5f * log2f( filterWidth2 );
    return mipLevel;
}

__device__ static __forceinline__ bool requestTexFootprint2DGrad( const TextureSampler& sampler,
                                                                  unsigned int*         referenceBits,
                                                                  unsigned int*         residenceBits,
                                                                  float                 x,
                                                                  float                 y,
                                                                  float                 dPdx_x,
                                                                  float                 dPdx_y,
                                                                  float                 dPdy_x,
                                                                  float                 dPdy_y )
{
    const CUaddress_mode wrapMode0 = static_cast<CUaddress_mode>( sampler.desc.wrapMode0 );
    const CUaddress_mode wrapMode1 = static_cast<CUaddress_mode>( sampler.desc.wrapMode1 );

    x = wrapTexCoord( x, wrapMode0 );
    y = wrapTexCoord( y, wrapMode1 );

    // Fix wrapping problem in the footprint instruction for non-power of 2 textures.
    // (If the tex coord is close to the right edge of the texture, move it to the left edge.)
    bool oddSizeWrap = false;
    if( wrapMode0 == CU_TR_ADDRESS_MODE_WRAP && isNonPowerOfTwo( sampler.width ) && x + fmax( fabs( dPdx_x ), fabs( dPdy_x ) ) >= 1.0f )
    {
        x           = 0.0f;
        oddSizeWrap = true;
    }
    if( wrapMode1 == CU_TR_ADDRESS_MODE_WRAP && isNonPowerOfTwo( sampler.height ) && y + fmax( fabs( dPdx_y ), fabs( dPdy_y ) ) >= 1.0f )
    {
        y           = 0.0f;
        oddSizeWrap = true;
    }

    unsigned int singleMipLevel;

    unsigned int desc   = *reinterpret_cast<const unsigned int*>( &sampler.desc );
    uint4        finefp = optixTexFootprint2DGrad( sampler.texture, desc, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y, FINE_MIP_LEVEL, &singleMipLevel );
    if( oddSizeWrap )
        fixOddSizeWrapFootprint( finefp );
    bool isResident = requestTexFootprint2D( referenceBits, residenceBits, sampler, finefp.x, finefp.y, finefp.z, finefp.w );

    uint4 coarsefp = uint4{0, 0, 0, 0};
    if( !singleMipLevel )
    {
        coarsefp = optixTexFootprint2DGrad( sampler.texture, desc, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y,
                                            COARSE_MIP_LEVEL, &singleMipLevel );
        if( oddSizeWrap )
            fixOddSizeWrapFootprint( coarsefp );
        isResident = requestTexFootprint2D( referenceBits, residenceBits, sampler, coarsefp.x, coarsefp.y, coarsefp.z, coarsefp.w ) && isResident;
    }

    // Handle discrepancy between mip levels in SW and HW footprint implementations.
    // If the texture instruction thinks the texture is not resident (texResident), but the footprint
    // routine thinks it is (footprintResident), scale the gradients and request again.
    const Texture2DFootprint* fineFootprint = reinterpret_cast<Texture2DFootprint*>( &finefp );
    const unsigned int        swFootprint   = fineFootprint->reserved1;
    if( swFootprint && isResident )
    {
        float mipLevel = getMipLevel( make_float2( dPdx_x, dPdx_y ), make_float2( dPdy_x, dPdy_y ), sampler.width,
                                      sampler.height, 1.0f / sampler.desc.maxAnisotropy );
        float fracLevel = mipLevel - floorf( mipLevel );

        const float MAX_SW_MIPLEVEL_ERROR = 0.18f;

        if( fracLevel < MAX_SW_MIPLEVEL_ERROR || fracLevel > (1.0f - MAX_SW_MIPLEVEL_ERROR) )
        {
            // scale to just over the integer boundary
            float        gradScale = ( fracLevel < 0.5f ) ? 0.99f * exp2f( -fracLevel ) : 1.01f * exp2f( 1.0f - fracLevel );
            unsigned int coarseLevel = ( fracLevel < 0.5f ) ? FINE_MIP_LEVEL : COARSE_MIP_LEVEL;

            uint4 fp = optixTexFootprint2DGrad( sampler.texture, desc, x, y, dPdx_x * gradScale, dPdx_y * gradScale,
                                                dPdy_x * gradScale, dPdy_y * gradScale, coarseLevel, &singleMipLevel );
            isResident = requestTexFootprint2D( referenceBits, residenceBits, sampler, fp.x, fp.y, fp.z, fp.w ) && isResident;

            // Handle case of landing exactly on mip level
            if( fracLevel == 0.0f )
            {
                gradScale   = 1.01f;
                coarseLevel = COARSE_MIP_LEVEL;

                fp = optixTexFootprint2DGrad( sampler.texture, desc, x, y, dPdx_x * gradScale, dPdx_y * gradScale,
                                            dPdy_x * gradScale, dPdy_y * gradScale, coarseLevel, &singleMipLevel );
                isResident = requestTexFootprint2D( referenceBits, residenceBits, sampler, fp.x, fp.y, fp.z, fp.w ) && isResident;
            }
        }
    }

    return isResident;
}


__device__ static __forceinline__ bool requestTexFootprint2DLod( const TextureSampler& sampler,
                                                                 unsigned int*         referenceBits,
                                                                 unsigned int*         residenceBits,
                                                                 float                 x,
                                                                 float                 y,
                                                                 float                 lod )
{
    const CUaddress_mode wrapMode0 = static_cast<CUaddress_mode>( sampler.desc.wrapMode0 );
    const CUaddress_mode wrapMode1 = static_cast<CUaddress_mode>( sampler.desc.wrapMode1 );

    x = wrapTexCoord( x, wrapMode0 );
    y = wrapTexCoord( y, wrapMode1 );

    // Fix wrapping problem in the footprint instruction for non-power of 2 textures
    // (If the tex coord is close to the right edge of the texture, move it to the left edge.)
    const float expMipLevel = exp2( lod );
    if( wrapMode0 == CU_TR_ADDRESS_MODE_WRAP && x + expMipLevel / sampler.width >= 1.0f )
        x = 0.0f;
    if( wrapMode1 == CU_TR_ADDRESS_MODE_WRAP && y + expMipLevel / sampler.height >= 1.0f )
        y = 0.0f;

    unsigned int singleMipLevel;
    unsigned int desc = *reinterpret_cast<const unsigned int*>( &sampler.desc );

    uint4 fp = optixTexFootprint2DLod( sampler.texture, desc, x, y, lod, FINE_MIP_LEVEL, &singleMipLevel );
    bool isResident = requestTexFootprint2D( referenceBits, residenceBits, sampler, fp.x, fp.y, fp.z, fp.w );

    if( !singleMipLevel )
    {
        fp = optixTexFootprint2DLod( sampler.texture, desc, x, y, lod, COARSE_MIP_LEVEL, &singleMipLevel );
        isResident = requestTexFootprint2D( referenceBits, residenceBits, sampler, fp.x, fp.y, fp.z, fp.w ) && isResident;
    }

    return isResident;
}


template <class TYPE> 
__device__ static __forceinline__ bool
getBaseColor( const DeviceContext& context, unsigned int textureId, TYPE& rval, bool* baseColorResident )
{
    const unsigned int maxTextures = context.pageTable.capacity >> 1;
    const unsigned long long baseVal = pagingMapOrRequest( context, textureId + maxTextures, baseColorResident );
    const half4* baseColor = reinterpret_cast<const half4*>( &baseVal );
    if( *baseColorResident && !__hisnan( baseColor->x ) ) // NaN indicates nonexistent base color
    {
        convertColor( *baseColor, rval );
        return true;
    }
    return false;
}

#endif  // ndef DOXYGEN_SKIP

/// Fetch from a demand-loaded texture with the specified identifer, obtained via DemandLoader::createTexture.
/// The given DeviceContext is typically a launch parameter, obtained via DemandLoader::launchPrepare,
/// that has been copied to device memory.
template <class TYPE>
__device__ static __forceinline__ TYPE
tex2DGrad( const DeviceContext& context, unsigned int textureId, float x, float y, float2 ddx, float2 ddy, bool* isResident )
{
    // Check for base color
    TYPE rval;
    bool baseColorResident;
    convertColor( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );

    const float minGradSquared = minf( ddx.x * ddx.x + ddx.y * ddx.y, ddy.x * ddy.x + ddy.y * ddy.y );
    if( minGradSquared >= 1.0f )
    {
        *isResident = getBaseColor<TYPE>( context, textureId, rval, &baseColorResident );
        if( *isResident || !baseColorResident )
            return rval;
    }

    // Check whether the texture sampler is resident.  The samplers occupy the first N entries of the page table.
    const TextureSampler* sampler =
        reinterpret_cast<const TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    if( !sampler )
    {
        if( *isResident )
            *isResident = getBaseColor<TYPE>( context, textureId, rval, &baseColorResident );
        return rval;
    }

    // Prevent footprint from exceeding min tile width for non-mipmapped textures
    if( sampler->desc.numMipLevels == 1 )
    {
        float       pixelSpanX       = max( fabsf( ddx.x ), fabsf( ddy.x ) ) * sampler->width;
        float       pixelSpanY       = max( fabsf( ddx.y ), fabsf( ddy.y ) ) * sampler->height;
        float       pixelSpan        = max( pixelSpanX, pixelSpanY );
        const float halfMinTileWidth = 32.0f;  // half min tile width for sparse textures

        if( pixelSpan > halfMinTileWidth )
        {
            float scale = halfMinTileWidth / pixelSpan;
            ddx         = make_float2( ddx.x * scale, ddx.y * scale );
            ddy         = make_float2( ddy.x * scale, ddy.y * scale );
        }
    }
    
    // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
    *isResident = !sampler->desc.isSparseTexture;
    if( context.requestIfResident == false )
        rval = tex2DGrad<TYPE>( sampler->texture, x, y, ddx, ddy, isResident );

    // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
    if( *isResident == false && sampler->desc.isSparseTexture)
        *isResident = requestTexFootprint2DGrad( *sampler, context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y );

    // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
    if( *isResident && context.requestIfResident )
        rval = tex2DGrad<TYPE>( sampler->texture, x, y, ddx, ddy ); // non-pedicated texture fetch

    // Debug Code: This checks consistency between residency result from requestTexFootprint2DGrad and predicated tex2DGrad
    /*
    if( *isResident && context.requestIfResident )
    {
        bool texResident;
        rval = tex2DGrad<TYPE>( sampler->texture, x, y, ddx, ddy, &texResident );
        if( !texResident )
            printf("ERROR: Mipmap mismatch between tex2DGrad and requestTexFootprint2DGrad!\n");
    }
    */

    return rval;
}


/// Fetch from a demand-loaded texture with the specified identifer, obtained via DemandLoader::createTexture.
/// The given DeviceContext is typically a launch parameter, obtained via DemandLoader::launchPrepare,
/// that has been copied to device memory.
template <class TYPE>
__device__ static __forceinline__ TYPE
tex2DLod( const DeviceContext& context, unsigned int textureId, float x, float y, float lod, bool* isResident )
{
    // Check whether the texture sampler is resident.  The samplers occupy the first N entries of the page table.
    const TextureSampler* sampler =
        reinterpret_cast<const TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    
    TYPE rval;
    convertColor( float4{1.0f, 0.0f, 1.0f, 0.0f}, rval );
    if( *isResident == false )
        return rval;

    // Prevent footprint from exceeding min tile width for non-mipmapped textures
    if( sampler && sampler->desc.numMipLevels == 1 )
        lod = 0.0f;

    // Check for base color.
    // Note: It would be preferable to check for baseColor before the sampler is loaded, but 
    // texture width and height are needed to determine if we are in the base color case from lod.
    if( !sampler || exp2f( lod ) >= max( sampler->width, sampler->height ) )
    {
        bool baseColorResident;
        if( getBaseColor<TYPE>( context, textureId, rval, &baseColorResident ) )
            return rval;
        *isResident = false;
        return rval;
    }

    // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
    *isResident = false;
    if( context.requestIfResident == false )
        rval = tex2DLod<TYPE>( sampler->texture, x, y, lod, isResident );

    // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
    if( *isResident == false && sampler->desc.isSparseTexture )
        *isResident = requestTexFootprint2DLod( *sampler, context.referenceBits, context.residenceBits, x, y, lod );

    // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
    if( *isResident && context.requestIfResident )
        rval = tex2DLod<TYPE>( sampler->texture, x, y, lod ); // non-pedicated texture fetch

    return rval;
}


/// Fetch from a demand-loaded texture with the specified identifer, obtained via DemandLoader::createTexture.
/// The given DeviceContext is typically a launch parameter, obtained via DemandLoader::launchPrepare,
/// that has been copied to device memory.
template <class TYPE>
__device__ static __forceinline__ TYPE
tex2D( const DeviceContext& context, unsigned int textureId, float x, float y, bool* isResident )
{
    return tex2DLod<TYPE>( context, textureId, x, y, 0.0f, isResident );
}


#endif

}  // namespace demandLoading
