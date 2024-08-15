// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>
#include "Util/VecMath.h"
#include "Util/Rasterize.h"
#include "CuOmmBakingImpl.h"

// Struct to track opacity states in an area.
struct OpacityStateSet
{
    __device__ OpacityStateSet() {}

    __device__ OpacityStateSet( uint32_t transparent, uint32_t opaque, uint32_t unknown )
    {
        h.x = (transparent != 0);
        h.y = (opaque != 0 );
        h.z = (unknown != 0 );
    }

    __device__ OpacityStateSet& operator+=( OpacityStateSet state )
    {
        *this = *this + state;
        return *this;
    }

    __device__ OpacityStateSet operator+( OpacityStateSet state ) const
    {
        // we don't actually care about the count, just if it's zero or not.
        // by using logic or we don't have to deal with overflows.
        return OpacityStateSet( h.x | state.h.x, h.y | state.h.y, h.z | state.h.z );
    }

    // return true if the set is a mixture of multiple states
    __device__ bool isMixed() const
    {
        if( ( ( h.x != 0 ) + ( h.y != 0 ) + ( h.z != 0 ) ) > 1 )
            return true;
        return false;
    }

    // return true if the set has only transparent states
    __device__ bool isTransparent() const
    {
        if( ( h.x != 0 ) && ( h.y == 0 ) && ( h.z == 0 ) )
            return true;
        return false;
    }

    // return true if the set has only opaque states
    __device__ bool isOpaque() const
    {
        if( ( h.x == 0 ) && ( h.y != 0 ) && ( h.z == 0 ) )
            return true;
        return false;
    }

    // return true if the set has transparent states
    __device__ bool hasTransparent() const
    {
        if( h.x != 0 )
            return true;
        return false;
    }

    // return true if the set is single state
    __device__ bool isUniform() const
    {
        if( ( ( h.x != 0 ) + ( h.y != 0 ) + ( h.z != 0 ) ) == 1 )
            return true;
        return false;
    }

    // TODO: as we only care about true/false, we could pack this. try to see if it's faster or not.
    uint3 h = {};
};

__device__ OpacityStateSet evalSumTableTile( const TextureData& texture )
{
    const int width = texture.width;
    const int height = texture.height;
    const int rowPitch = texture.getRowPitch();
    const int colPitch = texture.getColPitch();
    uint2     v = texture.sumTable[( height - 1 ) * rowPitch + ( width - 1 ) * colPitch];
    uint32_t  unknowns = 2 * width * height - v.x - v.y;

    return OpacityStateSet( v.x, v.y, unknowns );
}


// evaluate a range within one tile in the sum table
// pre: the aabb should be within the range [0,width)x[0,height)
__device__ OpacityStateSet evalSumTableTile( const TextureData& texture, int2 lo, int2 hi, int2 tile )
{
    int width = texture.width;
    int height = texture.height;

    if( ( tile.x & 1 ) == 1 && texture.addressMode[0] != cudaAddressModeWrap )
    {
        int tmp = lo.x;
        lo.x = width - 1 - hi.x;
        hi.x = width - 1 - tmp;
    }

    if( ( tile.y & 1 ) == 1 && texture.addressMode[1] != cudaAddressModeWrap )
    {
        int tmp = lo.y;
        lo.y = height - 1 - hi.y;
        hi.y = height - 1 - tmp;
    }

    // evaluate clipped sum table square
    const int rowPitch = texture.getRowPitch();
    const int colPitch = texture.getColPitch();
    uint2     v11 = texture.sumTable[( hi.y ) * rowPitch + ( hi.x ) * colPitch];
    uint2     v10 = ( lo.y > 0 ) ? texture.sumTable[( lo.y - 1 ) * rowPitch + ( hi.x ) * colPitch] : make_uint2( 0, 0 );
    uint2     v01 = ( lo.x > 0 ) ? texture.sumTable[( hi.y ) * rowPitch + ( lo.x - 1 ) * colPitch] : make_uint2( 0, 0 );
    uint2     v00 = ( lo.x > 0 && lo.y > 0 ) ? texture.sumTable[( lo.y - 1 ) * rowPitch + ( lo.x - 1 ) * colPitch] : make_uint2( 0, 0 );
    uint2     v = v11 - v10 - v01 + v00;

    uint32_t unknowns = 2 * ( hi.x - lo.x + 1 ) * ( hi.y - lo.y + 1 ) - v.x - v.y;
    return OpacityStateSet( v.x, v.y, unknowns );
}


// pre: the aabb should not be more 2^15 texels in either dimension to prevent overflows
__device__ OpacityStateSet evalSumTable( const TextureData& texture, int2 inLo, int2 inHi )
{
    int2 lo = inLo, hi = inHi;

    int width = texture.width;
    int height = texture.height;

    if( texture.addressMode[0] == cudaAddressModeClamp || texture.addressMode[0] == cudaAddressModeBorder )
    {
        lo.x = max( 0, lo.x );
        hi.x = max( lo.x, hi.x );

        hi.x = min( width, hi.x );
        lo.x = min( lo.x, hi.x );
    }

    if( texture.addressMode[1] == cudaAddressModeClamp || texture.addressMode[1] == cudaAddressModeBorder )
    {
        lo.y = max( 0, lo.y );
        hi.y = max( lo.y, hi.y );

        hi.y = min( height, hi.y );
        lo.y = min( lo.y, hi.y );
    }

    int2 extent = { hi.x - lo.x, hi.y - lo.y };

    // convert to tile space
    int2 lo_tile = { lo.x / width, lo.y / height };

    lo.x -= lo_tile.x * width;
    lo.y -= lo_tile.y * height;

    // for negative lo the tile index is rounded up (to zero). make sure lo is inside the lo tile.
    if( lo.x < 0 )
        lo_tile.x--, lo.x += width;
    if( lo.y < 0 )
        lo_tile.y--, lo.y += height;

    // clamp to two tiles. accounting for repeated tiles won't change the final visibility state.
    hi.x = min( lo.x + extent.x, 2 * width - 1 );
    hi.y = min( lo.y + extent.y, 2 * height - 1 );

    // upper-left corner
    OpacityStateSet states = evalSumTableTile( texture, lo, make_int2( min( hi.x, width - 1 ), min( hi.y, height - 1 ) ), lo_tile );

    if( hi.x >= width )
    {
        // upper-right corner
        states += evalSumTableTile( texture, { 0, lo.y }, make_int2( hi.x - width, min( hi.y, height - 1 ) ), { lo_tile.x + 1, lo_tile.y } );
    }

    if( hi.y >= height )
    {
        // lower-left corner
        states += evalSumTableTile( texture, { lo.x, 0 }, make_int2( min( hi.x, width - 1 ), hi.y - height ), { lo_tile.x, lo_tile.y + 1 } );

        if( hi.x >= width )
        {
            // lower-right corner
            states += evalSumTableTile( texture, { 0, 0 }, make_int2( hi.x - width, hi.y - height ), { lo_tile.x + 1, lo_tile.y + 1 } );
        }
    }

    return states;
}

__device__ OpacityStateSet sampleMemoryTexture( const TextureData& texture, float2 uv0, float2 uv1, float2 uv2, float filterKernelRadiusInTexels, unsigned int resolution = 1 )
{
    auto eval = [&]( int2 lo, int2 hi ) -> OpacityStateSet { return evalSumTable( texture, lo, hi ); };

    // compute conservative AABB in texel space
    float2 flo = { fminf( fminf( uv0.x, uv1.x ), uv2.x ) - filterKernelRadiusInTexels, fminf( fminf( uv0.y, uv1.y ), uv2.y ) - filterKernelRadiusInTexels };
    float2 fhi = { fmaxf( fmaxf( uv0.x, uv1.x ), uv2.x ) + filterKernelRadiusInTexels, fmaxf( fmaxf( uv0.y, uv1.y ), uv2.y ) + filterKernelRadiusInTexels };

    // we do all math in floats to prevent integer overlows

    float fwidth = ( float )texture.width;
    float fheight = ( float )texture.height;

    if( texture.addressMode[0] == cudaAddressModeClamp || texture.addressMode[0] == cudaAddressModeBorder )
    {
        flo.x = fmaxf( 0.f, flo.x );
        fhi.x = fmaxf( flo.x, fhi.x );

        fhi.x = fminf( fwidth, fhi.x );
        flo.x = fminf( flo.x, fhi.x );
    }

    if( texture.addressMode[1] == cudaAddressModeClamp || texture.addressMode[1] == cudaAddressModeBorder )
    {
        flo.y = fmaxf( 0.f, flo.y );
        fhi.y = fmaxf( flo.y, fhi.y );

        fhi.y = fminf( fheight, fhi.y );
        flo.y = fminf( flo.y, fhi.y );
    }

    float2 fextent = { ( fhi.x - flo.x ), ( fhi.y - flo.y ) };

    // convert to tile space
    float2 flo_tile = { flo.x / fwidth, flo.y / fheight };

    // use modff to guarentee that flo ends up in [0,1)x[0,1)
    float2 ilo_tile;
    flo.x = modff( flo_tile.x, &ilo_tile.x );
    flo.y = modff( flo_tile.y, &ilo_tile.y );

    // float to int conversion overflow may result in an undefined tile index, but we don't care
    int2 lo_tile = { ( int )ilo_tile.x, ( int )ilo_tile.y };
    if( flo.x < 0 )
        flo.x += 1.f, lo_tile.x++;
    if( flo.y < 0 )
        flo.y += 1.f, lo_tile.y++;

    // convert to texel space
    flo.x *= fwidth;
    flo.y *= fheight;

    fhi.x = flo.x + fextent.x;
    fhi.y = flo.y + fextent.y;

    bool doRasterize = ( resolution > 1 );

    if( fhi.x > 2 * fwidth || fhi.y > 2 * fheight )
    {
        // clamp to at most two tiles. including multiple full tiles won't change the visibility state.
        fhi.x = fminf( fhi.x, 2 * fwidth );
        fhi.y = fminf( fhi.y, 2 * fheight );

        // don't rasterize large triangles to prevent numerical issues with large uv coordinates
        // we could be less aggressive here, but triangles covering the texture space multiple times are unlikely to benefit from rasterization anyway.
        doRasterize = false;
    }

    // convert to conservative integer texel AABB. The range should be within [0,2*texture.width]x[0,2*texture.height] so we can ignore overflows.
    int width = texture.width;
    int height = texture.height;

    int2 lo, hi;
    lo.x = ( int )floorf( flo.x );
    lo.y = ( int )floorf( flo.y );
    hi.x = max( lo.x, ( int )( ceilf( fhi.x ) - 1.f ) );
    hi.y = max( lo.y, ( int )( ceilf( fhi.y ) - 1.f ) );

    // upper-left corner
    OpacityStateSet states = evalSumTableTile( texture, lo, make_int2( min( hi.x, width - 1 ), min( hi.y, height - 1 ) ), lo_tile );

    if( hi.x >= width )
    {
        // upper-right corner
        states += evalSumTableTile( texture, { 0, lo.y }, make_int2( hi.x - width, min( hi.y, height - 1 ) ), { lo_tile.x + 1, lo_tile.y } );
    }

    if( hi.y >= height )
    {
        // lower-left corner
        states += evalSumTableTile( texture, { lo.x, 0 }, make_int2( min( hi.x, width - 1 ), hi.y - height ), { lo_tile.x, lo_tile.y + 1 } );

        if( hi.x >= width )
        {
            // lower-right corner
            states += evalSumTableTile( texture, { 0, 0 }, make_int2( hi.x - width, hi.y - height ), { lo_tile.x + 1, lo_tile.y + 1 } );
        }
    }

    if( doRasterize && states.isMixed() )
    {
        // the AABB has varying states. perform a high resolution evaluation using rasterization
        states = rasterize::rasterize<OpacityStateSet>( eval,  // evaluation function
            uv0, uv1, uv2,
            filterKernelRadiusInTexels,  // filter kernel radius in pixels
            resolution );
    }

    return states;
}
