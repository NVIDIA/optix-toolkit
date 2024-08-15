// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cfloat>

namespace rasterize
{
    __device__ float cross( const float2& a, const float2& b )
    {
        return a.x * b.y - a.y * b.x;
    }

    __device__ float2 perp( const float2& a )
    {
        return { a.y, -a.x };
    }

    template <typename T>
    __device__ void swap( T& a, T& b )
    {
        T tmp = a;
        a = b;
        b = tmp;
    }

    template <typename T>
    __device__ T min( const T& a, const T& b )
    {
        return ( a > b ) ? b : a;
    }

    template <typename T>
    __device__ T max( const T& a, const T& b )
    {
        return ( a < b ) ? b : a;
    }

    // conservative rasterize, taking filter kernel with into acount.
    // the triangle is rasterized in up to N non-overlapping integer AABB in texel space.
    // the function returns the sum of all AABB evaluations.
    template <typename T, typename U>
    __device__ T rasterize( U            eval,  // evaluation function
        float2       v0,
        float2       v1,
        float2       v2,
        float        w,   // filter kernel width in pixels
        unsigned int N )  // resolution
    {
        T states = {};

        // sort vertices top to bottom
        if( v0.y > v1.y )
            swap( v0, v1 );
        if( v1.y > v2.y )
            swap( v1, v2 );
        if( v0.y > v1.y )
            swap( v0, v1 );

        // is the middle vertex to the left or right side?
        const bool  left_side = cross( v1 - v0, v2 - v0 ) < 0;
        const float wx_side = left_side ? ( -w ) : w;

        // lower edge
        const float  wy_low = ( v1.x < v0.x ) ? wx_side : ( -wx_side );
        const float2 e0a = { v0.x + wx_side, v0.y + wy_low };
        const float2 e0b = { v1.x + wx_side, v1.y + wy_low };

        // higher edge
        const float  wy_high = ( v2.x < v1.x ) ? wx_side : ( -wx_side );
        const float2 e1a = { v1.x + wx_side, v1.y + wy_high };
        const float2 e1b = { v2.x + wx_side, v2.y + wy_high };

        // long edge
        const float  wy_long = ( v2.x > v0.x ) ? wx_side : ( -wx_side );
        const float2 e2a = { v0.x - wx_side, v0.y + wy_long };
        const float2 e2b = { v2.x - wx_side, v2.y + wy_long };

        // edge equations
        const float2 n0 = perp( e0a - e0b );
        const float  c0 = dot( n0, e0a );
        const float2 n1 = perp( e1a - e1b );
        const float  c1 = dot( n1, e1a );
        const float2 n2 = perp( e2a - e2b );
        const float  c2 = dot( n2, e2a );

        // deal with degenerate short edges
        const float inv0 = ( n0.x != 0 ) ? ( 1.f / n0.x ) : ( ( left_side == ( n0.y > 0.f ) ) ? FLT_MAX : -FLT_MAX );
        const float inv1 = ( n1.x != 0 ) ? ( 1.f / n1.x ) : ( ( left_side == ( n1.y > 0.f ) ) ? -FLT_MAX : FLT_MAX );
        const float inv2 = 1.f / n2.x;  // degenerate long edge doesn't matter

        const float by = floorf( v0.y - w );
        const int   dy = ( int )ceilf( ( v2.y + w ) - by );

        // clamp iteration count.
        // there's no point in evaluate multiple consequitive 'stripes' within a single texel scanline.
        N = min( N, ( unsigned int )dy );

        // start at top
        float2 prev_interval = { v0.x - w, v0.x + w };
        float  prev_y = by;

        const float ddy = ( float )dy / ( float )N;  // Round up to next ulp? Not needed if N is power of two.

        for( unsigned int i = 1; i <= N; ++i )
        {
            // make sure next_y - prev_y >= 1
            const float next_y = fmaxf( prev_y + 1, by + ceilf( ddy * ( float )i ) );

            float2 next_interval;
            if( next_y <= v0.y )
            {
                // first interval
                next_interval = { v0.x - w, v0.x + w };
            }
            else if( next_y < v2.y )
            {
                // solve edge equations for x
                const float x0 = inv0 * ( c0 - next_y * n0.y );
                const float x1 = inv1 * ( c1 - next_y * n1.y );
                const float x2 = inv2 * ( c2 - next_y * n2.y );
                // construct interval for bottom

                if( left_side )
                    next_interval = { fmaxf( x0, x1 ), x2 };
                else
                    next_interval = { x2, fminf( x0, x1 ) };
            }
            else
            {
                // last interval
                next_interval = { v2.x - w, v2.x + w };
            }

            // eval
            {
                float2 interval = next_interval;

                // include middle vertex
                if( v1.y + w >= prev_y && v1.y - w < next_y )
                {
                    interval = { fminf( interval.x, v1.x - w ), fmaxf( interval.y, v1.x + w ) };
                }

                int2 lo = {
                    ( int )floorf( fminf( interval.x, prev_interval.x ) ),
                    ( int )prev_y  // already rounded
                };

                int2 hi = {
                    max( lo.x, ( int )ceilf( fmaxf( interval.y, prev_interval.y ) ) - 1 ),
                    ( int )next_y - 1  // already rounded
                };

                states += eval( lo, hi );
            }

            // advance scanline
            prev_interval = next_interval;
            prev_y = next_y;
        }

        return states;
    }

}; // namespace rasterize