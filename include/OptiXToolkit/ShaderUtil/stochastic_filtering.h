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

/// \file stochastic_filtering.h
/// Header for doing stochastic texture filtering. See
/// https://github.com/NVIDIA/otk-shader-util/blob/master/docs/StochasticTextureFiltering.pdf

#include <OptiXToolkit/ShaderUtil/vec_math.h>

OTK_DEVICE const float MAX_HW_ANISOTROPY = 16.0f;
OTK_DEVICE const float GAUSSIAN_STANDARD_WIDTH = 0.40824829f; // scale Gaussian to have same variance as tent

template <class TYPE>  OTK_INLINE OTK_HOSTDEVICE void swap(TYPE& a, TYPE& b) { TYPE tmp=a; a=b; b=tmp; }

/// Transform a point in [0,1)^2 to a box filter distribution in [-0.5, 0.5)^2
OTK_INLINE OTK_HOSTDEVICE float2 boxFilter( float2 xi )
{
    return xi - float2{0.5f, 0.5f};
}

/// Transform a point in [0,1) to a tent filter distribution in [-1,1)
OTK_INLINE OTK_HOSTDEVICE float tentFilter( float x )
{
    return ( x < 0.5f ) ? -1.0f + sqrtf( x * 2.0f ) : 1.0f - sqrtf( 2.0f - x * 2.0f );
}

/// Transform a point in [0,1)^2 to a tent filter distribution in [-1,1)^2
OTK_INLINE OTK_HOSTDEVICE float2 tentFilter( float2 xi )
{
    return float2{tentFilter( xi.x ), tentFilter( xi.y )};
}

/// Transform a point in [0,1) to a point on the unit circle
OTK_INLINE OTK_HOSTDEVICE float2 sampleCircle( float x )
{
    const float theta = 2.0f * M_PIf * x;
    return float2{cosf( theta ), sinf( theta )};
}

/// Transform a point in [0,1)^2 to a 2D gaussian with std deviation 1
OTK_INLINE OTK_HOSTDEVICE float2 boxMuller( float2 xi )
{
    float r = sqrtf( -2.0f * logf( xi.x ) );
    return r * sampleCircle( xi.y );
}

/// Return jitter for a stochastic EWA filter with texture gradients (ddx, ddy), and random value xi in [0,1)^2.
OTK_INLINE OTK_HOSTDEVICE float2 jitterEWA( float2 ddx, float2 ddy, float2 xi )
{
    float2 jitter = GAUSSIAN_STANDARD_WIDTH * boxMuller( xi );
    return jitter.x * ddx + jitter.y * ddy;
}

/// Shorten ddx or ddy to fit in MAX_HW_ANISOTROPY, and return jitter to make up for the shortened length
OTK_INLINE OTK_HOSTDEVICE float2 extendAnisotropy( float2& ddx, float2& ddy, float2 xi )
{
    float xlen = otk::length( ddx );
    float ylen = otk::length( ddy );

    if( ylen > xlen * MAX_HW_ANISOTROPY )
    {
        swap<float2>( ddx, ddy );
        swap<float>( xlen, ylen );
    }

    if( xlen > ylen * MAX_HW_ANISOTROPY )
    {
        float2 jitter = GAUSSIAN_STANDARD_WIDTH * boxMuller( xi );  // Gaussian distribution
        float xScale = MAX_HW_ANISOTROPY * ylen / xlen;             // How much to scale ddx
        float xJitterWidth = 1.0f - xScale * xScale;                // How much to jitter on ddx
        float2 finalJitter = ( jitter.x * xJitterWidth ) * ddx;     // Compute jitter vector in texture space
        ddx *= xScale;                                              // scale ddx so anisotropy is MAX_HW_ANISOTROPY
        return finalJitter;
    }

    return float2{0.0f, 0.0f}; // no jitter if within HW anisotropy
}

// Coefficients for stretched cubic approximation to inverse(integral(lanczos))
// Base
#define LANCZOS_PWEIGHT 0.00569f
#define LANCZOS_NWEIGHT 0.16220f
#define LANCZOS_POS float4{ 0.32691f, -0.41896f, 1.08948f, 0.00010f }
#define LANCZOS_NEG float4{ 0.09974f,  0.32391f, 0.52500f, 0.99597f }
#define LANCZOS LANCZOS_POS, LANCZOS_NEG, LANCZOS_PWEIGHT
// Box
#define LANCZOS_BOX_PWEIGHT 0.00569f
#define LANCZOS_BOX_NWEIGHT 0.35800f
#define LANCZOS_BOX_POS float4{  0.20599f, -0.29053f, 0.89795f, 0.00066f }
#define LANCZOS_BOX_NEG float4{ -0.13260f,  0.45519f, 0.44461f, 0.81565f }
#define LANCZOS_BOX LANCZOS_BOX_POS, LANCZOS_BOX_NEG, LANCZOS_BOX_PWEIGHT
// Tent
#define LANCZOS_TENT_PWEIGHT 0.00569f
#define LANCZOS_TENT_NWEIGHT 0.66602f
#define LANCZOS_TENT_POS float4{  0.09793f, -0.18720f, 0.72571f, 0.00088f }
#define LANCZOS_TENT_NEG float4{ -0.14182f,  0.34999f, 0.38825f, 0.63841f }
#define LANCZOS_TENT LANCZOS_TENT_POS, LANCZOS_TENT_NEG, LANCZOS_TENT_PWEIGHT

// Coefficients for stretched cubic approximation to inverse(integral(mitchell))
// Base
#define MITCHELL_PWEIGHT 0.00107f
#define MITCHELL_NWEIGHT 0.06767f
#define MITCHELL_POS float4{  0.52640f, -0.59468f, 1.19929f, -0.00225f }
#define MITCHELL_NEG float4{ -0.27787f,  0.71069f, 0.36349f,  1.14912f }
#define MITCHELL MITCHELL_POS, MITCHELL_NEG, MITCHELL_PWEIGHT
// Box
#define MITCHELL_BOX_PWEIGHT 0.00107f
#define MITCHELL_BOX_NWEIGHT 0.1547f
#define MITCHELL_BOX_POS float4{  0.31533f, -0.377528f, 0.96381f, 0.00066f }
#define MITCHELL_BOX_NEG float4{ -0.28331f,  0.54979f,  0.37258f, 0.91094f }
#define MITCHELL_BOX MITCHELL_BOX_POS, MITCHELL_BOX_NEG, MITCHELL_BOX_PWEIGHT
// Tent
#define MITCHELL_TENT_PWEIGHT 0.00107f
#define MITCHELL_TENT_NWEIGHT 0.38806f
#define MITCHELL_TENT_POS float4{  0.12211f, -0.22668f, 0.79068f, 0.00009f }
#define MITCHELL_TENT_NEG float4{ -0.16519f,  0.30708f, 0.35141f, 0.68957f }
#define MITCHELL_TENT MITCHELL_TENT_POS, MITCHELL_TENT_NEG, MITCHELL_TENT_PWEIGHT

// Coefficients for stretched cubic approximation to inverse(integral(cyindricalLanczos))
// Base
#define CLANCZOS_WEIGHT 0.10993f
#define CLANCZOS_POS float4{ -0.36293f, 0.81733f, 0.73176f, 0.01236f }
#define CLANCZOS_NEG float4{ -0.09393f, 0.50941f, 0.50873f, 1.21916f }
#define CLANCZOS CLANCZOS_POS, CLANCZOS_NEG, CLANCZOS_WEIGHT
// Box
#define CLANCZOS_BOX_WEIGHT 0.27468f
#define CLANCZOS_BOX_POS float4{ -0.42461f, 0.80229f, 0.56141f, 0.01275f }
#define CLANCZOS_BOX_NEG float4{ -0.17198f, 0.43065f, 0.47907f, 0.96529f }
#define CLANCZOS_BOX CLANCZOS_BOX_POS, CLANCZOS_BOX_NEG, LANCZOS_BOX_NWEIGHT
// Tent
#define CLANCZOS_TENT_WEIGHT 0.57724f
#define CLANCZOS_TENT_POS float4{ -0.47606f, 0.74485f, 0.45055f, 0.01083f }
#define CLANCZOS_TENT_NEG float4{ -0.16013f, 0.31668f, 0.40466f, 0.74300f }
#define CLANCZOS_TENT CLANCZOS_TENT_POS, CLANCZOS_TENT_NEG, CLANCZOS_TENT_WEIGHT

/// Evaluate a cubic where the x coordinate (in [0,1)) is stretched on both ends by a sqrt
OTK_INLINE OTK_HOSTDEVICE float stretchedCubic01( float4 coeff, float x )
{
    const float scale = 0.5f / sqrtf(0.5f);
    x = (x <= 0.5f) ? scale * sqrtf( x ) : 1.0f - scale * sqrtf( 1.0f - x );
    return otk::dot( coeff, float4{x*x*x, x*x, x, 1.0f} );
}

/// Evaluate a cubic where the x coordinate (in [0,1)) is stretched at the 1 end by a sqrt
OTK_INLINE OTK_HOSTDEVICE float stretchedCubic1( float4 coeff, float x )
{
    x = 1.0f - sqrtf( 1.0f - x );
    return otk::dot( coeff, float4{x*x*x, x*x, x, 1.0f} );
}

/// Compute the sign of a number
OTK_INLINE OTK_HOSTDEVICE float sign( float x ) { return (x < 0.0f) ? -1.0f : 1.0f; }

/// Fold and scale a number in [0, 1) so that (0.5, 0], and [0.5, 1) map to [0, 1)
OTK_INLINE OTK_HOSTDEVICE float fold( float x ) { return 2.0f * fabsf( x - 0.5f ); }

/// Sample one of the positive lobes of a sharpening kernel stored as stretched cubics
OTK_INLINE OTK_HOSTDEVICE float2 sampleSharpenPos( float4 posCoeff, float4 negCoeff, float negWeight, float2 xi )
{
    float2 flip = float2{ sign(xi.x - 0.5f), sign(xi.y - 0.5f) };
    float x = fold( xi.x );
    float y = fold( xi.y );
    float p = 1.0f / ( 1.0f + negWeight );

    if( x < p )
    {
        return flip * float2{ stretchedCubic1( posCoeff, x / p ), stretchedCubic1( posCoeff, y ) };	
    }
    return flip * float2{ stretchedCubic01( negCoeff, ( x - p ) / negWeight ), stretchedCubic01( negCoeff, y ) };
}

/// Sample one of the negative lobes of a sharpening kernel stored as stretched cubics
OTK_INLINE OTK_HOSTDEVICE float2 sampleSharpenNeg( float4 posCoeff, float4 negCoeff, float negWeight, float2 xi )
{
    float2 flip = float2{ sign(xi.x - 0.5f), sign(xi.y - 0.5f) };
    float x = fold( xi.x );
    float y = fold( xi.y );

    if( x <= 0.5f ) // both lobes have probability 0.5
    {
		return flip * float2{ stretchedCubic1( posCoeff, x * 2.0f ), stretchedCubic01( negCoeff, y ) };
    }
	return flip * float2{ stretchedCubic01( negCoeff, ( x - 0.5f ) * 2.0f ), stretchedCubic1( posCoeff, y ) };
}
