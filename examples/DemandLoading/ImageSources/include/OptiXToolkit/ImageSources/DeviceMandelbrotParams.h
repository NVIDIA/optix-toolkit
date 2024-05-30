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

namespace imageSources {

const int MAX_MANDELBROT_COLORS = 16;

struct MandelbrotParams
{
    // Width and height of image
    unsigned int width;
    unsigned int height;

    // Clip edge of image for odd-size tiles
    unsigned int clip_width;
    unsigned int clip_height;

    // Whether to load all mip levels, or just 1
    bool all_mip_levels;

    // Portion of complex plane to cover
    double xmin;
    double ymin;
    double xmax;
    double ymax;

    // Mandelbrot params
    int max_iterations;
    float4 colors[MAX_MANDELBROT_COLORS];
    int num_colors;

    float4* output_buffer;
};

__host__ __device__ __forceinline__ float4 mandelbrotColor( double x, double y, const MandelbrotParams& params )
{
    double2      z{x, y};
    int          n      = 0;
    const double maxval = 4.0f;

    for( n = 0; n < params.max_iterations; ++n )
    {
        if( z.x * z.x + z.y * z.y > maxval )
            break;
        z = {( x + z.x * z.x - z.y * z.y ), ( y + z.x * z.y + z.y * z.x )};
    }

    if( n == params.max_iterations )
        return float4{0.0f, 0.0f, 0.0f, 0.0f};

    float v = static_cast<float>( n ) / static_cast<float>( params.max_iterations );

    float f = static_cast<float>( v * ( params.num_colors - 1 ) );
    int   i = static_cast<unsigned int>( f );
    if( i > params.num_colors - 2 )
        i = params.num_colors - 2;
    f = sqrtf( static_cast<float>( f - i ) );

    const float4 c0 = params.colors[i];
    const float4 c1 = params.colors[i + 1];
    return float4{( c0.x * ( 1.0f - f ) + c1.x * f ), ( c0.y * ( 1.0f - f ) + c1.y * f ),
                  ( c0.z * ( 1.0f - f ) + c1.z * f ), ( c0.w * ( 1.0f - f ) + c1.w * f )};
}

} // namespace imageSources
