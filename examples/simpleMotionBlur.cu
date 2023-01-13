//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include <optix.h>

#include "simpleMotionBlur.h"
#include "random.h"

#include "vec_math.h"
#include "helpers.h"


extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ float3 traceCamera(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  ray_time
        )
{
    unsigned int r, g, b;

    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            0.0f,                     // tmin
            1e16f,                    // tmax
            ray_time,
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            r, g, b );

    return make_float3(
            __uint_as_float( r ),
            __uint_as_float( g ),
            __uint_as_float( b )
            );
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}


extern "C" __global__ void __raygen__rg()
{
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>( idx.y*w + idx.x, subframe_index );
    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter = make_float2( rnd( seed ), rnd( seed ) );

    const float2 d = 2.0f * make_float2(
            ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
            ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
            ) - 1.0f;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);
    float3 ray_origin    = eye;

    const float3 result        = traceCamera( params.handle, ray_origin, ray_direction, rnd( seed ) );

    const int image_index = idx.y*w + idx.x;
    float3 accum_color = result;
    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}


extern "C" __global__ void __miss__camera()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setPayload( rt_data->color );
}


extern "C" __global__ void __closesthit__camera()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    setPayload( rt_data->color );
}


extern "C" __global__ void __intersection__sphere()
{
    HitGroupData* hg_data  = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3 orig = optixGetObjectRayOrigin();
    const float3 dir  = optixGetObjectRayDirection();

    const float3 center = hg_data->center;
    const float  radius = hg_data->radius;

    const float3 O      = orig - center;
    const float  l      = 1 / length( dir );
    const float3 D      = dir * l;

    const float b    = dot( O, D );
    const float c    = dot( O, O ) - radius * radius;
    const float disc = b * b - c;
    if( disc > 0.0f )
    {
        const float sdisc = sqrtf( disc );
        const float root1 = ( -b - sdisc );

        const float        root11        = 0.0f;
        const float3       shading_normal = ( O + ( root1 + root11 ) * D ) / radius;
        unsigned int p0, p1, p2;
        p0 = __float_as_uint( shading_normal.x );
        p1 = __float_as_uint( shading_normal.y );
        p2 = __float_as_uint( shading_normal.z );

        optixReportIntersection(
                root1,      // t hit
                0,          // user hit kind
                p0, p1, p2
                );
    }
}
