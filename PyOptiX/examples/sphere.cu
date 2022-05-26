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

#include "sphere.h"
#include "helpers.h"

#include "vec_math.h"

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd
        )
{
    unsigned int p0, p1, p2;
    p0 = float_as_int( prd->x );
    p1 = float_as_int( prd->y );
    p2 = float_as_int( prd->z );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd->x = int_as_float( p0 );
    prd->y = int_as_float( p1 );
    prd->z = int_as_float( p2 );
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            int_as_float( optixGetPayload_0() ),
            int_as_float( optixGetPayload_1() ),
            int_as_float( optixGetPayload_2() )
            );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rt_data = reinterpret_cast<RayGenData*>( optixGetSbtDataPointer() );

    const float3 U = rt_data->camera_u;
    const float3 V = rt_data->camera_v;
    const float3 W = rt_data->camera_w;

    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    const float3 origin      = rt_data->cam_eye;
    const float3 direction   = normalize( d.x * U + d.y * V + W );
    float3       payload_rgb = make_float3( 0.5f, 0.5f, 0.5f );
    trace( params.handle,
            origin,
            direction,
            0.00f,  // tmin
            1e16f,  // tmax
            &payload_rgb );

    params.image[idx.y * params.image_width + idx.x] = make_color( payload_rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3    payload = getPayload();
    setPayload( make_float3( rt_data->r, rt_data->g, rt_data->b ) );
}


extern "C" __global__ void __closesthit__ch()
{
    const float3 shading_normal =
        make_float3(
                int_as_float( optixGetAttribute_0() ),
                int_as_float( optixGetAttribute_1() ),
                int_as_float( optixGetAttribute_2() )
                );
    setPayload( normalize( optixTransformNormalFromObjectToWorldSpace( shading_normal ) ) * 0.5f + 0.5f );
}


#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" __global__ void __intersection__sphere()
{
    const SphereHitGroupData* hit_group_data = reinterpret_cast<SphereHitGroupData*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const Sphere sphere = hit_group_data->sphere;

    const float3 O      = ray_orig - ( sphere.center_x, sphere.center_y, sphere.center_z );
    const float  l      = 1.0f / length( ray_dir );
    const float3 D      = ray_dir * l;
    const float  radius = sphere.radius;

    float b    = dot( O, D );
    float c    = dot( O, O ) - radius * radius;
    float disc = b * b - c;
    if( disc > 0.0f )
    {
        float sdisc        = sqrtf( disc );
        float root1        = ( -b - sdisc );
        float root11       = 0.0f;
        bool  check_second = true;

        const bool do_refine = fabsf( root1 ) > ( 10.0f * radius );

        if( do_refine )
        {
            // refine root1
            float3 O1 = O + root1 * D;
            b         = dot( O1, D );
            c         = dot( O1, O1 ) - radius * radius;
            disc      = b * b - c;

            if( disc > 0.0f )
            {
                sdisc  = sqrtf( disc );
                root11 = ( -b - sdisc );
            }
        }

        float  t;
        float3 normal;
        t = ( root1 + root11 ) * l;
        if( t > ray_tmin && t < ray_tmax )
        {
            normal = ( O + ( root1 + root11 ) * D ) / radius;
            if( optixReportIntersection( t, 0, float3_as_ints( normal ), float_as_int( radius ) ) )
                check_second = false;
        }

        if( check_second )
        {
            float root2 = ( -b + sdisc ) + ( do_refine ? root1 : 0 );
            t           = root2 * l;
            normal      = ( O + root2 * D ) / radius;
            if( t > ray_tmin && t < ray_tmax )
                optixReportIntersection( t, 0, float3_as_ints( normal ), float_as_int( radius ) );
        }
    }
}