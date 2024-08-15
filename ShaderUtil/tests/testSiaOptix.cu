// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "testSia.h"

#include <float.h>
#include <math.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <curand_kernel.h>
#include <cstdint>

using namespace otk;  // for vec_math operators

extern "C" {
    __constant__ Params params;
}

// generate a normal 3D vector from polar coordinates
__device__ inline float3 polar( float phi, float theta )
{
    float cphi = cosf( phi );
    float sphi = sinf( phi );

    float ctheta = cosf( theta );
    float stheta = sinf( theta );

    return make_float3(
        cphi * ctheta,
        cphi * stheta,
        sphi );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint32_t id = idx.x + dim.x * idx.y;

    params.spawn[id].set = false;

    /* Each thread gets same seed, a different sequence
       number, no offset */
    curandState state;
    curand_init( 0, id, 0, &state );

    float2 bary;
    bary.x = curand_uniform( &state );
    bary.y = curand_uniform( &state ) * ( 1 - bary.x );

    float3 v0 = ( 1 - params.time ) * params.vertices[0] + params.time * params.vertices[1];
    float3 v1 = ( 1 - params.time ) * params.vertices[2] + params.time * params.vertices[3];
    float3 v2 = ( 1 - params.time ) * params.vertices[4] + params.time * params.vertices[5];

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    // setup object space ray aiming for the triangle
    float3 orig = v0 + bary.x * e1 + bary.y * e2;

    // transform to world ray
    OptixTraversableHandle tlist[MAX_GRAPH_DEPTH];
    for( unsigned i = 0; i < params.depth; ++i )
    {
        OptixTraversableHandle handle = params.handles[i];
        OptixTransformType     type   = optixGetTransformTypeFromHandle( handle );

        // take first instance in each IAS
        if( type == OPTIX_TRANSFORM_TYPE_NONE )
            handle = optixGetInstanceTraversableFromIAS( handle, 0 );

        float4 trf0, trf1, trf2;
        optix_impl::optixGetInterpolatedTransformationFromHandle( trf0, trf1, trf2, handle, params.time, /*objectToWorld*/ true );

        float3 o;
        o.x = orig.x * trf0.x + orig.y * trf0.y + orig.z * trf0.z + trf0.w;
        o.y = orig.x * trf1.x + orig.y * trf1.y + orig.z * trf1.z + trf1.w;
        o.z = orig.x * trf2.x + orig.y * trf2.y + orig.z * trf2.z + trf2.w;

        orig = o;

        // construct a world-to-object tlist
        tlist[params.depth-i-1] = handle;
    };

    // random world direction
    float3 dir = polar( curand_uniform( &state ) * M_PIf, curand_uniform( &state ) * 2 * M_PIf );

    // aim for triangle spawn point
    orig -= dir;

    unsigned p0 = 0;
    optixTrace( params.root, orig, dir, 0.f, FLT_MAX, params.time, 1, 0, 0, 0, 0, p0);

    // validate that the context-free offset matches the hit-context offset
    if( p0 == 0 )
    {
        SpawnPoint spawn;
        spawn.set = true;

        // generate object space spawn point and offset
        SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( spawn.objPos, spawn.objNorm, spawn.objOffset, v0, v1, v2, params.barys[id] );

        // convert object space spawn point and offset to world space
        SelfIntersectionAvoidance::transformSafeSpawnOffset( spawn.wldPos, spawn.wldNorm, spawn.wldOffset, spawn.objPos, spawn.objNorm, spawn.objOffset, params.time, params.depth, tlist );

        // offset world space spawn point to generate self intersection safe front and back spawn points
        SelfIntersectionAvoidance::offsetSpawnPoint( spawn.wldFront, spawn.wldBack, spawn.wldPos, spawn.wldNorm, spawn.wldOffset );

        if( spawn != params.spawn[id] )
        {
            atomicAdd( &params.stats->contextContextFreeMissmatch, 1 );
        }
    }
}

__device__ void swap( float3& a, float3& b )
{
    float3 tmp = a;
    a = b;
    b = tmp;
}

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ( "mov.u32 %0, %laneid;" : "=r"( ret ) );
    return ret;
}

extern "C" __global__ void __closesthit__ch()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    unsigned id = idx.x + dim.x * idx.y;
    params.barys[id] = optixGetTriangleBarycentrics();

    float3 scatterDir = optixGetWorldRayDirection();

    SpawnPoint spawn;
    spawn.set = true;
   
    if( optixIsTriangleHit() )
    {
        // generate object space spawn point and offset
        SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( spawn.objPos, spawn.objNorm, spawn.objOffset );
    }
    else
    {
        // ...
    }

    // convert object space spawn point and offset to world space
    SelfIntersectionAvoidance::transformSafeSpawnOffset( spawn.wldPos, spawn.wldNorm, spawn.wldOffset, spawn.objPos, spawn.objNorm, spawn.objOffset );

    // offset world space spawn point to generate self intersection safe front and back spawn points
    SelfIntersectionAvoidance::offsetSpawnPoint( spawn.wldFront, spawn.wldBack, spawn.wldPos, spawn.wldNorm, spawn.wldOffset );

    params.spawn[id] = spawn;

    // pick safe spawn point for secondary scatter ray
    uint32_t rayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;

    // filter out backward facing hits for grazing rays
    float3 front = spawn.wldFront, back = spawn.wldBack;
    if( dot( scatterDir, spawn.wldNorm ) > 0.f )
    {
        rayFlags |= OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
    }
    else
    {
        swap( front, back );
        rayFlags |= OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    }

    // shoot ray from the front and the back
    unsigned p0 = 0, p1 = 0;
    optixTrace( params.root, front, scatterDir, 0.f, FLT_MAX, params.time, 1, rayFlags, 0, 0, 0, p0 );
    optixTrace( params.root, back,  scatterDir, 0.f, FLT_MAX, params.time, 1, rayFlags, 0, 0, 0, p1 );

    unsigned activeMask = __activemask();
    unsigned frontMissBackHit  = __popc( __ballot_sync( activeMask, p0 == 1 && p1 == 0 ) );
    unsigned frontHitBackMiss  = __popc( __ballot_sync( activeMask, p0 == 0 && p1 == 1 ) );
    unsigned frontHitBackHit   = __popc( __ballot_sync( activeMask, p0 == 0 && p1 == 0 ) );
    unsigned frontMissBackMiss = __popc( __ballot_sync( activeMask, p0 == 1 && p1 == 1 ) );

    unsigned laneId = lane_id();
    if( ( ( ( 1 << laneId ) - 1 ) & activeMask ) == 0 )
    {
        if( frontMissBackHit )  atomicAdd( &params.stats->frontMissBackHit,  ( unsigned long long )frontMissBackHit  );
        if( frontHitBackMiss )  atomicAdd( &params.stats->frontHitBackMiss,  ( unsigned long long )frontHitBackMiss  );
        if( frontHitBackHit )   atomicAdd( &params.stats->frontHitBackHit,   ( unsigned long long )frontHitBackHit   );
        if( frontMissBackMiss ) atomicAdd( &params.stats->frontMissBackMiss, ( unsigned long long )frontMissBackMiss );
    }
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0( 1 );
}
