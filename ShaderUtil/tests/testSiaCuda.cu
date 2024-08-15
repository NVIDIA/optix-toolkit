// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdint>

#include "testSia.h"

#include <OptiXToolkit/ShaderUtil/vec_math.h>

using namespace otk;  // for vec_math operators

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ( "mov.u32 %0, %laneid;" : "=r"( ret ) );
    return ret;
}

__global__ void cudaValidate( const Params params )
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < params.size && params.spawn[idx].set )
    {
        // generate cuda spawn point
        SpawnPoint spawn;
        spawn.set = true;

        float3 v0 = ( 1 - params.time ) * params.vertices[0] + params.time * params.vertices[1];
        float3 v1 = ( 1 - params.time ) * params.vertices[2] + params.time * params.vertices[3];
        float3 v2 = ( 1 - params.time ) * params.vertices[4] + params.time * params.vertices[5];

        // generate object space spawn point and offset
        SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( spawn.objPos, spawn.objNorm, spawn.objOffset, v0, v1, v2, params.barys[idx] );

        // convert object space spawn point and offset to world space
        SelfIntersectionAvoidance::transformSafeSpawnOffset( spawn.wldPos, spawn.wldNorm, spawn.wldOffset, spawn.objPos, spawn.objNorm, spawn.objOffset, params.time, params.depth, params.transforms );

        // offset world space spawn point to generate self intersection safe front and back spawn points
        SelfIntersectionAvoidance::offsetSpawnPoint( spawn.wldFront, spawn.wldBack, spawn.wldPos, spawn.wldNorm, spawn.wldOffset );

        unsigned activeMask  = __activemask();
        unsigned missmatches = __popc( __ballot_sync( activeMask, spawn != params.spawn[idx] ) );

        // compare to optix spawn point
        unsigned laneId = lane_id();
        if( ( ( ( 1 << laneId ) - 1 ) & activeMask ) == 0 )
        {
            if( missmatches ) atomicAdd( &params.stats->optixCudaMissmatch, (unsigned long long)missmatches );
        }
    }
}

cudaError_t launchCudaValidate( const Params params, unsigned int size, cudaStream_t stream )
{
    dim3     threadsPerBlock( 128, 1 );
    unsigned numBlocks = ( unsigned )( ( size + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( size )
        cudaValidate << <numBlocks, threadsPerBlock, 0, stream >> > ( params );
    return cudaGetLastError();
}
