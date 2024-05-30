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
