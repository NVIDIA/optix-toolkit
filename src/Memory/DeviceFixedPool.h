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
#include "DeviceRingBuffer.h"
#include <vector>

// DeviceFixedPool implements a pool of fixed-size items as a circular queue.
// Items must be freed to be reused.

struct DeviceFixedPool
{
    char* buffer;                // Backing store for the items in the pool
    char** itemGroups;           // Pointers item groups (itemSize, or itemSize*warpSize bytes).
    char** itemGroupsCopy;       // A copy of pointers item groups (to copy to itemGroups in launchPrepare)
    unsigned int numItemGroups;  // Total number of item groups. Must be a power of 2.
    unsigned int itemSize;       // The item size in bytes.  
    bool allocByWarp;            // Whether to allocate by whole warps, or by individual threads
    
    unsigned int* nextItemGroupId;     // Index of the next item group to be allocated (allows 2^32 allocs per launch)
    unsigned int* discardItemGroupId;  // Index where next freed item group should be placed

// Host functions
#ifndef __CUDACC__
    /// Allocate buffers and initialize the pool
    __host__ void init(unsigned int _itemSize, unsigned int numItems, bool _allocByWarp )
    {
        itemSize = _itemSize;
        allocByWarp = _allocByWarp;
        size_t ws = allocByWarp ? WARP_SIZE : 1;
        numItemGroups = numItems / ws;
        size_t itemGroupsBuffSize = numItemGroups * sizeof(char*);

        DEMAND_CUDA_CHECK( cudaMalloc( &buffer, static_cast<size_t>(itemSize) * numItems ) );
        DEMAND_CUDA_CHECK( cudaMalloc( &itemGroups, itemGroupsBuffSize ) );
        DEMAND_CUDA_CHECK( cudaMalloc( &itemGroupsCopy, itemGroupsBuffSize ) );
        DEMAND_CUDA_CHECK( cudaMalloc( &nextItemGroupId, 2 * sizeof(unsigned int) ) );
        discardItemGroupId = &nextItemGroupId[1];

        std::vector<char*> hostItemGroups( numItemGroups, nullptr );
        for( unsigned int itemGroupId = 0; itemGroupId < numItemGroups; ++itemGroupId )
            hostItemGroups[itemGroupId] = buffer + ( itemSize * ws * itemGroupId );

        DEMAND_CUDA_CHECK( cudaMemcpy( itemGroupsCopy, hostItemGroups.data(), itemGroupsBuffSize, cudaMemcpyHostToDevice ) );
        clear( 0 );
    }

    /// Free all of the buffers allocated for the pool
    __host__ void tearDown()
    {
        DEMAND_CUDA_CHECK( cudaFree( buffer ) );
        DEMAND_CUDA_CHECK( cudaFree( itemGroups ) );
        DEMAND_CUDA_CHECK( cudaFree( itemGroupsCopy ) );
        DEMAND_CUDA_CHECK( cudaFree( nextItemGroupId ) );
    }

    /// Clear the allocator
    __host__ void clear( CUstream stream )
    {
        DEMAND_CUDA_CHECK( cudaMemsetAsync( nextItemGroupId, 0, 2 * sizeof(unsigned int), stream ) ) ;
        DEMAND_CUDA_CHECK( cudaMemcpyAsync( itemGroups, itemGroupsCopy, numItemGroups * sizeof(char*), cudaMemcpyDeviceToDevice, stream ) );
    }
#endif

// Device functions
#ifdef __CUDACC__

    /// Allocate an item in circular-queue manner
    __forceinline__ __device__
    char* alloc()
    {
        // Warp-based
        if( allocByWarp )
        {
            const unsigned int activeMask = __activemask();
            const unsigned int leadLane   = __ffs( activeMask ) - 1;
            const unsigned int laneId     = getLaneId();
            
            unsigned long long ptrBase = 0ULL;
            if( laneId == leadLane )
            {
                unsigned int itemGroupId = atomicAdd(nextItemGroupId, 1u) & (numItemGroups-1);
                ptrBase = atomicExch( reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ), 0ULL );
            }
            ptrBase = __shfl_sync( activeMask, ptrBase, leadLane );

            return (ptrBase) ? reinterpret_cast<char*>( ptrBase + ( itemSize * laneId ) ) : nullptr;
        }

        // Thread-based
        unsigned int itemGroupId = atomicAdd(nextItemGroupId, 1u) & (numItemGroups-1);
        unsigned long long ptr = atomicExch( reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ), 0ULL );
        return reinterpret_cast<char*>( ptr );
    }

    /// Free an item, placing it in the next open slot, warp-based
    __forceinline__ __device__
    bool free(char* ptr)
    {
        if( ptr == nullptr )
            return true;

        // Warp-based
        if( allocByWarp )
        {
            const unsigned int activeMask = __activemask();
            const unsigned int leadLane   = __ffs( activeMask ) - 1; 
            const unsigned int laneId     = getLaneId();

            if( laneId == leadLane )
            {
                unsigned long long ptrBase = reinterpret_cast<unsigned long long>( ptr ) - ( itemSize * leadLane );
                unsigned int itemGroupId = atomicAdd(discardItemGroupId, 1u) & (numItemGroups-1);
                unsigned long long p = atomicExch(reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ), ptrBase );
                return (p == 0ULL); // If p is not 0, the buffer overflowed
            }
            return true;
        }
    
        // Thread-based
        unsigned int itemGroupId = atomicAdd(discardItemGroupId, 1u) & (numItemGroups-1);
        unsigned long long p = atomicExch(reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ), 
                                          reinterpret_cast<unsigned long long>( ptr ) );
        return (p == 0ULL); // If p is not 0, the buffer overflowed
    }

    /// Get the current number of free items.
    __forceinline__ __device__
    unsigned int numFreeItems()
    {
        // Warp-based
        if( allocByWarp )
        {
            const unsigned int activeMask = __activemask();
            const unsigned int leadLane   = __ffs( activeMask ) - 1;
            const unsigned int laneId     = getLaneId();
            
            unsigned int numGroups = 0;
            if( laneId == leadLane )
            {
                numGroups = numItemGroups - (atomicAdd(nextItemGroupId, 0) - atomicAdd(discardItemGroupId, 0));
            }
            return WARP_SIZE * __shfl_sync( activeMask, numGroups, leadLane );
        }

        // Thread-based
        return numItemGroups - (atomicAdd(nextItemGroupId, 0) - atomicAdd(discardItemGroupId, 0));
    }
#endif
};
