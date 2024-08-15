// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Memory/DeviceRingBuffer.h>

#include <vector>

namespace otk {

// DeviceFixedPool implements a pool of fixed-size items as a circular queue.
// Items must be freed to be reused.

struct DeviceFixedPool
{
    char*        buffer;          // Backing store for the items in the pool
    char**       itemGroups;      // Pointers item groups (itemSize, or itemSize*warpSize bytes).
    char**       itemGroupsCopy;  // A copy of pointers item groups (to copy to itemGroups in launchPrepare)
    unsigned int numItemGroups;   // Total number of item groups. Must be a power of 2.
    unsigned int itemSize;        // The item size in bytes.
    AllocMode    allocMode;       // Which allocation mode

    unsigned int* nextItemGroupId;     // Index of the next item group to be allocated (allows 2^32 allocs per launch)
    unsigned int* discardItemGroupId;  // Index where next freed item group should be placed

// Host functions
#ifndef __CUDACC__

    /// Allocate buffers and initialize the pool
    void init( unsigned int itemSize_, unsigned int numItems, AllocMode allocMode_ )
    {
        itemSize                  = itemSize_;
        allocMode                 = allocMode_;
        size_t ws                 = ( allocMode == THREAD_BASED ) ? 1 : WARP_SIZE;
        numItemGroups             = numItems / ws;
        size_t itemGroupsBuffSize = numItemGroups * sizeof( char* );

        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &buffer ), static_cast<size_t>( itemSize ) * numItems ) );
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &itemGroups ), itemGroupsBuffSize ) );
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &itemGroupsCopy ), itemGroupsBuffSize ) );
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &nextItemGroupId ), 2 * sizeof( unsigned int ) ) );
        discardItemGroupId = &nextItemGroupId[1];

        std::vector<char*> hostItemGroups( numItemGroups, nullptr );
        for( unsigned int itemGroupId   = 0; itemGroupId < numItemGroups; ++itemGroupId )
            hostItemGroups[itemGroupId] = buffer + ( itemSize * ws * itemGroupId );

        OTK_ERROR_CHECK( cuMemcpy( reinterpret_cast<CUdeviceptr>( itemGroupsCopy ),
                                         reinterpret_cast<CUdeviceptr>( hostItemGroups.data() ), itemGroupsBuffSize ) );
        clear( 0 );
    }

    /// Free all of the buffers allocated for the pool
    void tearDown()
    {
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( buffer ) ) );
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( itemGroups ) ) );
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( itemGroupsCopy ) ) );
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( nextItemGroupId ) ) );
    }

    /// Clear the allocator
    void clear( CUstream stream )
    {
        OTK_ERROR_CHECK(
            cuMemsetD8Async( reinterpret_cast<CUdeviceptr>( nextItemGroupId ), 0, 2 * sizeof( unsigned int ), stream ) );
        OTK_ERROR_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( itemGroups ), reinterpret_cast<CUdeviceptr>( itemGroupsCopy ),
                                              numItemGroups * sizeof( char* ), stream ) );
    }

#endif // Host functions

// Device functions
#ifdef __CUDACC__

    /// Allocate an item in circular-queue manner
    __forceinline__ __device__ char* alloc()
    {
        // Warp-based
        if( allocMode != THREAD_BASED )
        {
            const unsigned int activeMask = __activemask();
            const unsigned int leadLane   = __ffs( activeMask ) - 1;
            const unsigned int laneId     = getLaneId();

            unsigned long long ptrBase = 0ULL;
            if( laneId == leadLane )
            {
                unsigned int itemGroupId = atomicAdd( nextItemGroupId, 1u ) & ( numItemGroups - 1 );
                ptrBase = atomicExch( reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ), 0ULL );
            }
            ptrBase = __shfl_sync( activeMask, ptrBase, leadLane );

            if( allocMode == WARP_INTERLEAVED )
                ptrBase += INTERLEAVE_BYTES * laneId;
            else if ( allocMode == WARP_NON_INTERLEAVED )
                ptrBase += itemSize * laneId;
            return reinterpret_cast<char*>( ptrBase );
        }

        // Thread-based
        unsigned int itemGroupId = atomicAdd( nextItemGroupId, 1u ) & ( numItemGroups - 1 );
        unsigned long long ptr = atomicExch( reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ), 0ULL );
        return reinterpret_cast<char*>( ptr );
    }

    /// Free an item, placing it in the next open slot, warp-based
    __forceinline__ __device__ bool free( char* ptr )
    {
        if( ptr == nullptr )
            return true;

        // Warp-based
        if( allocMode != THREAD_BASED )
        {
            const unsigned int activeMask = __activemask();
            const unsigned int leadLane   = __ffs( activeMask ) - 1;
            const unsigned int laneId     = getLaneId();

            if( laneId == leadLane )
            {
                unsigned long long ptrBase = reinterpret_cast<unsigned long long>( ptr );
                if( allocMode == WARP_INTERLEAVED )
                    ptrBase -= INTERLEAVE_BYTES * laneId;
                else if( allocMode == WARP_NON_INTERLEAVED )
                    ptrBase -= itemSize * laneId;

                unsigned int itemGroupId = atomicAdd( discardItemGroupId, 1u ) & ( numItemGroups - 1 );
                unsigned long long p = atomicExch( reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ), ptrBase );
                return ( p == 0ULL );  // If p is not 0, the buffer overflowed
            }
            return true;
        }

        // Thread-based
        unsigned int itemGroupId = atomicAdd( discardItemGroupId, 1u ) & ( numItemGroups - 1 );
        unsigned long long p = atomicExch( reinterpret_cast<unsigned long long*>( &itemGroups[itemGroupId] ),
                                           reinterpret_cast<unsigned long long>( ptr ) );
        return ( p == 0ULL );  // If p is not 0, the buffer overflowed
    }

    /// Get the current number of free items.
    __forceinline__ __device__ unsigned int numFreeItems()
    {
        // Warp-based
        if( allocMode != THREAD_BASED )
        {
            const unsigned int activeMask = __activemask();
            const unsigned int leadLane   = __ffs( activeMask ) - 1;
            const unsigned int laneId     = getLaneId();

            unsigned int numGroups = 0;
            if( laneId == leadLane )
            {
                numGroups = numItemGroups - ( atomicAdd( nextItemGroupId, 0 ) - atomicAdd( discardItemGroupId, 0 ) );
            }
            return WARP_SIZE * __shfl_sync( activeMask, numGroups, leadLane );
        }

        // Thread-based
        return numItemGroups - ( atomicAdd( nextItemGroupId, 0 ) - atomicAdd( discardItemGroupId, 0 ) );
    }

#endif // Device functions
};

}  // namespace otk
