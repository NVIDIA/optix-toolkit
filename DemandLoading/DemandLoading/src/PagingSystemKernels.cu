// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "PagingSystemKernels.h"

#include <OptiXToolkit/DemandLoading/Paging.h>

namespace demandLoading {

__device__ __forceinline__ unsigned int countSetBits( unsigned int bits )
{
    return __popc( bits );
}

__device__ __forceinline__ unsigned int countStaleBitsAboveThreshold( unsigned int  staleBits,
                                                                      unsigned int  pageBitOffset,
                                                                      unsigned int* lruTable,
                                                                      unsigned int  lruThreshold,
                                                                      unsigned int  launchNum )
{
    // If no lruTable, count all the stale bits
    if( !lruTable )
        return __popc( staleBits );

    // Count stale bits (resident but not referenced) with lru val >= threshold
    unsigned int numSetBits = 0;
    while( staleBits != 0 )
    {
        // Find index of least significant bit and clear it
        unsigned int bitIndex = __ffs( staleBits ) - 1;
        staleBits ^= ( 1U << bitIndex );

        // Increment the count if the lru value is above the threshold
        unsigned int pageId = pageBitOffset + bitIndex;
        unsigned int oldVal = getHalfByte( pageId, lruTable );
        unsigned int newVal = lruInc( oldVal, launchNum + pageId );
        if( newVal != oldVal )
            atomicAddHalfByte( pageId, 1, lruTable );
        if( newVal >= lruThreshold && newVal < NON_EVICTABLE_LRU_VAL )
            numSetBits++;
    }
    return numSetBits;
}

__device__ __forceinline__ void resetLruCountersForFreshPages( unsigned int freshBits, unsigned int pageBitOffset, unsigned int* lruTable )
{
    if( !lruTable )
        return;

    while( freshBits != 0 )
    {
        unsigned int bitIndex = __ffs( freshBits ) - 1;
        freshBits ^= ( 1U << bitIndex );
        unsigned int pageId = pageBitOffset + bitIndex;
        unsigned int lruVal = getHalfByte( pageId, lruTable );
        if( lruVal != 0 && lruVal != NON_EVICTABLE_LRU_VAL )
            atomicClearHalfByte( pageId, lruTable );
    }
}

__device__ __forceinline__ unsigned int calcListStartIndex( const unsigned int laneId, unsigned int numSetBits, unsigned int* pageCount )
{
    unsigned int index = numSetBits;

// Calculate the index where the current thread can start adding pages to a global list.
// Coordinate within warps to avoid overuse of atomics.

#if defined( __CUDACC__ )
#pragma unroll
#endif
    // Compute total set bits in the warp (stored in numSetBits)
    // Also compute a prefix sum set bits over the warp (stored in index)
    for( unsigned int i = 1; i < 32; i *= 2 )
    {
        numSetBits += __shfl_xor_sync( 0xFFFFFFFF, numSetBits, i );
        unsigned int n = __shfl_up_sync( 0xFFFFFFFF, index, i );

        if( laneId >= i )
            index += n;
    }
    index = __shfl_up_sync( 0xFFFFFFFF, index, 1 );  // Shift index values by 1 thread

    // The thread at laneId 0 from each warp updates the global pageCount.
    int warpStartIndex;
    if( laneId == 0 )
    {
        index = 0;  // index for laneId 0 wasn't zeroed out in the __shfl_up_sync above
        if( numSetBits )
            warpStartIndex = atomicAdd( pageCount, numSetBits );
    }

    // The start index for the thread is the warpStartIndex plus the index within the warp
    return __shfl_sync( 0xFFFFFFFF, warpStartIndex, 0 ) + index;
}


__device__ __forceinline__ void addPagesToList( unsigned int  startingIndex,
                                                unsigned int  pageBits,
                                                unsigned int  pageBitOffset,
                                                unsigned int  maxCount,
                                                unsigned int* outputArray )
{
    while( pageBits != 0 && ( startingIndex < maxCount ) )
    {
        // Find index of least significant bit and clear it
        unsigned int bitIndex = __ffs( pageBits ) - 1;
        pageBits ^= ( 1U << bitIndex );

        // Add the requested page to the queue
        outputArray[startingIndex++] = pageBitOffset + bitIndex;
    }
}

__device__ __forceinline__ void addStalePagesToList( unsigned int        startingIndex,
                                                     unsigned int        pageBits,
                                                     unsigned int        pageBitOffset,
                                                     unsigned int        maxCount,
                                                     unsigned long long* pageTable,
                                                     unsigned int*       lruTable,
                                                     unsigned int        lruThreshold,
                                                     StalePage*          outputArray )
{
    while( pageBits != 0 && ( startingIndex < maxCount ) )
    {
        // Find index of least significant bit and clear it
        unsigned int bitIndex = __ffs( pageBits ) - 1;
        pageBits ^= ( 1U << bitIndex );

        // Add the requested page to the queue
        unsigned int pageId = pageBitOffset + bitIndex;
        unsigned int lruVal = ( lruTable != nullptr ) ? getHalfByte( pageId, lruTable ) : MAX_LRU_VAL;

        if( lruVal >= lruThreshold && lruVal != NON_EVICTABLE_LRU_VAL )
        {
            outputArray[startingIndex++] = StalePage{0, lruVal, pageId};
        }
    }
}


// The context was passed by value, so it resides in device memory.
__global__ void devicePullRequests( DeviceContext context, unsigned int launchNum, unsigned int lruThreshold, unsigned int startPage, unsigned int endPage )
{
    const unsigned int startIndex = startPage / 32;
    const unsigned int endIndex   = ( endPage + 31 ) / 32;

    unsigned int       globalIndex = threadIdx.x + blockIdx.x * blockDim.x + startIndex;
    const unsigned int laneId      = globalIndex % 32;

    while( globalIndex < endIndex )
    {
        // Compute rotated reference/residence index
        unsigned int referenceWordIndex = globalIndex;
        if( referenceWordIndex > endIndex )
            referenceWordIndex -= ( 1 + endIndex - startIndex );
        const unsigned int pageBitOffset = referenceWordIndex * 32;

        const unsigned int referenceWord = context.referenceBits[referenceWordIndex];
        const unsigned int residenceWord = context.residenceBits[referenceWordIndex];

        // Gather requested pages (reference bit true, but not resident),
        // and add them to the request list.
        const unsigned int requestedPages = referenceWord & ~residenceWord;
        const unsigned int numRequestBits = countSetBits( requestedPages );
        const unsigned int requestIndex =
            calcListStartIndex( laneId, numRequestBits, &context.arrayLengths.data[PAGE_REQUESTS_LENGTH] );
        addPagesToList( requestIndex, requestedPages, pageBitOffset, context.requestedPages.capacity,
                        context.requestedPages.data );

        // Only do the work of finding stale pages when they are requested
        if( context.stalePages.capacity > 0 )
        {
            // Reset LRU counters for fresh pages (requested and resident)
            const unsigned int freshPages = referenceWord & residenceWord;
            resetLruCountersForFreshPages( freshPages, pageBitOffset, context.lruTable );

            // Gather the stale pages (resident but not requested) with lruVal >= lruThreshold,
            // and add them to the the stale pages list
            const unsigned int stalePages = ~referenceWord & residenceWord;
            const unsigned int numStaleBits =
                countStaleBitsAboveThreshold( stalePages, pageBitOffset, context.lruTable, lruThreshold, launchNum );
            const unsigned int staleIndex =
                calcListStartIndex( laneId, numStaleBits, &context.arrayLengths.data[STALE_PAGES_LENGTH] );
            addStalePagesToList( staleIndex, stalePages, pageBitOffset, context.stalePages.capacity,
                                 context.pageTable.data, context.lruTable, lruThreshold, context.stalePages.data );
        }

        globalIndex += gridDim.x * blockDim.x;
    }

    // TODO: Gather the evictable pages?

    // Clamp counts of returned pages, since they may have been over-incremented
    if( laneId == 0 )
    {
        atomicMin( &context.arrayLengths.data[PAGE_REQUESTS_LENGTH], context.requestedPages.capacity );
        atomicMin( &context.arrayLengths.data[STALE_PAGES_LENGTH], context.stalePages.capacity );
    }
}

__global__ void devicePushMappings( unsigned long long* pageTable,
                                    unsigned int        numPageTableEntries,
                                    unsigned int*       residenceBits,
                                    unsigned int*       lruTable,
                                    PageMapping*        devFilledPages,
                                    int                 filledPageCount )
{
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    while( globalIndex < filledPageCount )
    {
        const PageMapping& devFilledPage = devFilledPages[globalIndex];
        // Page table entries are use only for samplers, not tiles.
        if( devFilledPage.id < numPageTableEntries )
            pageTable[devFilledPage.id] = devFilledPage.page;
        atomicSetBit( devFilledPage.id, residenceBits );

        // Set the LRU value
        if( lruTable )
        {
            atomicClearHalfByte( devFilledPage.id, lruTable );
            if( devFilledPage.lruVal != 0 ) // avoid double atomic in common case where lruVal = 0
                atomicOrHalfByte( devFilledPage.id, devFilledPage.lruVal, lruTable );
        }

        globalIndex += gridDim.x * blockDim.x;
    }
}

__global__ void deviceInvalidatePages( unsigned int* residenceBits, unsigned int* devInvalidatedPages, int invalidatedPageCount )
{
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    while( globalIndex < invalidatedPageCount )
    {
        atomicUnsetBit( devInvalidatedPages[globalIndex], residenceBits );
        globalIndex += gridDim.x * blockDim.x;
    }
}

}  // namespace demandLoading
