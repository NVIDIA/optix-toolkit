// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/LRU.h>

#include <cuda.h>

namespace demandLoading {

#ifndef DOXYGEN_SKIP

__device__ __forceinline__ unsigned char lruInc( unsigned int count, unsigned int launchNum )
{
    // Logarithmic increment. Slows down as count increases, assuming launchNum increases by 1 each time.
    // If 0 is always passed in for launchNum, the count will increment every launch.
    unsigned int mask = ( 1u << count ) - 1;
    return ( ( mask & launchNum ) == 0 && count < MAX_LRU_VAL ) ? count + 1u : count;
}

__device__ inline void atomicSetBit( unsigned int bitIndex, unsigned int* bitVector )
{
    const unsigned int wordIndex = bitIndex / 32;
    const unsigned int bitOffset = bitIndex % 32;
    const unsigned int mask      = 1U << bitOffset;
    atomicOr( bitVector + wordIndex, mask );
}

__device__ inline void atomicUnsetBit( int bitIndex, unsigned int* bitVector )
{
    const int wordIndex = bitIndex / 32;
    const int bitOffset = bitIndex % 32;

    const int mask = ~( 1U << bitOffset );
    atomicAnd( bitVector + wordIndex, mask );
}

__device__ inline bool checkBitSet( unsigned int bitIndex, const unsigned int* bitVector )
{
    const unsigned int wordIndex = bitIndex / 32;
    const unsigned int bitOffset = bitIndex % 32;
    return ( bitVector[wordIndex] & ( 1U << bitOffset ) ) != 0;
}

__device__ inline void atomicAddHalfByte( int index, unsigned int val, unsigned int* words )
{
    unsigned int wordIndex = index >> 3;
    unsigned int shiftVal  = val << ( 4 * ( index & 0x7 ) );
    atomicAdd( &words[wordIndex], shiftVal );
}

__device__ inline void atomicOrHalfByte( int index, unsigned int mask, unsigned int* words )
{
    unsigned int wordIndex = index >> 3;
    unsigned int wordMask  = mask << ( 4 * ( index & 0x7 ) );
    atomicOr( &words[wordIndex], wordMask );
}

__device__ inline void atomicClearHalfByte( int index, unsigned int* words )
{
    unsigned int wordIndex = index >> 3;
    unsigned int wordMask  = ~( 0xf << ( 4 * ( index & 0x7 ) ) );
    atomicAnd( &words[wordIndex], wordMask );
}

__device__ inline unsigned int getHalfByte( unsigned int index, unsigned int* words )
{
    unsigned int wordIndex = index >> 3;
    unsigned int shiftVal  = 4 * ( index & 0x7 );
    return ( words[wordIndex] >> shiftVal ) & 0xf;
}

__device__ inline unsigned long long pagingMapOrRequest( const DeviceContext& context, unsigned int page, bool* valid )
{
    bool requested = checkBitSet( page, context.referenceBits );
    if( !requested )
        atomicSetBit( page, context.referenceBits );

    bool mapped = checkBitSet( page, context.residenceBits );
    *valid      = mapped;

    return ( mapped && context.pageTable.data ) ? context.pageTable.data[page] : 0;
}

__device__ inline void pagingRequest( unsigned int* referenceBits, unsigned int page )
{
    bool requested = checkBitSet( page, referenceBits );
    if( !requested )
        atomicSetBit( page, referenceBits );
}

__device__ inline void pagingRequestWord( unsigned int* wordPtr, unsigned int word )
{
    if( ( word != 0 ) && ( ( *wordPtr & word ) != word ) )
        atomicOr( wordPtr, word );
}

#endif  // ndef DOXYGEN_SKIP

}  // namespace demandLoading
