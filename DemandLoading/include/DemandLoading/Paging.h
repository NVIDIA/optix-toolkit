//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <DemandLoading/DeviceContext.h>

namespace demandLoading {

#ifndef DOXYGEN_SKIP

const unsigned int MAX_LRU_VAL           = 14u;
const unsigned int NON_EVICTABLE_LRU_VAL = 15u;

#if defined( __CUDACC__ )

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

#endif  // __CUDACC__
#endif  // ndef DOXYGEN_SKIP

}  // namespace demandLoading
