// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file AliasTable.h
/// Functions to invert and sample a pdf with an alias table

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <OptiXToolkit/ShaderUtil/Preprocessor.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK( call )                                                                                                                                                                                                                                                             \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t error = call;                                                                                                                                                                                                                                                      \
        if( error != cudaSuccess )                                                                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                                                                              \
            return error;                                                                                                                                                                                                                                             \
        }                                                                                                                                                                                                                                                                              \
    }
#endif

struct AliasRecord
{
    float prob;
    int alias;
};

struct AliasTable
{
    int size;
    AliasRecord* table;
};

/// Allocate an alias table on the device 
inline cudaError_t allocAliasTableDevice( AliasTable& at, int size )
{
    at.size = size;
    CUDA_CHECK( cudaMalloc(&at.table, at.size * sizeof(AliasRecord) ) );
    return cudaSuccess;
}

/// Free an alias table on the device
inline cudaError_t freeAliasTableDevice( AliasTable& at )
{
    CUDA_CHECK( cudaFree( at.table ) );
    return cudaSuccess;
}

/// Allocate an alias table on the host
inline void allocAliasTableHost( AliasTable& at, int size )
{
    at.size = size;
    at.table = (AliasRecord*) malloc( at.size * sizeof(AliasRecord) );
}

/// Free an alias table on the host
inline void freeAliasTableHost( AliasTable& at )
{
    free( at.table );
}

/// Create an alias table (on the host) from a pdf array of the same size 
inline void makeAliasTable( AliasTable& at, float* pdf )
{
    // find average
    double dAve = 0.0f;
    for( int i = 0; i < at.size; ++i )
        dAve += pdf[i];
    float ave = static_cast<float>( dAve / at.size );

    // Initialize above and below indices
    int a = 0; // "above average" index
    while( a < at.size - 1 && pdf[a] <= ave ) 
        a++;
    int b = 0; // "below average" index
    while( b < at.size - 1 && pdf[b] > ave ) 
        b++;

    // Initialize the table
    for( int i = 0; i < at.size; ++i )
    {
        // Find an index n with pdf[n] >= ave, where n is either a or b.
        int n; // next index
        if( (pdf[a] <= ave && pdf[a] >= 0.0f) || pdf[b] < 0.0f )
        {
            n = a;
            if( a < at.size-1 ) 
                a++;
            while( a < at.size-1 && ( pdf[a] <= ave || pdf[a] < 0.0f ) ) 
                a++;
        }
        else
        {
            n = b;
            if( b < at.size-1 ) 
                b++;
            while( b < at.size-1 && ( pdf[b] > ave || pdf[b] < 0.0f ) ) 
                b++;
        }

        at.table[n] = AliasRecord{ pdf[n] / ave, a };
        pdf[a] = pdf[a] - ave * (1.0f - pdf[n] / ave);
        pdf[n] = -1.0f;
    }
}

inline cudaError_t copyToDevice( AliasTable& atHost, AliasTable& atDev )
{
    return cudaMemcpy( atDev.table, atHost.table, atHost.size * sizeof(AliasRecord), cudaMemcpyHostToDevice );
}

OTK_INLINE OTK_HOSTDEVICE float frac( float x ) 
{
    return x - static_cast<int>( x );
}

OTK_INLINE OTK_HOSTDEVICE float clampUnderOne( float x )
{
    return ( x < 1.0f ) ? x : 0.9999999f;
}

OTK_INLINE OTK_HOSTDEVICE int alias( const AliasTable& at, float x )
{
    x = clampUnderOne( x );
    int idx = static_cast<int>( x * at.size );
    float p = ( x * at.size ) - idx;
    if( p <= at.table[idx].prob )
        return idx;
    return at.table[idx].alias;
}

OTK_INLINE OTK_HOSTDEVICE int alias( const AliasTable& at, unsigned int x )
{
    const float twoNeg32 = scalbnf( 1.0f, -32 );
    unsigned idx = static_cast<int>( x * ( at.size * twoNeg32 ) );
    float p = ( x * x ) * twoNeg32; // mix up bits of x by a multiplication
    if( p <= at.table[idx].prob )
        return idx;
    return at.table[idx].alias;
}

OTK_INLINE OTK_HOSTDEVICE float2 alias2D( const AliasTable& at, int width, int height, float2 xi )
{
    xi = float2{clampUnderOne( xi.x ), clampUnderOne( xi.y )};
    const float mix = 102.31435f; // random constant to mix bits of p
    int idx = static_cast<int>( xi.y * height ) * width + static_cast<int>( xi.x * width );
    float p = frac( mix * ( xi.x + 1.0f ) * ( xi.y + 1.0f ) );
    if( p < at.table[idx].prob )
        return xi;

    // Get the coordinates of element idx
    idx = at.table[idx].alias;
    int j = int( float( idx ) / width );
    int i = idx - j * width;

    // Make random offsets from the bits of p
    float dx = frac( mix * p );
    float dy = frac( mix * dx );
    return float2{( dx + i ) / width, ( dy + j ) / height};
}

