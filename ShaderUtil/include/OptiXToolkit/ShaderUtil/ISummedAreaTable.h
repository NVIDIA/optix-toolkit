
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

/// \file ISummedAreaTable.h

#include <OptiXToolkit/ShaderUtil/Preprocessor.h>
#include <math.h>
#include <cuda_runtime.h>

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

/// A summed area table that uses unsigned ints, and also holds the
/// transpose of the table.
struct ISummedAreaTable
{
    int width;
    int height;
    unsigned int* table;
    unsigned int* transposeTable;
    
    unsigned int& val(int x, int y) { return table[y * width + x]; }
    unsigned int& tval(int x, int y) { return transposeTable[x * height + y]; }
};

/// Allocate the ISummedAreaTable on the host
inline void allocHost( ISummedAreaTable& sat, int width, int height )
{
    sat.width = width;
    sat.height = height;
    sat.table = reinterpret_cast<unsigned int*>( malloc( width * height * sizeof(unsigned int) ) );
    sat.transposeTable = reinterpret_cast<unsigned int*>( malloc( width * height * sizeof(unsigned int) ) );
}

/// Free the ISummedAreaTable on the host
inline void freeHost( ISummedAreaTable& sat )
{
    free( sat.table );
    free( sat.transposeTable );
}

/// Allocate the ISummedAreaTable on the device
inline cudaError_t allocDevice( ISummedAreaTable& sat, int width, int height )
{
    sat.width = width;
    sat.height = height;
    CUDA_CHECK( cudaMalloc( &sat.table, width * height * sizeof(unsigned int) ) );
    return cudaMalloc( &sat.transposeTable, width * height * sizeof(unsigned int) );
}

/// Free the ISummedAreaTable on the device
inline cudaError_t freeDevice( ISummedAreaTable& sat )
{
    CUDA_CHECK( cudaFree( sat.table ) );
    return cudaFree( sat.transposeTable );
}

/// Copy ISummedAreaTable data from the host to the device
inline cudaError_t copyToDevice( ISummedAreaTable& satHost, ISummedAreaTable& satDev )
{
    size_t tableSize = satHost.width * satHost.height * sizeof( unsigned int );
    CUDA_CHECK( cudaMemcpy( satDev.table, satHost.table, tableSize, cudaMemcpyHostToDevice ) );
    return cudaMemcpy( satDev.transposeTable, satHost.transposeTable, tableSize, cudaMemcpyHostToDevice );
}

// Utility function to transpose a ISummedAreaTable
inline void transposeTable( ISummedAreaTable& sat )
{
    // Iterate over blocks
    const int blockSize = 8;
    for( int ystart = 0; ystart < sat.height; ystart += blockSize )
    {
        for( int xstart = 0; xstart < sat.width; xstart += blockSize )
        {
            // Transpose a block
            int xend = ( sat.width < xstart + blockSize ) ? sat.width : xstart + blockSize;
            int yend = ( sat.height < ystart + blockSize ) ? sat.height : ystart + blockSize;
            for( int y = ystart; y < yend; ++y )
            {
                for( int x = xstart; x < xend; ++x )
                {
                    sat.tval(x, y) = sat.val(x, y);
                }
            }
        }
    }
}

/// Initialize a ISummedAreaTable for a pdf
inline void init( ISummedAreaTable& sat, float* pdf )
{
    double sum = 0.0;
    int tableEntries = sat.width * sat.height;
    for( int i = 0; i < tableEntries; ++i )
        sum += pdf[i];
    double scale = ( 0xffffffffU - tableEntries ) / sum;

    for( int j = 0; j < sat.height; ++j )
    {
        for( int i = 0; i < sat.width; ++i )
        {
            // Make sure each table entry has a positive value
            unsigned int val = 1 + static_cast<unsigned int>( pdf[ j * sat.width + i ] * scale );
            if( i > 0 ) val += sat.val( i - 1, j );
            if( i > 0 && j > 0 ) val -= sat.val( i - 1, j - 1 );
            if( j > 0 ) val += sat.val( i, j - 1 );
            sat.val( i, j ) = val;
        }
    }

    transposeTable( sat );
}

/// Get Rectangle Sum in a ISummedAreaTable
OTK_INLINE OTK_HOSTDEVICE unsigned int getRectSum( ISummedAreaTable& sat, int x0, int y0, int x1, int y1 )
{
    return ( sat.val(x1,y1) - sat.val(x1,y0) ) - ( sat.val(x0,y1) - sat.val(x0,y0) );
}

OTK_INLINE OTK_HOSTDEVICE unsigned int getValSafe( unsigned int* a, int i )
{
    return ( a != nullptr && i >= 0 ) ? a[i] : 0;
}

// Utility function  for sampleRect function
OTK_INLINE OTK_HOSTDEVICE float splitSearch( int x0, int x1, unsigned int a00, unsigned int a01,
                                             unsigned int* row0, unsigned int* row1, unsigned int targetSum )
{
    // Bin search
    while( x0 < x1-1 )
    {
        int mid = ( x0 + x1 ) >> 1;
        unsigned int midSum = (getValSafe(row1, mid) - getValSafe(row0, mid)) - (a01 - a00);
        if( midSum > targetSum )
            x1 = mid;
        else
            x0 = mid;
    }

    // Interpolate
    unsigned int sum0 = (getValSafe(row1, x0) - getValSafe(row0, x0)) - (a01 - a00);
    unsigned int sum1 = (getValSafe(row1, x1) - getValSafe(row0, x1)) - (a01 - a00);
    float frac = float(targetSum - sum0) / float(sum1 - sum0);
    return x0 + 1 + frac;
}

/// Compute a sample location within a rectangle in a ISummedAreaTable
OTK_INLINE OTK_HOSTDEVICE float2 sampleRect( ISummedAreaTable& sat, int x0, int y0, int x1, int y1, float u, float v )
{
    unsigned int a00 = ( x0 > 0 && y0 > 0 ) ? sat.val(x0-1, y0-1) : 0;
    unsigned int a01 = ( x0 > 0 ) ? sat.val(x0-1, y1) : 0;
    unsigned int a10 = ( y0 > 0 ) ? sat.val(x1, y0-1) : 0;
    unsigned int a11 = sat.val(x1, y1);
    unsigned int rectSum = (a11 - a10) - (a01 - a00);

    unsigned int* row0 = ( y0 > 0 ) ? &sat.val(0, y0-1) : nullptr;
    unsigned int* row1 = &sat.val(0, y1);
    float x = splitSearch( x0-1, x1, a00, a01, row0, row1, (unsigned int)(rectSum * u * 0.999999f) );

    unsigned int* col0 = ( x0 > 0 ) ? &sat.tval(x0-1, 0) : nullptr;
    unsigned int* col1 = &sat.tval(x1, 0);
    float y = splitSearch( y0-1, y1, a00, a10, col0, col1, (unsigned int)(rectSum * v * 0.999999f) );
    return float2{x / sat.width, y / sat.height};
}
