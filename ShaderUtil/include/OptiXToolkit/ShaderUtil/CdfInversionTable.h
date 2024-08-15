// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file CdfInversionTable.h
/// Functions to invert and sample an environment map or other textured light source.

#include <OptiXToolkit/ShaderUtil/Preprocessor.h>

#include <cuda_runtime.h>

#include <math.h>
#include <cstdlib>

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

#define NEAR_ONE 0.9999999f
#define NEAR_64K 65535.999f

/// Cdf and inversion arrays for a 2D image
struct CdfInversionTable
{
    int width;
    int height;
    float aveValue; // weighted average value in the pdf 
    float* cdfRows; // (width x height) initialized as an unnormalized pdf
    float* cdfMarginal; // (height)
    unsigned short* invCdfRows; // (width x height)
    unsigned short* invCdfMarginal; // (height)
};

/// Allocate the CdfInversionTable object on the device
inline void allocCdfInversionTableHost( CdfInversionTable& cit, int width, int height )
{
    cit.width = width;
    cit.height = height;
    cit.cdfRows = (float*) malloc( width * height * sizeof(float) );
    cit.cdfMarginal = (float*) malloc( height * sizeof(float) );
    cit.invCdfRows = (unsigned short*) malloc( width * height * sizeof(short) );
    cit.invCdfMarginal = (unsigned short*) malloc( height * sizeof(short) );
}

/// Free the CdfInversionTable object
inline void freeCdfInversionTableHost( CdfInversionTable& cit )
{
    free( cit.cdfRows );
    free( cit.cdfMarginal );
    free( cit.invCdfRows );
    free( cit.invCdfMarginal );
}

/// Allocate the CDF arrays on the device
inline cudaError_t allocCdfInversionTableDevice( CdfInversionTable& cit, int width, int height, bool allocCdf )
{
    cit.width = width;
    cit.height = height;

    cit.cdfRows = nullptr;
    cit.cdfMarginal = nullptr;
    if( allocCdf )
    {
        CUDA_CHECK( cudaMalloc(&cit.cdfRows, width * height * sizeof(float) ) );
        CUDA_CHECK( cudaMalloc(&cit.cdfMarginal, height * sizeof(float) ) );
    }

    CUDA_CHECK( cudaMalloc(&cit.invCdfRows, width * height * sizeof(short) ) );
    CUDA_CHECK( cudaMalloc(&cit.invCdfMarginal, height * sizeof(short) ) );
    return cudaSuccess;
}

/// Free the CDF arrays on the device
inline cudaError_t freeCdfInversionTableDevice( CdfInversionTable& cit )
{
    CUDA_CHECK( cudaFree( cit.cdfRows ) );
    CUDA_CHECK( cudaFree( cit.cdfMarginal ) );
    CUDA_CHECK( cudaFree( cit.invCdfRows ) );
    CUDA_CHECK( cudaFree( cit.invCdfMarginal ) );
    return cudaSuccess;
}

/// Copy CdfInversionTable data from the host to the device
inline cudaError_t copyToDevice( CdfInversionTable& citHost, CdfInversionTable& citDev )
{
    int w = citHost.width;
    int h = citHost.height;
    citDev.aveValue = citHost.aveValue;

    if( citDev.cdfRows != nullptr )
    {
        CUDA_CHECK( cudaMemcpy( citDev.cdfRows,     citHost.cdfRows,     w * h * sizeof(float), cudaMemcpyHostToDevice ) );
        CUDA_CHECK( cudaMemcpy( citDev.cdfMarginal, citHost.cdfMarginal, h * sizeof(float),     cudaMemcpyHostToDevice ) );
    }
    
    CUDA_CHECK( cudaMemcpy( citDev.invCdfRows,     citHost.invCdfRows,     w * h * sizeof(short), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( citDev.invCdfMarginal, citHost.invCdfMarginal, h * sizeof(short),     cudaMemcpyHostToDevice ) );
    return cudaSuccess;
}

/// Invert a 1D pdf converting it to a cdf
inline float invertPdf1D( float* pdf, int width )
{
    for( int i = 1; i < width; ++i )
    {
        pdf[i] += pdf[i-1];
    }

    float sum = pdf[width-1];
    float invSum = 1.0f / sum;
    for( int i = 0; i < width; ++i )
    {
        pdf[i] *= invSum;
    }
    return sum;
}

/// Invert a 2D pdf (stored in cit.cdfRows) to a 2D cdf, along with its marginal
inline void invertPdf2D( CdfInversionTable& cit )
{
    for( int j = 0; j < cit.height; ++j )
    {
        cit.cdfMarginal[j] = invertPdf1D( &cit.cdfRows[j * cit.width], cit.width );
    }
    invertPdf1D( cit.cdfMarginal, cit.height );
}

/// Invert a 1D cdf
inline void invertCdf1D( float* cdf, int cdfWidth, unsigned short* invCdf, int invCdfWidth )
{
    invCdf[0] = 0;
    int j = 0;
    float step = 1.0f / static_cast<float>( invCdfWidth - 1.0f );
    for( int i = 1; i < invCdfWidth; ++i )
    {
        float target = i * step;
        while( j < cdfWidth - 1 && cdf[j] <= target )
            j++;
        float a = cdf[j];
        float b = cdf[j + 1];
        if( a <= target && b > target )
        {
            float val = ( j + ( target - a ) / ( b - a ) ) * ( NEAR_64K / cdfWidth );
            invCdf[i] = static_cast<unsigned short>( val );
        }
        else 
        {
            invCdf[i] = static_cast<unsigned short>( j * NEAR_64K / cdfWidth );
        }
    }
}

/// Invert a 2D cdf, along with its marginal
inline void invertCdf2D( CdfInversionTable& cit )
{
    for( int j = 0; j < cit.height; ++j )
    {
        invertCdf1D( &cit.cdfRows[j * cit.width], cit.width, &cit.invCdfRows[j * cit.width], cit.width );
    }
    invertCdf1D( cit.cdfMarginal, cit.height, cit.invCdfMarginal, cit.height );
}

/// Sample a normalized 1D CDF by binary search
OTK_INLINE OTK_HOSTDEVICE float sampleCdfBinSearch( float* cdf, int width, float x )
{
    x = ( x < NEAR_ONE ) ? x : NEAR_ONE;
    int left = 0;
    int right = width-1;
    while( left <= right )
    {
        int mid = (left+right) >> 1;
        bool midBigger = ( cdf[mid] >= x );
        right = midBigger ? mid - 1 : right;
        left = midBigger ? left : mid + 1;
    }

    int index = left;
    float minVal = ( index > 0 ) ? cdf[index - 1] : 0.0f;
    float maxVal = ( index < width ) ? cdf[index] : NEAR_ONE;

    float rval = index + (x - minVal) / (maxVal - minVal);
    if( rval >= width )
        rval = width * NEAR_ONE;
    return rval;
}

/// Sample a CDF from a CdfInversionTable struct by binary search
OTK_INLINE OTK_HOSTDEVICE float2 sampleCdfBinSearch( CdfInversionTable& cit, float2 xi )
{
    const float y = sampleCdfBinSearch( cit.cdfMarginal, cit.height, xi.y );
    int row = static_cast<int>( y );
    const float x = sampleCdfBinSearch( &cit.cdfRows[row * cit.width], cit.width, xi.x );
    return float2{x / cit.width, y / cit.height};
}

/// Sample a normalized 1D CDF by linear search
OTK_INLINE OTK_HOSTDEVICE float sampleCdfLinSearch( float* cdf, unsigned short* invCdf, int width, float x )
{
    x = ( x < NEAR_ONE ) ? x : NEAR_ONE;
    int index = ( invCdf[ static_cast<int>( x * (width-1) ) ] * width ) >> 16;
    do
    {
        if( cdf[index] > x )
        {
            float minVal = ( index > 0 ) ? cdf[index - 1] : 0.0f;
            float maxVal = cdf[index];
            return index + (x - minVal) / (maxVal - minVal);
        }
        ++index;
    }  while ( index < width );

    return width * NEAR_ONE;
}

/// Sample a CDF from a CdfInversionTable by linear search
OTK_INLINE OTK_HOSTDEVICE float2 sampleCdfLinSearch( CdfInversionTable& cit, float2 xi )
{
    const float y = sampleCdfLinSearch( cit.cdfMarginal, cit.invCdfMarginal, cit.height, xi.y );
    int row = static_cast<int>( y );
    const float x = sampleCdfLinSearch( &cit.cdfRows[row * cit.width], &cit.invCdfRows[row * cit.width], cit.width, xi.x );
    return float2{x / cit.width, y / cit.height};
}

/// Sample a inverted CDF by direct lookup
OTK_INLINE OTK_HOSTDEVICE float sampleCdfDirectLookup( unsigned short* invCdf, int width, float x )
{
    x = ( x < NEAR_ONE ) ? x : NEAR_ONE;
    const int index = static_cast<int>( x * width );
    const float frac = x * width - index;

    const float a = invCdf[index];
    const float b = ( index + 1 < width ) ? invCdf[index + 1] : NEAR_64K;
    return ( ( 1.0f - frac ) * a + frac * b ) * ( 1.0f / 65536.0f );
}

/// Sample a CDF from a CdfInversionTable struct by direct lookup
OTK_INLINE OTK_HOSTDEVICE float2 sampleCdfDirectLookup( CdfInversionTable& cit, float2 xi )
{
    const float y = sampleCdfDirectLookup( cit.invCdfMarginal, cit.height, xi.y );
    const int row = static_cast<int>( y * cit.height );
    const float x = sampleCdfDirectLookup( &cit.invCdfRows[row * cit.width], cit.width, xi.x );
    return float2{x, y};
}
