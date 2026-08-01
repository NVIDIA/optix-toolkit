// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Params.h"

#include <cuda.h>

#include <cmath>

namespace demandPbrtScene {

#if defined( __CUDACC__ )
#define DEMAND_PBRT_SCENE_FOURIER_HD __host__ __device__ __forceinline__
#else
#define DEMAND_PBRT_SCENE_FOURIER_HD inline
#endif

constexpr int FOURIER_BSDF_EVAL_MAX_ORDER{ 512 };
constexpr int FOURIER_BSDF_EVAL_MAX_CHANNELS{ 3 };

enum class FourierBsdfTransportMode
{
    RADIANCE,
    IMPORTANCE,
};

struct FourierBsdfEvalResult
{
    float3 value;
    float  pdf;
};

struct FourierBsdfInterpolation
{
    int   offsetI;
    int   offsetO;
    float weightsI[4];
    float weightsO[4];
};

struct FourierBsdfCoefficientScratch
{
    float coefficients[FOURIER_BSDF_EVAL_MAX_CHANNELS * FOURIER_BSDF_EVAL_MAX_ORDER];
};

DEMAND_PBRT_SCENE_FOURIER_HD float fourierAbs( float value )
{
    return value < 0.0f ? -value : value;
}

DEMAND_PBRT_SCENE_FOURIER_HD float fourierMax( float lhs, float rhs )
{
    return lhs > rhs ? lhs : rhs;
}

DEMAND_PBRT_SCENE_FOURIER_HD float fourierClamp( float value, float low, float high )
{
    if( value < low )
        return low;
    if( value > high )
        return high;
    return value;
}

DEMAND_PBRT_SCENE_FOURIER_HD float fourierSqrt( float value )
{
#if defined( __CUDA_ARCH__ )
    return sqrtf( value );
#else
    return std::sqrt( value );
#endif
}

DEMAND_PBRT_SCENE_FOURIER_HD const float* fourierFloatData( CUdeviceptr ptr )
{
    return reinterpret_cast<const float*>( ptr );
}

DEMAND_PBRT_SCENE_FOURIER_HD const int* fourierIntData( CUdeviceptr ptr )
{
    return reinterpret_cast<const int*>( ptr );
}

DEMAND_PBRT_SCENE_FOURIER_HD int fourierFindInterval( int size, const float* nodes, float x )
{
    int first = 0;
    int len   = size;
    while( len > 0 )
    {
        const int half   = len >> 1;
        const int middle = first + half;
        if( nodes[middle] <= x )
        {
            first = middle + 1;
            len -= half + 1;
        }
        else
        {
            len = half;
        }
    }

    const int value = first - 1;
    if( value < 0 )
    {
        return 0;
    }
    const int maxValue = size - 2;
    return value > maxValue ? maxValue : value;
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierCatmullRomWeights( int size, const float* nodes, float x, int& offset, float weights[4] )
{
    if( size < 2 || nodes == nullptr || !( x >= nodes[0] && x <= nodes[size - 1] ) )
    {
        return false;
    }

    const int idx = fourierFindInterval( size, nodes, x );
    offset        = idx - 1;
    const float x0{ nodes[idx] };
    const float x1{ nodes[idx + 1] };
    const float width{ x1 - x0 };
    if( width == 0.0f )
    {
        return false;
    }

    const float t{ ( x - x0 ) / width };
    const float t2{ t * t };
    const float t3{ t2 * t };

    weights[1] = 2.0f * t3 - 3.0f * t2 + 1.0f;
    weights[2] = -2.0f * t3 + 3.0f * t2;

    if( idx > 0 )
    {
        const float w0{ ( t3 - 2.0f * t2 + t ) * width / ( x1 - nodes[idx - 1] ) };
        weights[0] = -w0;
        weights[2] += w0;
    }
    else
    {
        const float w0{ t3 - 2.0f * t2 + t };
        weights[0] = 0.0f;
        weights[1] -= w0;
        weights[2] += w0;
    }

    if( idx + 2 < size )
    {
        const float w3{ ( t3 - t2 ) * width / ( nodes[idx + 2] - x0 ) };
        weights[1] -= w3;
        weights[3] = w3;
    }
    else
    {
        const float w3{ t3 - t2 };
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0.0f;
    }
    return true;
}

DEMAND_PBRT_SCENE_FOURIER_HD float fourierSeries( const float* coefficients, int order, float cosPhi )
{
    double       value     = 0.0;
    double       cosKm1Phi = static_cast<double>( cosPhi );
    double       cosKPhi   = 1.0;
    const double cosPhiD   = static_cast<double>( cosPhi );
    for( int k = 0; k < order; ++k )
    {
        value += static_cast<double>( coefficients[k] ) * cosKPhi;
        const double cosKp1Phi{ 2.0 * cosPhiD * cosKPhi - cosKm1Phi };
        cosKm1Phi = cosKPhi;
        cosKPhi   = cosKp1Phi;
    }
    return static_cast<float>( value );
}

DEMAND_PBRT_SCENE_FOURIER_HD float fourierCosDPhi( const float3& wa, const float3& wb )
{
    const float waXY{ wa.x * wa.x + wa.y * wa.y };
    const float wbXY{ wb.x * wb.x + wb.y * wb.y };
    if( waXY == 0.0f || wbXY == 0.0f )
    {
        return 1.0f;
    }
    return fourierClamp( ( wa.x * wb.x + wa.y * wb.y ) / fourierSqrt( waXY * wbXY ), -1.0f, 1.0f );
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierHasRequiredData( const FourierBsdfTableDeviceData& table )
{
    return table.nMu >= 2 && table.maxOrder > 0 && table.maxOrder <= FOURIER_BSDF_EVAL_MAX_ORDER
           && ( table.nChannels == 1 || table.nChannels == 3 ) && table.mu != CUdeviceptr{}
           && table.cdf != CUdeviceptr{} && table.coefficientOffsets != CUdeviceptr{}
           && table.coefficientCounts != CUdeviceptr{} && table.coefficients != CUdeviceptr{};
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierInterpolation( const FourierBsdfTableDeviceData& table, float muI, float muO, FourierBsdfInterpolation& interpolation )
{
    const float* mu{ fourierFloatData( table.mu ) };
    return fourierCatmullRomWeights( table.nMu, mu, muI, interpolation.offsetI, interpolation.weightsI )
           && fourierCatmullRomWeights( table.nMu, mu, muO, interpolation.offsetO, interpolation.weightsO );
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierAccumulateCoefficients( const FourierBsdfTableDeviceData& table,
                                                                 const FourierBsdfInterpolation&   interpolation,
                                                                 FourierBsdfCoefficientScratch&    scratch,
                                                                 int&                              order )
{
    for( int i = 0; i < FOURIER_BSDF_EVAL_MAX_CHANNELS * FOURIER_BSDF_EVAL_MAX_ORDER; ++i )
    {
        scratch.coefficients[i] = 0.0f;
    }

    const int*   offsets{ fourierIntData( table.coefficientOffsets ) };
    const int*   counts{ fourierIntData( table.coefficientCounts ) };
    const float* coefficients{ fourierFloatData( table.coefficients ) };

    order = 0;
    for( int o = 0; o < 4; ++o )
    {
        for( int i = 0; i < 4; ++i )
        {
            const float weight{ interpolation.weightsI[i] * interpolation.weightsO[o] };
            if( weight == 0.0f )
            {
                continue;
            }

            const int offsetI{ interpolation.offsetI + i };
            const int offsetO{ interpolation.offsetO + o };
            if( offsetI < 0 || offsetI >= table.nMu || offsetO < 0 || offsetO >= table.nMu )
            {
                return false;
            }

            const int entry{ offsetO * table.nMu + offsetI };
            if( entry < 0 || static_cast<uint_t>( entry ) >= table.gridSize )
            {
                return false;
            }

            const int coefficientOffset{ offsets[entry] };
            const int entryOrder{ counts[entry] };
            if( coefficientOffset < 0 || entryOrder < 0 || entryOrder > table.maxOrder
                || coefficientOffset + entryOrder * table.nChannels > table.nCoefficients )
            {
                return false;
            }

            order = entryOrder > order ? entryOrder : order;
            for( int channel = 0; channel < table.nChannels; ++channel )
            {
                const int sourceBase{ coefficientOffset + channel * entryOrder };
                const int targetBase{ channel * table.maxOrder };
                for( int k = 0; k < entryOrder; ++k )
                {
                    scratch.coefficients[targetBase + k] += weight * coefficients[sourceBase + k];
                }
            }
        }
    }
    return true;
}

DEMAND_PBRT_SCENE_FOURIER_HD float3 fourierEvaluateRgb( const FourierBsdfTableDeviceData&    table,
                                                        float                                muI,
                                                        float                                muO,
                                                        float                                cosPhi,
                                                        FourierBsdfTransportMode             mode,
                                                        const FourierBsdfCoefficientScratch& scratch,
                                                        int                                  order )
{
    const float luminance{ fourierSeries( scratch.coefficients, order, cosPhi ) };
    const float y{ fourierMax( 0.0f, luminance ) };
    float       scale{ muI != 0.0f ? 1.0f / fourierAbs( muI ) : 0.0f };
    if( mode == FourierBsdfTransportMode::RADIANCE && muI * muO > 0.0f )
    {
        const float eta{ muI > 0.0f ? 1.0f / table.eta : table.eta };
        scale *= eta * eta;
    }

    if( table.nChannels == 1 )
    {
        const float monochrome{ y * scale };
        return make_float3( monochrome, monochrome, monochrome );
    }

    const float r{ fourierSeries( scratch.coefficients + table.maxOrder, order, cosPhi ) };
    const float b{ fourierSeries( scratch.coefficients + 2 * table.maxOrder, order, cosPhi ) };
    const float g{ 1.39829f * y - 0.100913f * b - 0.297375f * r };
    return make_float3( fourierMax( 0.0f, r * scale ), fourierMax( 0.0f, g * scale ), fourierMax( 0.0f, b * scale ) );
}

DEMAND_PBRT_SCENE_FOURIER_HD float fourierPdf( const FourierBsdfTableDeviceData&    table,
                                               const FourierBsdfInterpolation&      interpolation,
                                               const FourierBsdfCoefficientScratch& scratch,
                                               int                                  order,
                                               float                                cosPhi )
{
    constexpr float TWO_PI{ 6.2831853071795864769f };
    const float*    cdf{ fourierFloatData( table.cdf ) };

    float rho{ 0.0f };
    for( int o = 0; o < 4; ++o )
    {
        if( interpolation.weightsO[o] == 0.0f )
        {
            continue;
        }

        const int offsetO{ interpolation.offsetO + o };
        if( offsetO < 0 || offsetO >= table.nMu )
        {
            continue;
        }
        rho += interpolation.weightsO[o] * cdf[offsetO * table.nMu + table.nMu - 1] * TWO_PI;
    }

    const float y{ fourierSeries( scratch.coefficients, order, cosPhi ) };
    return rho > 0.0f && y > 0.0f ? y / rho : 0.0f;
}

DEMAND_PBRT_SCENE_FOURIER_HD FourierBsdfEvalResult evaluateFourierBsdfTable( const FourierBsdfTableDeviceData& table,
                                                                             const float3&                     outgoing,
                                                                             const float3&                     incoming,
                                                                             FourierBsdfTransportMode          mode )
{
    FourierBsdfEvalResult result{ make_float3( 0.0f, 0.0f, 0.0f ), 0.0f };
    if( !fourierHasRequiredData( table ) )
    {
        return result;
    }

    const float3 minusIncoming{ make_float3( -incoming.x, -incoming.y, -incoming.z ) };
    const float  muI{ minusIncoming.z };
    const float  muO{ outgoing.z };
    const float  cosPhi{ fourierCosDPhi( minusIncoming, outgoing ) };

    FourierBsdfInterpolation interpolation{};
    if( !fourierInterpolation( table, muI, muO, interpolation ) )
    {
        return result;
    }

    FourierBsdfCoefficientScratch scratch{};
    int                           order{};
    if( !fourierAccumulateCoefficients( table, interpolation, scratch, order ) )
    {
        return result;
    }

    result.value = fourierEvaluateRgb( table, muI, muO, cosPhi, mode, scratch, order );
    result.pdf   = fourierPdf( table, interpolation, scratch, order, cosPhi );
    return result;
}

DEMAND_PBRT_SCENE_FOURIER_HD FourierBsdfEvalResult evaluateFourierBsdf( const FourierMaterialResource& resource,
                                                                        const float3&                  outgoing,
                                                                        const float3&                  incoming,
                                                                        FourierBsdfTransportMode       mode )
{
    if( !hasFourierBsdfTableResource( resource ) )
    {
        return FourierBsdfEvalResult{ make_float3( 0.0f, 0.0f, 0.0f ), 0.0f };
    }

    const FourierBsdfTableDeviceData* table{ reinterpret_cast<const FourierBsdfTableDeviceData*>( resource.table ) };
    return evaluateFourierBsdfTable( *table, outgoing, incoming, mode );
}

#undef DEMAND_PBRT_SCENE_FOURIER_HD

}  // namespace demandPbrtScene
