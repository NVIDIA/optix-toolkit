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

constexpr int FOURIER_BSDF_EVAL_MAX_ORDER{ 530 };
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

struct FourierBsdfSampleResult
{
    bool   valid;
    float3 value;
    float  pdf;
    float3 direction;
    float3 throughput;
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

DEMAND_PBRT_SCENE_FOURIER_HD float fourierCos( float value )
{
#if defined( __CUDA_ARCH__ )
    return cosf( value );
#else
    return std::cos( value );
#endif
}

DEMAND_PBRT_SCENE_FOURIER_HD float fourierSin( float value )
{
#if defined( __CUDA_ARCH__ )
    return sinf( value );
#else
    return std::sin( value );
#endif
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierIsFinite( float value )
{
#if defined( __CUDA_ARCH__ )
    return isfinite( value ) != 0;
#else
    return std::isfinite( value );
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
    float       value     = 0.0f;
    float       cosKm1Phi = cosPhi;
    float       cosKPhi   = 1.0f;
    const float cosPhiD   = cosPhi;
    for( int k = 0; k < order; ++k )
    {
        value += coefficients[k] * cosKPhi;
        const float cosKp1Phi{ 2.0f * cosPhiD * cosKPhi - cosKm1Phi };
        cosKm1Phi = cosKPhi;
        cosKPhi   = cosKp1Phi;
    }
    return value;
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

DEMAND_PBRT_SCENE_FOURIER_HD float fourierLength( const float3& value )
{
    return fourierSqrt( value.x * value.x + value.y * value.y + value.z * value.z );
}

DEMAND_PBRT_SCENE_FOURIER_HD float3 fourierNormalize( const float3& value )
{
    const float length{ fourierLength( value ) };
    if( length == 0.0f )
    {
        return make_float3( 0.0f, 0.0f, 0.0f );
    }
    return make_float3( value.x / length, value.y / length, value.z / length );
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierHasRequiredData( const FourierBsdfTableDeviceData& table )
{
    return table.nMu >= 2 && table.maxOrder > 0 && table.maxOrder <= FOURIER_BSDF_EVAL_MAX_ORDER
           && ( table.nChannels == 1 || table.nChannels == 3 ) && table.mu != CUdeviceptr{}
           && table.cdf != CUdeviceptr{} && table.coefficientOffsets != CUdeviceptr{}
           && table.coefficientCounts != CUdeviceptr{} && table.coefficients != CUdeviceptr{};
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierHasSamplingData( const FourierBsdfTableDeviceData& table )
{
    return fourierHasRequiredData( table ) && table.zeroOrderCoefficients != CUdeviceptr{};
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

DEMAND_PBRT_SCENE_FOURIER_HD float fourierInterpolateTableRow( const float* array, int size1, int size2, int offset, const float weights[4], int idx )
{
    float value{ 0.0f };
    for( int i = 0; i < 4; ++i )
    {
        if( weights[i] == 0.0f )
        {
            continue;
        }

        const int row{ offset + i };
        if( row < 0 || row >= size1 )
        {
            continue;
        }
        value += array[row * size2 + idx] * weights[i];
    }
    return value;
}

DEMAND_PBRT_SCENE_FOURIER_HD int fourierFindCatmullRom2DInterval( const float* cdf, int size1, int size2, int offset, const float weights[4], float u )
{
    int first = 0;
    int len   = size2;
    while( len > 0 )
    {
        const int half   = len >> 1;
        const int middle = first + half;
        if( fourierInterpolateTableRow( cdf, size1, size2, offset, weights, middle ) <= u )
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
    const int maxValue = size2 - 2;
    return value > maxValue ? maxValue : value;
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierSampleCatmullRom2D( const FourierBsdfTableDeviceData& table, float alpha, float u, float& sample, float& pdf )
{
    int          offset{};
    float        weights[4]{};
    const float* mu{ fourierFloatData( table.mu ) };
    if( !fourierCatmullRomWeights( table.nMu, mu, alpha, offset, weights ) )
    {
        return false;
    }

    const float* values{ fourierFloatData( table.zeroOrderCoefficients ) };
    const float* cdf{ fourierFloatData( table.cdf ) };
    const float  maximum{ fourierInterpolateTableRow( cdf, table.nMu, table.nMu, offset, weights, table.nMu - 1 ) };
    if( maximum <= 0.0f )
    {
        return false;
    }

    u *= maximum;
    const int   idx{ fourierFindCatmullRom2DInterval( cdf, table.nMu, table.nMu, offset, weights, u ) };
    const float f0{ fourierInterpolateTableRow( values, table.nMu, table.nMu, offset, weights, idx ) };
    const float f1{ fourierInterpolateTableRow( values, table.nMu, table.nMu, offset, weights, idx + 1 ) };
    const float x0{ mu[idx] };
    const float x1{ mu[idx + 1] };
    const float width{ x1 - x0 };
    if( width == 0.0f )
    {
        return false;
    }

    u = ( u - fourierInterpolateTableRow( cdf, table.nMu, table.nMu, offset, weights, idx ) ) / width;

    const float d0{ idx > 0 ? width * ( f1 - fourierInterpolateTableRow( values, table.nMu, table.nMu, offset, weights, idx - 1 ) )
                                  / ( x1 - mu[idx - 1] ) :
                              f1 - f0 };
    const float d1{ idx + 2 < table.nMu ?
                        width * ( fourierInterpolateTableRow( values, table.nMu, table.nMu, offset, weights, idx + 2 ) - f0 )
                            / ( mu[idx + 2] - x0 ) :
                        f1 - f0 };

    float t{};
    if( f0 != f1 )
    {
        t = ( f0 - fourierSqrt( fourierMax( 0.0f, f0 * f0 + 2.0f * u * ( f1 - f0 ) ) ) ) / ( f0 - f1 );
    }
    else if( f0 != 0.0f )
    {
        t = u / f0;
    }
    else
    {
        return false;
    }

    float a{ 0.0f };
    float b{ 1.0f };
    float fhat{};
    for( int iter = 0; iter < 32; ++iter )
    {
        if( !( t >= a && t <= b ) )
        {
            t = 0.5f * ( a + b );
        }

        const float fhatIntegral =
            t * ( f0 + t * ( 0.5f * d0 + t * ( ( 1.0f / 3.0f ) * ( -2.0f * d0 - d1 ) + f1 - f0 + t * ( 0.25f * ( d0 + d1 ) + 0.5f * ( f0 - f1 ) ) ) ) );
        fhat = f0 + t * ( d0 + t * ( -2.0f * d0 - d1 + 3.0f * ( f1 - f0 ) + t * ( d0 + d1 + 2.0f * ( f0 - f1 ) ) ) );

        if( fourierAbs( fhatIntegral - u ) < 1.0e-6f || b - a < 1.0e-6f )
        {
            break;
        }

        if( fhatIntegral - u < 0.0f )
        {
            a = t;
        }
        else
        {
            b = t;
        }

        if( fhat != 0.0f )
        {
            t -= ( fhatIntegral - u ) / fhat;
        }
    }

    sample = x0 + width * t;
    pdf    = fhat / maximum;
    return pdf > 0.0f;
}

DEMAND_PBRT_SCENE_FOURIER_HD bool fourierSampleFourier( const float* coefficients, int order, float u, float& value, float& pdf, float& phi )
{
    constexpr float PI{ 3.14159265358979323846f };
    constexpr float TWO_PI{ 2.0f * PI };
    constexpr float INV_TWO_PI{ 1.0f / TWO_PI };
    if( coefficients == nullptr || order <= 0 || coefficients[0] <= 0.0f )
    {
        return false;
    }

    const bool flip{ u >= 0.5f };
    if( flip )
    {
        u = 1.0f - 2.0f * ( u - 0.5f );
    }
    else
    {
        u *= 2.0f;
    }

    float a{ 0.0f };
    float b{ PI };
    phi = 0.5f * PI;
    float f{};
    for( int iter = 0; iter < 32; ++iter )
    {
        const float cosPhi{ fourierCos( phi ) };
        const float sinPhi{ fourierSqrt( fourierMax( 0.0f, 1.0f - cosPhi * cosPhi ) ) };
        float       cosPhiPrev{ cosPhi };
        float       cosPhiCur{ 1.0f };
        float       sinPhiPrev{ -sinPhi };
        float       sinPhiCur{ 0.0f };

        float cdf{ coefficients[0] * phi };
        f = coefficients[0];
        for( int k = 1; k < order; ++k )
        {
            const float sinPhiNext{ 2.0f * cosPhi * sinPhiCur - sinPhiPrev };
            const float cosPhiNext{ 2.0f * cosPhi * cosPhiCur - cosPhiPrev };
            sinPhiPrev = sinPhiCur;
            sinPhiCur  = sinPhiNext;
            cosPhiPrev = cosPhiCur;
            cosPhiCur  = cosPhiNext;

            cdf += coefficients[k] * ( 1.0f / static_cast<float>( k ) ) * sinPhiNext;
            f += coefficients[k] * cosPhiNext;
        }
        cdf -= u * coefficients[0] * PI;

        if( cdf > 0.0f )
        {
            b = phi;
        }
        else
        {
            a = phi;
        }

        if( fourierAbs( cdf ) < 1.0e-6f || b - a < 1.0e-6f )
        {
            break;
        }

        if( f != 0.0f )
        {
            phi -= cdf / f;
        }
        if( !( phi > a && phi < b ) )
        {
            phi = 0.5f * ( a + b );
        }
    }

    if( flip )
    {
        phi = TWO_PI - phi;
    }
    value = f;
    pdf   = INV_TWO_PI * f / coefficients[0];
    return pdf > 0.0f;
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

DEMAND_PBRT_SCENE_FOURIER_HD float3 fourierSampleRgb( const FourierBsdfTableDeviceData&    table,
                                                      float                                muI,
                                                      float                                muO,
                                                      float                                cosPhi,
                                                      FourierBsdfTransportMode             mode,
                                                      const FourierBsdfCoefficientScratch& scratch,
                                                      int                                  order,
                                                      float                                y )
{
    float scale{ muI != 0.0f ? 1.0f / fourierAbs( muI ) : 0.0f };
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

DEMAND_PBRT_SCENE_FOURIER_HD FourierBsdfSampleResult sampleFourierBsdfTable( const FourierBsdfTableDeviceData& table,
                                                                             const float3&                     outgoing,
                                                                             const float2&                     u,
                                                                             FourierBsdfTransportMode          mode )
{
    FourierBsdfSampleResult result{ false, make_float3( 0.0f, 0.0f, 0.0f ), 0.0f, make_float3( 0.0f, 0.0f, 0.0f ),
                                    make_float3( 0.0f, 0.0f, 0.0f ) };
    if( !fourierHasSamplingData( table ) )
    {
        return result;
    }

    const float muO{ outgoing.z };
    float       pdfMu{};
    float       muI{};
    if( !fourierSampleCatmullRom2D( table, muO, u.y, muI, pdfMu ) )
    {
        return result;
    }

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

    float y{};
    float pdfPhi{};
    float phi{};
    if( !fourierSampleFourier( scratch.coefficients, order, u.x, y, pdfPhi, phi ) )
    {
        return result;
    }
    result.pdf = fourierMax( 0.0f, pdfPhi * pdfMu );

    const float sin2ThetaI{ fourierMax( 0.0f, 1.0f - muI * muI ) };
    const float sin2ThetaO{ fourierMax( 0.0f, 1.0f - outgoing.z * outgoing.z ) };
    float       norm{ sin2ThetaO != 0.0f ? fourierSqrt( sin2ThetaI / sin2ThetaO ) : 0.0f };
    if( !fourierIsFinite( norm ) )
    {
        norm = 0.0f;
    }

    const float sinPhi{ fourierSin( phi ) };
    const float cosPhi{ fourierCos( phi ) };
    result.direction = fourierNormalize( make_float3( -norm * ( cosPhi * outgoing.x - sinPhi * outgoing.y ),
                                                      -norm * ( sinPhi * outgoing.x + cosPhi * outgoing.y ), -muI ) );
    result.value     = fourierSampleRgb( table, muI, muO, cosPhi, mode, scratch, order, y );
    if( result.pdf > 0.0f )
    {
        const float cosineOverPdf{ fourierAbs( result.direction.z ) / result.pdf };
        result.throughput =
            make_float3( result.value.x * cosineOverPdf, result.value.y * cosineOverPdf, result.value.z * cosineOverPdf );
    }
    result.valid = result.pdf > 0.0f
                   && ( result.throughput.x * result.throughput.x + result.throughput.y * result.throughput.y
                        + result.throughput.z * result.throughput.z )
                          > 0.0f;
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

DEMAND_PBRT_SCENE_FOURIER_HD FourierBsdfSampleResult sampleFourierBsdf( const FourierMaterialResource& resource,
                                                                        const float3&                  outgoing,
                                                                        const float2&                  u,
                                                                        FourierBsdfTransportMode       mode )
{
    if( !hasFourierBsdfTableResource( resource ) )
    {
        return FourierBsdfSampleResult{ false, make_float3( 0.0f, 0.0f, 0.0f ), 0.0f, make_float3( 0.0f, 0.0f, 0.0f ),
                                        make_float3( 0.0f, 0.0f, 0.0f ) };
    }

    const FourierBsdfTableDeviceData* table{ reinterpret_cast<const FourierBsdfTableDeviceData*>( resource.table ) };
    return sampleFourierBsdfTable( *table, outgoing, u, mode );
}

#undef DEMAND_PBRT_SCENE_FOURIER_HD

}  // namespace demandPbrtScene
