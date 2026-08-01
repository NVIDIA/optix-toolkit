// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/FourierBsdfEval.h>
#include <DemandPbrtScene/FourierBsdfTableResource.h>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <vector>

using namespace demandPbrtScene;

namespace {

std::filesystem::path pbrtReferenceDir()
{
    return std::filesystem::path{ DEMAND_PBRT_SCENE_TEST_SOURCE_DIR } / "pbrt-reference";
}

template <typename T>
CUdeviceptr hostPtr( const std::vector<T>& values )
{
    return reinterpret_cast<CUdeviceptr>( values.data() );
}

FourierBsdfTableDeviceData makeHostDescriptor( const FourierBsdfTable& table )
{
    return makeFourierBsdfTableDeviceData( table, hostPtr( table.mu ), hostPtr( table.cdf ),
                                           hostPtr( table.coefficientOffsets ), hostPtr( table.coefficientCounts ),
                                           hostPtr( table.zeroOrderCoefficients ), hostPtr( table.coefficients ) );
}

FourierBsdfTable makeCoatedCopperOrderShapeTable()
{
    FourierBsdfTable table{};
    table.flags     = 1;
    table.nMu       = 2;
    table.maxOrder  = FOURIER_BSDF_EVAL_MAX_ORDER;
    table.nChannels = 3;
    table.nBases    = 1;
    table.eta       = 1.0f;
    table.mu        = { -1.0f, 1.0f };
    table.cdf       = { 0.0f, 1.0f, 0.0f, 1.0f };

    const std::size_t gridSize{ table.mu.size() * table.mu.size() };
    table.nCoefficients =
        static_cast<int>( gridSize * static_cast<std::size_t>( table.nChannels ) * static_cast<std::size_t>( table.maxOrder ) );
    table.coefficientOffsets.resize( gridSize );
    table.coefficientCounts.assign( gridSize, table.maxOrder );
    table.zeroOrderCoefficients.assign( gridSize, 1.0f );
    table.coefficients.assign( static_cast<std::size_t>( table.nCoefficients ), 0.0f );
    for( std::size_t entry = 0; entry < gridSize; ++entry )
    {
        const int offset{ static_cast<int>( entry ) * table.nChannels * table.maxOrder };
        table.coefficientOffsets[entry]                                             = offset;
        table.coefficients[static_cast<std::size_t>( offset )]                      = 1.0f;
        table.coefficients[static_cast<std::size_t>( offset + table.maxOrder - 1 )] = 0.01f;
        table.coefficients[static_cast<std::size_t>( offset + table.maxOrder )]     = 0.8f;
        table.coefficients[static_cast<std::size_t>( offset + 2 * table.maxOrder )] = 0.6f;
    }
    return table;
}

FourierMaterialResource makeFourierResource( const FourierBsdfTableDeviceData& table )
{
    return makeFourierMaterialResource( 1U, reinterpret_cast<CUdeviceptr>( &table ) );
}

float3 normalize( const float3& value )
{
    const float length{ std::sqrt( value.x * value.x + value.y * value.y + value.z * value.z ) };
    return make_float3( value.x / length, value.y / length, value.z / length );
}

float pbrtY( const float3& rgb )
{
    return 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z;
}

float relativeError( float value, float reference )
{
    return std::abs( ( value - reference ) / reference );
}

bool isFinite( const float3& value )
{
    return std::isfinite( value.x ) && std::isfinite( value.y ) && std::isfinite( value.z );
}

}  // namespace

TEST( TestFourierBsdfEval, returnsBlackWhenResourceHasNoTable )
{
    const FourierMaterialResource resource{};
    const FourierBsdfEvalResult   result{ evaluateFourierBsdf(
        resource, make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 0.0f, -1.0f ), FourierBsdfTransportMode::IMPORTANCE ) };

    EXPECT_FLOAT_EQ( 0.0f, result.value.x );
    EXPECT_FLOAT_EQ( 0.0f, result.value.y );
    EXPECT_FLOAT_EQ( 0.0f, result.value.z );
    EXPECT_FLOAT_EQ( 0.0f, result.pdf );
}

TEST( TestFourierBsdfEval, evaluatesAndSamplesCoatedCopperOrderShape )
{
    const FourierBsdfTable           tableStorage{ makeCoatedCopperOrderShapeTable() };
    const FourierBsdfTableDeviceData table{ makeHostDescriptor( tableStorage ) };
    const FourierMaterialResource    resource{ makeFourierResource( table ) };

    FourierBsdfInterpolation interpolation{};
    ASSERT_TRUE( fourierInterpolation( table, 1.0f, 1.0f, interpolation ) );
    FourierBsdfCoefficientScratch scratch{};
    int                           order{};
    EXPECT_EQ( static_cast<std::size_t>( FOURIER_BSDF_EVAL_MAX_ORDER ),
               sizeof( scratch.coefficients ) / sizeof( scratch.coefficients[0] ) );
    ASSERT_TRUE( fourierAccumulateCoefficients( table, interpolation, scratch, order ) );
    EXPECT_EQ( 530, order );
    EXPECT_FLOAT_EQ( 0.01f, scratch.coefficients[529] );
    ASSERT_TRUE( fourierAccumulateCoefficients( table, interpolation, scratch, order, 1 ) );
    EXPECT_EQ( 530, order );
    EXPECT_FLOAT_EQ( 0.8f, scratch.coefficients[0] );
    ASSERT_TRUE( fourierAccumulateCoefficients( table, interpolation, scratch, order, 2 ) );
    EXPECT_EQ( 530, order );
    EXPECT_FLOAT_EQ( 0.6f, scratch.coefficients[0] );

    const FourierBsdfEvalResult eval{ evaluateFourierBsdf( resource, make_float3( 0.0f, 0.0f, 1.0f ),
                                                           make_float3( 0.0f, 0.0f, -1.0f ), FourierBsdfTransportMode::IMPORTANCE ) };
    EXPECT_TRUE( isFinite( eval.value ) );
    EXPECT_GT( pbrtY( eval.value ), 0.0f );
    EXPECT_GT( eval.pdf, 0.0f );

    const FourierBsdfSampleResult sample{ sampleFourierBsdf( resource, make_float3( 0.0f, 0.0f, 1.0f ),
                                                             make_float2( 0.25f, 0.75f ), FourierBsdfTransportMode::IMPORTANCE ) };
    ASSERT_TRUE( sample.valid );
    EXPECT_TRUE( isFinite( sample.value ) );
    EXPECT_TRUE( isFinite( sample.direction ) );
    EXPECT_TRUE( isFinite( sample.throughput ) );
    EXPECT_GT( sample.pdf, 0.0f );
    EXPECT_GT( pbrtY( sample.throughput ), 0.0f );
}

TEST( TestFourierBsdfEval, matchesPbrtRoughGoldEvaluateAndPdfSamples )
{
    const std::filesystem::path      fixture{ pbrtReferenceDir() / "bsdfs" / "roughgold_alpha_0.2.bsdf" };
    const FourierBsdfTableLoadResult tableResult{ loadFourierBsdfTable( fixture.string() ) };
    ASSERT_TRUE( tableResult ) << tableResult.diagnostic;

    const FourierBsdfTableDeviceData table{ makeHostDescriptor( tableResult.table ) };
    const FourierMaterialResource    resource{ makeFourierResource( table ) };
    const float3                     outgoing{ normalize( make_float3( -0.5f, -0.5f, 0.8f ) ) };
    const float3                     incoming{ normalize( make_float3( 0.4f, 0.52f, 0.7f ) ) };

    const FourierBsdfEvalResult result{ evaluateFourierBsdf( resource, outgoing, incoming, FourierBsdfTransportMode::IMPORTANCE ) };
    const FourierBsdfEvalResult reverse{ evaluateFourierBsdf( resource, incoming, outgoing, FourierBsdfTransportMode::IMPORTANCE ) };

    EXPECT_LT( relativeError( pbrtY( result.value ), 2.679294f ), 0.001f );
    EXPECT_LT( relativeError( result.pdf, 2.438230f ), 0.001f );
    EXPECT_LT( relativeError( reverse.pdf, 2.503326f ), 0.001f );
}

TEST( TestFourierBsdfEval, matchesPbrtRoughGoldSample )
{
    const std::filesystem::path      fixture{ pbrtReferenceDir() / "bsdfs" / "roughgold_alpha_0.2.bsdf" };
    const FourierBsdfTableLoadResult tableResult{ loadFourierBsdfTable( fixture.string() ) };
    ASSERT_TRUE( tableResult ) << tableResult.diagnostic;

    const FourierBsdfTableDeviceData table{ makeHostDescriptor( tableResult.table ) };
    const FourierMaterialResource    resource{ makeFourierResource( table ) };
    const float3                     outgoing{ normalize( make_float3( -0.5f, -0.5f, 0.8f ) ) };

    const FourierBsdfSampleResult sample{
        sampleFourierBsdf( resource, outgoing, make_float2( 0.1f, 0.8f ), FourierBsdfTransportMode::IMPORTANCE ) };

    ASSERT_TRUE( sample.valid );
    EXPECT_LT( relativeError( pbrtY( sample.value ), 2.596391f ), 0.001f );
    EXPECT_LT( relativeError( sample.pdf, 1.855472f ), 0.001f );
    EXPECT_LT( relativeError( sample.direction.x, 0.539052f ), 0.001f );
    EXPECT_LT( relativeError( sample.direction.y, 0.617347f ), 0.001f );
    EXPECT_LT( relativeError( sample.direction.z, 0.572980f ), 0.001f );
    EXPECT_NEAR( pbrtY( sample.value ) * std::abs( sample.direction.z ) / sample.pdf, pbrtY( sample.throughput ), 1.0e-5f );
}

TEST( TestFourierBsdfEval, evaluatesPbrtRoughGoldAtNormalIncidence )
{
    const std::filesystem::path      fixture{ pbrtReferenceDir() / "bsdfs" / "roughgold_alpha_0.2.bsdf" };
    const FourierBsdfTableLoadResult tableResult{ loadFourierBsdfTable( fixture.string() ) };
    ASSERT_TRUE( tableResult ) << tableResult.diagnostic;

    const FourierBsdfTableDeviceData table{ makeHostDescriptor( tableResult.table ) };
    const FourierMaterialResource    resource{ makeFourierResource( table ) };

    const FourierBsdfEvalResult result{ evaluateFourierBsdf( resource, make_float3( 0.0f, 0.0f, 1.0f ),
                                                             make_float3( 0.0f, 0.0f, 1.0f ), FourierBsdfTransportMode::RADIANCE ) };

    EXPECT_GT( pbrtY( result.value ), 0.0f );
    EXPECT_GT( result.pdf, 0.0f );
}

TEST( TestFourierBsdfEval, samplesPbrtRoughGoldAtNormalIncidence )
{
    const std::filesystem::path      fixture{ pbrtReferenceDir() / "bsdfs" / "roughgold_alpha_0.2.bsdf" };
    const FourierBsdfTableLoadResult tableResult{ loadFourierBsdfTable( fixture.string() ) };
    ASSERT_TRUE( tableResult ) << tableResult.diagnostic;

    const FourierBsdfTableDeviceData table{ makeHostDescriptor( tableResult.table ) };
    const FourierMaterialResource    resource{ makeFourierResource( table ) };
    const float3                     outgoing{ make_float3( 0.0f, 0.0f, 1.0f ) };

    const FourierBsdfSampleResult sample{
        sampleFourierBsdf( resource, outgoing, make_float2( 0.1f, 0.8f ), FourierBsdfTransportMode::RADIANCE ) };

    ASSERT_TRUE( sample.valid );
    EXPECT_GT( sample.pdf, 0.0f );
    EXPECT_GT( pbrtY( sample.throughput ), 0.0f );
    EXPECT_NEAR( 1.0f,
                 std::sqrt( sample.direction.x * sample.direction.x + sample.direction.y * sample.direction.y
                            + sample.direction.z * sample.direction.z ),
                 1.0e-5f );
}
