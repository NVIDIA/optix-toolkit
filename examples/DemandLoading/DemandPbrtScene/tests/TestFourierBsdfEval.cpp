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
