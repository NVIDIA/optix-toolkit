// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/Testing/ParamsPrinters.h>

#include <DemandPbrtScene/Params.h>

#include <gtest/gtest.h>

using namespace demandPbrtScene;

TEST( TestDirectionalLightEquality, same )
{
    const DirectionalLight lhs{ make_float3( 1.0f, 2.0f, 3.0f ), make_float3( 4.0f, 5.0f, 6.0f ) };
    const DirectionalLight rhs{ make_float3( 1.0f, 2.0f, 3.0f ), make_float3( 4.0f, 5.0f, 6.0f ) };

    EXPECT_EQ( lhs, rhs );
}

TEST( TestDirectionalLightEquality, directionsDiffer )
{
    const DirectionalLight lhs{ make_float3( 0.0f, 1.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ) };
    const DirectionalLight rhs{ make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 1.0f, 1.0f, 1.0f ) };

    EXPECT_NE( lhs, rhs );
}

TEST( TestDirectionalLightEquality, colorsDiffer )
{
    const DirectionalLight lhs{ make_float3( 0.0f, 1.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ) };
    const DirectionalLight rhs{ make_float3( 0.0f, 1.0f, 1.0f ), make_float3( 1.0f, 0.0f, 1.0f ) };

    EXPECT_NE( lhs, rhs );
}

TEST( TestInfiniteLightEquality, same )
{
    const InfiniteLight lhs{ make_float3( 1.0f, 2.0f, 3.0f ) };
    const InfiniteLight rhs{ make_float3( 1.0f, 2.0f, 3.0f ) };

    EXPECT_EQ( lhs, rhs );
}

TEST( TestInfiniteLightEquality, colorsDiffer )
{
    const InfiniteLight lhs{ make_float3( 0.0f, 1.0f, 0.0f ) };
    const InfiniteLight rhs{ make_float3( 0.0f, 1.0f, 1.0f ) };

    EXPECT_NE( lhs, rhs );
}

TEST( TestInfiniteLightEquality, textureIdsDiffer )
{
    const InfiniteLight lhs{ make_float3( 1.0f, 2.0f, 3.0f ), 1234U };
    const InfiniteLight rhs{ make_float3( 1.0f, 2.0f, 3.0f ), 5678U };

    EXPECT_NE( lhs, rhs );
}

class TestParamPrinters : public ::testing::Test
{
  protected:
    std::ostringstream m_str;
};

TEST_F( TestParamPrinters, infiniteLight )
{
    const InfiniteLight val{ make_float3( 1.0f, 2.0f, 3.0f ), make_float3( 4.0f, 5.0f, 6.0f ), 1234U };

    m_str << val;

    EXPECT_EQ( "InfiniteLight{ color: (1, 2, 3), scale: (4, 5, 6), textureId: 1234 }", m_str.str() );
}

TEST( TestFallbackShaderContract, localFallbackDefaultsToNoMdlBackendReason )
{
    const MaterialState state{ makeMaterialState( 42U, MaterialBackend::LOCAL_FALLBACK ) };

    EXPECT_EQ( 42U, state.materialId );
    EXPECT_EQ( MaterialBackend::LOCAL_FALLBACK, state.backend );
    EXPECT_EQ( 0U, state.shaderKey );
    EXPECT_EQ( MaterialFallbackReason::NO_MDL_BACKEND, state.fallbackReason );
    EXPECT_TRUE( usesFallbackShader( state ) );
}

TEST( TestFallbackShaderContract, mdlReadyDoesNotUseFallback )
{
    const MaterialState state{ makeMaterialState( 42U, MaterialBackend::MDL_READY, 1234U ) };

    EXPECT_EQ( 42U, state.materialId );
    EXPECT_EQ( MaterialBackend::MDL_READY, state.backend );
    EXPECT_EQ( 1234U, state.shaderKey );
    EXPECT_EQ( MaterialFallbackReason::NONE, state.fallbackReason );
    EXPECT_FALSE( usesFallbackShader( state ) );
}

TEST( TestFallbackShaderContract, mdlPendingDefaultsToPendingReason )
{
    const MaterialState state{ makeMaterialState( 42U, MaterialBackend::MDL_PENDING, 1234U ) };

    EXPECT_EQ( MaterialFallbackReason::MDL_PENDING, state.fallbackReason );
    EXPECT_TRUE( usesFallbackShader( state ) );
}

TEST( TestFallbackShaderContract, mdlFailedDefaultsToFailedReason )
{
    const MaterialState state{ makeMaterialState( 42U, MaterialBackend::MDL_FAILED, 1234U ) };

    EXPECT_EQ( MaterialFallbackReason::MDL_FAILED, state.fallbackReason );
    EXPECT_TRUE( usesFallbackShader( state ) );
}

TEST( TestFallbackShaderContract, explicitReasonOverridesBackendDefault )
{
    const MaterialState state{ makeMaterialState( 42U, MaterialBackend::LOCAL_FALLBACK, 0U, MaterialFallbackReason::UNSUPPORTED ) };

    EXPECT_EQ( MaterialFallbackReason::UNSUPPORTED, state.fallbackReason );
    EXPECT_TRUE( usesFallbackShader( state ) );
}

TEST( TestFallbackShaderContract, defaultMaterialUsesConstantKdOnly )
{
    const PhongMaterial material{};

    const FallbackShaderFeature features{ fallbackFeaturesForMaterial( material ) };

    EXPECT_TRUE( flagSet( features, FallbackShaderFeature::CONSTANT_KD ) );
    EXPECT_FALSE( flagSet( features, FallbackShaderFeature::DIFFUSE_TEXTURE ) );
    EXPECT_FALSE( flagSet( features, FallbackShaderFeature::ALPHA_CUTOUT ) );
}

TEST( TestFallbackShaderContract, allocatedDiffuseAndAlphaMapsEnableTextureFeatures )
{
    PhongMaterial material{};
    material.flags = MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED | MaterialFlags::ALPHA_MAP
                     | MaterialFlags::ALPHA_MAP_ALLOCATED;

    const FallbackShaderFeature features{ fallbackFeaturesForMaterial( material ) };

    EXPECT_TRUE( flagSet( features, FallbackShaderFeature::CONSTANT_KD ) );
    EXPECT_TRUE( flagSet( features, FallbackShaderFeature::DIFFUSE_TEXTURE ) );
    EXPECT_TRUE( flagSet( features, FallbackShaderFeature::ALPHA_CUTOUT ) );
}

TEST( TestFallbackShaderContract, unallocatedTextureRequestsAreNotFallbackTextureFeatures )
{
    PhongMaterial material{};
    material.flags = MaterialFlags::DIFFUSE_MAP | MaterialFlags::ALPHA_MAP;

    const FallbackShaderFeature features{ fallbackFeaturesForMaterial( material ) };

    EXPECT_TRUE( flagSet( features, FallbackShaderFeature::CONSTANT_KD ) );
    EXPECT_FALSE( flagSet( features, FallbackShaderFeature::DIFFUSE_TEXTURE ) );
    EXPECT_FALSE( flagSet( features, FallbackShaderFeature::ALPHA_CUTOUT ) );
}

TEST( TestFallbackShaderContract, unsupportedFallbackAddsDiagnosticColorFeature )
{
    const MaterialState state{ makeMaterialState( 42U, MaterialBackend::LOCAL_FALLBACK, 0U, MaterialFallbackReason::UNSUPPORTED ) };
    const PhongMaterial material{};

    const FallbackShaderContract contract{ fallbackShaderContract( state, material ) };

    EXPECT_EQ( MaterialFallbackReason::UNSUPPORTED, contract.reason );
    EXPECT_TRUE( flagSet( contract.featureMask, FallbackShaderFeature::CONSTANT_KD ) );
    EXPECT_TRUE( flagSet( contract.featureMask, FallbackShaderFeature::DIAGNOSTIC_COLOR ) );
}
