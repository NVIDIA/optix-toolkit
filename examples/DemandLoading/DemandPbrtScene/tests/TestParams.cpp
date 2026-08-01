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

TEST( TestFallbackShaderContract, fourierTableReadyKeepsUnsupportedFallbackUntilGpuEvaluationExists )
{
    const MaterialState state{ makeMaterialState( 42U, MaterialBackend::FOURIER_TABLE_READY, 77U ) };

    EXPECT_EQ( 42U, state.materialId );
    EXPECT_EQ( MaterialBackend::FOURIER_TABLE_READY, state.backend );
    EXPECT_EQ( 77U, state.shaderKey );
    EXPECT_EQ( MaterialFallbackReason::UNSUPPORTED, state.fallbackReason );
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

#ifdef OTK_USE_MDL
TEST( TestMdlMaterialTextureBindings, defaultMdlMaterialShaderHasNoBoundTextures )
{
    const MdlMaterialShader data{ 7U, 4U };

    EXPECT_EQ( 7U, data.callableBaseIndex );
    EXPECT_EQ( 4U, data.callableCount );
    EXPECT_EQ( 0U, data.textureBindingCount );
    for( uint_t i = 0; i < MDL_MATERIAL_TEXTURE_BINDING_COUNT; ++i )
    {
        EXPECT_EQ( invalidMdlMaterialTextureBinding(), data.textureBindings[i] );
    }
}

TEST( TestFourierBsdfTableResourceBinding, defaultResourceHasNoBoundTable )
{
    const FourierMaterialResource resource{};

    EXPECT_EQ( INVALID_FOURIER_BSDF_TABLE_RESOURCE_ID, resource.resourceId );
    EXPECT_EQ( CUdeviceptr{}, resource.table );
    EXPECT_FALSE( hasFourierBsdfTableResource( resource ) );
}

TEST( TestFourierBsdfTableResourceBinding, resourceStoresCompactHandleAndDevicePointer )
{
    const FourierMaterialResource resource{ makeFourierMaterialResource( 55U, static_cast<CUdeviceptr>( 0x12340000U ) ) };

    EXPECT_EQ( 55U, resource.resourceId );
    EXPECT_EQ( static_cast<CUdeviceptr>( 0x12340000U ), resource.table );
    EXPECT_TRUE( hasFourierBsdfTableResource( resource ) );
}

TEST( TestMdlMaterialTextureBindings, setBindingTracksCountAndRejectsOverflow )
{
    MdlMaterialShader data{ 7U, 4U };

    EXPECT_TRUE( setMdlMaterialTextureBinding( data, MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX, 333U,
                                               make_float3( 0.25f, 0.5f, 0.75f ), make_float3( 0.125f, 0.25f, 0.375f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( data, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX, 444U,
                                               make_float3( 0.5f, 0.25f, 0.125f ), make_float3( 0.375f, 0.25f, 0.125f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( data, MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX, 555U,
                                               make_float3( 0.75f, 0.5f, 0.25f ), make_float3( 0.625f, 0.5f, 0.375f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( data, MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX, 666U,
                                               make_float3( 0.875f, 0.75f, 0.625f ), make_float3( 0.5f, 0.375f, 0.25f ) ) );

    EXPECT_EQ( MDL_MATERIAL_TEXTURE_BINDING_COUNT, data.textureBindingCount );
    EXPECT_EQ( 14U, MDL_MATERIAL_TEXTURE_BINDING_COUNT );
    EXPECT_EQ( 0U, MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 1U, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 2U, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 3U, MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 4U, MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 8U, MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 9U, MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 13U, MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX );
    EXPECT_EQ( 333U, data.textureBindings[MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX].textureId );
    EXPECT_EQ( make_float3( 0.25f, 0.5f, 0.75f ), data.textureBindings[MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX].scale );
    EXPECT_EQ( make_float3( 0.125f, 0.25f, 0.375f ), data.textureBindings[MDL_MATERIAL_DIFFUSE_TEXTURE_BINDING_INDEX].bias );
    EXPECT_EQ( 444U, data.textureBindings[MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX].textureId );
    EXPECT_EQ( make_float3( 0.5f, 0.25f, 0.125f ), data.textureBindings[MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX].scale );
    EXPECT_EQ( make_float3( 0.375f, 0.25f, 0.125f ), data.textureBindings[MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX].bias );
    EXPECT_EQ( 555U, data.textureBindings[MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX].textureId );
    EXPECT_EQ( make_float3( 0.75f, 0.5f, 0.25f ), data.textureBindings[MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX].scale );
    EXPECT_EQ( make_float3( 0.625f, 0.5f, 0.375f ), data.textureBindings[MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX].bias );
    EXPECT_EQ( 666U, data.textureBindings[MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX].textureId );
    EXPECT_EQ( make_float3( 0.875f, 0.75f, 0.625f ),
               data.textureBindings[MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX].scale );
    EXPECT_EQ( make_float3( 0.5f, 0.375f, 0.25f ),
               data.textureBindings[MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX].bias );
    EXPECT_FALSE( setMdlMaterialTextureBinding( data, MDL_MATERIAL_TEXTURE_BINDING_COUNT, 444U,
                                                make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
}
#endif

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
