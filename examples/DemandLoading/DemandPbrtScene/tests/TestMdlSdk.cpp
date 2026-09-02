// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <gmock/gmock.h>

#include "DemandPbrtScene/FourierBsdfTable.h"
#include "DemandPbrtScene/FourierMdlMeasuredBsdfCapability.h"
#include "DemandPbrtScene/MdlBsdfCompiler.h"
#include "DemandPbrtScene/MdlHandleTypes.h"
#include "DemandPbrtScene/MdlSdkSession.h"
#include "DemandPbrtScene/MdlShaderCache.h"

#include <mi/mdl_sdk.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using demandPbrtScene::MdlSdkSession;

using demandPbrtScene::BackendApiHandle;
using demandPbrtScene::BackendHandle;
using demandPbrtScene::BsdfMeasurementHandle;
using demandPbrtScene::ColorValueHandle;
using demandPbrtScene::CompiledMaterialHandle;
using demandPbrtScene::ConstColorValueHandle;
using demandPbrtScene::ConstExpressionConstantHandle;
using demandPbrtScene::ConstExpressionHandle;
using demandPbrtScene::ConstFloatValueHandle;
using demandPbrtScene::ConstStringHandle;
using demandPbrtScene::DatabaseHandle;
using demandPbrtScene::ExecutionContextHandle;
using demandPbrtScene::ExpressionConstantHandle;
using demandPbrtScene::ExpressionFactoryHandle;
using demandPbrtScene::FloatValueHandle;
using demandPbrtScene::FunctionCallHandle;
using demandPbrtScene::FunctionDefinitionHandle;
using demandPbrtScene::MaterialInstanceHandle;
using demandPbrtScene::MdlFactoryHandle;
using demandPbrtScene::MdlImpexpApiHandle;
using demandPbrtScene::MessageHandle;
using demandPbrtScene::ModuleHandle;
using demandPbrtScene::ScopeHandle;
using demandPbrtScene::TargetCodeHandle;
using demandPbrtScene::TransactionHandle;
using demandPbrtScene::TypeFactoryHandle;
using demandPbrtScene::TypeHandle;
using demandPbrtScene::ValueFactoryHandle;

constexpr mi::Float32 PBRT_KD_RED       = 0.25f;
constexpr mi::Float32 PBRT_KD_GREEN     = 0.50f;
constexpr mi::Float32 PBRT_KD_BLUE      = 0.75f;
constexpr mi::Float32 PBRT_KD_ALT_RED   = 0.75f;
constexpr mi::Float32 PBRT_KD_ALT_GREEN = 0.20f;
constexpr mi::Float32 PBRT_KD_ALT_BLUE  = 0.10f;

std::filesystem::path pbrtReferenceDir()
{
    return std::filesystem::path{ DEMAND_PBRT_SCENE_TEST_SOURCE_DIR } / "pbrt-reference";
}

struct BoundMdlColor
{
    mi::Float32 red{};
    mi::Float32 green{};
    mi::Float32 blue{};
};

mi::Float32 conductorNormalReflectance( mi::Float32 eta, mi::Float32 k )
{
    const mi::Float32 etaMinusOne{ eta - 1.0f };
    const mi::Float32 etaPlusOne{ eta + 1.0f };
    return ( etaMinusOne * etaMinusOne + k * k ) / ( etaPlusOne * etaPlusOne + k * k );
}

BoundMdlColor conductorNormalReflectance( const BoundMdlColor& eta, const BoundMdlColor& k )
{
    return BoundMdlColor{ conductorNormalReflectance( eta.red, k.red ), conductorNormalReflectance( eta.green, k.green ),
                          conductorNormalReflectance( eta.blue, k.blue ) };
}

std::string describeContextMessages( const mi::neuraylib::IMdl_execution_context* context )
{
    if( !context )
        return {};

    std::ostringstream out;
    for( mi::Size i = 0; i < context->get_messages_count(); ++i )
    {
        MessageHandle message( context->get_message( i ) );
        if( message.is_valid_interface() )
            out << message->get_string() << '\n';
    }
    return out.str();
}

void addRgbSpectrum( ::pbrt::ParamSet& params, const std::string& name, float red, float green, float blue )
{
    std::unique_ptr<::pbrt::Float[]> values{ new ::pbrt::Float[3] };
    values[0] = red;
    values[1] = green;
    values[2] = blue;
    params.AddRGBSpectrum( name, std::move( values ), 3 );
}

void addFloat( ::pbrt::ParamSet& params, const std::string& name, float value )
{
    std::unique_ptr<::pbrt::Float[]> values{ new ::pbrt::Float[1] };
    values[0] = value;
    params.AddFloat( name, std::move( values ), 1 );
}

void addString( ::pbrt::ParamSet& params, const std::string& name, const std::string& value )
{
    std::unique_ptr<std::string[]> values{ new std::string[1] };
    values[0] = value;
    params.AddString( name, std::move( values ), 1 );
}

otk::pbrt::PbrtTexture imageMapTexture( const std::string& name, const std::string& fileName, const std::string& valueType )
{
    otk::pbrt::PbrtTexture texture;
    texture.name      = name;
    texture.valueType = valueType;
    texture.type      = "imagemap";
    addString( texture.params, "filename", fileName );
    return texture;
}

otk::pbrt::PbrtTexture constantColorTexture( const std::string& name, const BoundMdlColor& value )
{
    otk::pbrt::PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "color";
    texture.type      = "constant";
    addRgbSpectrum( texture.params, "value", value.red, value.green, value.blue );
    return texture;
}

otk::pbrt::PbrtMaterial matteMaterial( float red, float green, float blue )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "matte";
    addRgbSpectrum( material.params, "Kd", red, green, blue );
    return material;
}

otk::pbrt::PbrtMaterial matteMaterialWithKdTexture( const BoundMdlColor& kd )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "matte";
    material.params.AddTexture( "Kd", "albedo" );
    material.graph.textures["color:albedo"] = constantColorTexture( "albedo", kd );
    return material;
}

otk::pbrt::PbrtMaterial matteMaterialWithSigma( float red, float green, float blue, float sigma )
{
    otk::pbrt::PbrtMaterial material{ matteMaterial( red, green, blue ) };
    addFloat( material.params, "sigma", sigma );
    return material;
}

otk::pbrt::PbrtMaterial matteMaterialWithBumpmap()
{
    otk::pbrt::PbrtMaterial material{ matteMaterial( 0.2f, 0.3f, 0.4f ) };
    material.params.AddTexture( "bumpmap", "height" );
    material.graph.textures["float:height"] = imageMapTexture( "height", "height.exr", "float" );
    return material;
}

otk::pbrt::PbrtMaterial plasticMaterial( const BoundMdlColor& kd, const BoundMdlColor& ks, float roughness )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "plastic";
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addRgbSpectrum( material.params, "Ks", ks.red, ks.green, ks.blue );
    addFloat( material.params, "roughness", roughness );
    return material;
}

otk::pbrt::PbrtMaterial plasticMaterial()
{
    return plasticMaterial( BoundMdlColor{ 0.2f, 0.3f, 0.4f }, BoundMdlColor{ 0.5f, 0.6f, 0.7f }, 0.25f );
}

otk::pbrt::PbrtMaterial uberMaterial( const BoundMdlColor& kd,
                                      const BoundMdlColor& ks,
                                      const BoundMdlColor& kr,
                                      const BoundMdlColor& kt,
                                      float                roughness,
                                      float                alpha,
                                      const BoundMdlColor& opacity )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "uber";
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addRgbSpectrum( material.params, "Ks", ks.red, ks.green, ks.blue );
    addRgbSpectrum( material.params, "Kr", kr.red, kr.green, kr.blue );
    addRgbSpectrum( material.params, "Kt", kt.red, kt.green, kt.blue );
    addFloat( material.params, "roughness", roughness );
    addFloat( material.params, "index", 1.4f );
    addFloat( material.params, "alpha", alpha );
    addRgbSpectrum( material.params, "opacity", opacity.red, opacity.green, opacity.blue );
    return material;
}

otk::pbrt::PbrtMaterial uberMaterial( const BoundMdlColor& kd,
                                      const BoundMdlColor& ks,
                                      const BoundMdlColor& kr,
                                      const BoundMdlColor& kt,
                                      float                roughness,
                                      float                alpha,
                                      float                opacity )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "uber";
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addRgbSpectrum( material.params, "Ks", ks.red, ks.green, ks.blue );
    addRgbSpectrum( material.params, "Kr", kr.red, kr.green, kr.blue );
    addRgbSpectrum( material.params, "Kt", kt.red, kt.green, kt.blue );
    addFloat( material.params, "roughness", roughness );
    addFloat( material.params, "index", 1.4f );
    addFloat( material.params, "alpha", alpha );
    addFloat( material.params, "opacity", opacity );
    return material;
}

otk::pbrt::PbrtMaterial uberMaterial( const BoundMdlColor& kd, const BoundMdlColor& ks, float roughness )
{
    return uberMaterial( kd, ks, BoundMdlColor{ 0.0f, 0.0f, 0.0f }, BoundMdlColor{ 0.0f, 0.0f, 0.0f }, roughness, 0.8f, 0.7f );
}

otk::pbrt::PbrtMaterial uberMaterial()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "uber";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addRgbSpectrum( material.params, "Kr", 0.1f, 0.2f, 0.3f );
    addRgbSpectrum( material.params, "Kt", 0.0f, 0.1f, 0.2f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "uroughness", 0.2f );
    addFloat( material.params, "vroughness", 0.3f );
    addFloat( material.params, "index", 1.4f );
    addFloat( material.params, "alpha", 0.8f );
    addFloat( material.params, "opacity", 0.7f );
    return material;
}

otk::pbrt::PbrtMaterial substrateMaterial( const BoundMdlColor& kd, const BoundMdlColor& ks, float roughness, float uroughness, float vroughness )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "substrate";
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addRgbSpectrum( material.params, "Ks", ks.red, ks.green, ks.blue );
    addFloat( material.params, "roughness", roughness );
    addFloat( material.params, "uroughness", uroughness );
    addFloat( material.params, "vroughness", vroughness );
    return material;
}

otk::pbrt::PbrtMaterial substrateMaterial()
{
    return substrateMaterial( BoundMdlColor{ 0.2f, 0.3f, 0.4f }, BoundMdlColor{ 0.5f, 0.6f, 0.7f }, 0.25f, 0.2f, 0.3f );
}

otk::pbrt::PbrtMaterial mirrorMaterial()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "mirror";
    addRgbSpectrum( material.params, "Kr", 0.2f, 0.3f, 0.4f );
    return material;
}

otk::pbrt::PbrtMaterial glassMaterial( float index, float roughness, float uroughness, float vroughness )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "glass";
    addRgbSpectrum( material.params, "Kr", 0.9f, 0.9f, 0.9f );
    addRgbSpectrum( material.params, "Kt", 0.7f, 0.8f, 1.0f );
    addFloat( material.params, "index", index );
    addFloat( material.params, "roughness", roughness );
    addFloat( material.params, "uroughness", uroughness );
    addFloat( material.params, "vroughness", vroughness );
    return material;
}

otk::pbrt::PbrtMaterial glassMaterial()
{
    return glassMaterial( 1.5f, 0.05f, 0.04f, 0.06f );
}

otk::pbrt::PbrtMaterial metalMaterial( const BoundMdlColor& eta, const BoundMdlColor& k, float roughness, float uroughness, float vroughness )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "metal";
    addRgbSpectrum( material.params, "eta", eta.red, eta.green, eta.blue );
    addRgbSpectrum( material.params, "k", k.red, k.green, k.blue );
    addFloat( material.params, "roughness", roughness );
    addFloat( material.params, "uroughness", uroughness );
    addFloat( material.params, "vroughness", vroughness );
    return material;
}

otk::pbrt::PbrtMaterial metalMaterial()
{
    return metalMaterial( BoundMdlColor{ 0.2f, 0.3f, 0.45f }, BoundMdlColor{ 2.2f, 2.8f, 3.4f }, 0.18f, 0.16f, 0.2f );
}

otk::pbrt::PbrtMaterial translucentMaterial( const BoundMdlColor& kd,
                                             const BoundMdlColor& ks,
                                             const BoundMdlColor& reflect,
                                             const BoundMdlColor& transmit,
                                             float                roughness,
                                             const BoundMdlColor& opacity )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "translucent";
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addRgbSpectrum( material.params, "Ks", ks.red, ks.green, ks.blue );
    addRgbSpectrum( material.params, "reflect", reflect.red, reflect.green, reflect.blue );
    addRgbSpectrum( material.params, "transmit", transmit.red, transmit.green, transmit.blue );
    addFloat( material.params, "roughness", roughness );
    addRgbSpectrum( material.params, "opacity", opacity.red, opacity.green, opacity.blue );
    return material;
}

otk::pbrt::PbrtMaterial translucentMaterial( const BoundMdlColor& kd,
                                             const BoundMdlColor& ks,
                                             const BoundMdlColor& reflect,
                                             const BoundMdlColor& transmit,
                                             float                roughness,
                                             float                opacity )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "translucent";
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addRgbSpectrum( material.params, "Ks", ks.red, ks.green, ks.blue );
    addRgbSpectrum( material.params, "reflect", reflect.red, reflect.green, reflect.blue );
    addRgbSpectrum( material.params, "transmit", transmit.red, transmit.green, transmit.blue );
    addFloat( material.params, "roughness", roughness );
    addFloat( material.params, "opacity", opacity );
    return material;
}

otk::pbrt::PbrtMaterial translucentMaterial()
{
    return translucentMaterial( BoundMdlColor{ 0.2f, 0.3f, 0.4f }, BoundMdlColor{ 0.5f, 0.6f, 0.7f },
                                BoundMdlColor{ 0.8f, 0.6f, 0.4f }, BoundMdlColor{ 0.2f, 0.4f, 0.6f }, 0.25f, 0.7f );
}

otk::pbrt::PbrtMaterial subsurfaceMaterial( const BoundMdlColor& kr,
                                            const BoundMdlColor& kt,
                                            const BoundMdlColor& sigmaA,
                                            const BoundMdlColor& sigmaS,
                                            float                scale,
                                            float                eta )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "subsurface";
    addRgbSpectrum( material.params, "Kr", kr.red, kr.green, kr.blue );
    addRgbSpectrum( material.params, "Kt", kt.red, kt.green, kt.blue );
    addRgbSpectrum( material.params, "sigma_a", sigmaA.red, sigmaA.green, sigmaA.blue );
    addRgbSpectrum( material.params, "sigma_s", sigmaS.red, sigmaS.green, sigmaS.blue );
    addFloat( material.params, "scale", scale );
    addFloat( material.params, "eta", eta );
    addFloat( material.params, "g", 0.0f );
    addFloat( material.params, "uroughness", 0.0f );
    addFloat( material.params, "vroughness", 0.0f );
    return material;
}

otk::pbrt::PbrtMaterial kdSubsurfaceMaterial( const BoundMdlColor& kd,
                                              const BoundMdlColor& kr,
                                              const BoundMdlColor& kt,
                                              const BoundMdlColor& mfp,
                                              float                scale,
                                              float                eta )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "kdsubsurface";
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addRgbSpectrum( material.params, "Kr", kr.red, kr.green, kr.blue );
    addRgbSpectrum( material.params, "Kt", kt.red, kt.green, kt.blue );
    addRgbSpectrum( material.params, "mfp", mfp.red, mfp.green, mfp.blue );
    addFloat( material.params, "scale", scale );
    addFloat( material.params, "eta", eta );
    addFloat( material.params, "g", 0.0f );
    addFloat( material.params, "uroughness", 0.0f );
    addFloat( material.params, "vroughness", 0.0f );
    return material;
}

otk::pbrt::PbrtNamedMaterial namedMatteMaterial( const std::string& name, const BoundMdlColor& kd )
{
    otk::pbrt::PbrtNamedMaterial material;
    material.name = name;
    material.type = "matte";
    addString( material.params, "type", "matte" );
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    return material;
}

otk::pbrt::PbrtNamedMaterial namedMatteMaterial( const std::string& name, float kd )
{
    return namedMatteMaterial( name, BoundMdlColor{ kd, kd, kd } );
}

otk::pbrt::PbrtNamedMaterial namedUberMaterialWithIndex( const std::string& name, const BoundMdlColor& kd )
{
    otk::pbrt::PbrtNamedMaterial material;
    material.name = name;
    material.type = "uber";
    addString( material.params, "type", "uber" );
    addRgbSpectrum( material.params, "Kd", kd.red, kd.green, kd.blue );
    addFloat( material.params, "index", 1.4f );
    return material;
}

otk::pbrt::PbrtMaterial mixMaterial( const BoundMdlColor& firstKd, const BoundMdlColor& secondKd, float amount )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", "front" );
    addString( material.params, "namedmaterial2", "back" );
    addFloat( material.params, "amount", amount );
    material.graph.namedMaterials["front"] = namedMatteMaterial( "front", firstKd );
    material.graph.namedMaterials["back"]  = namedMatteMaterial( "back", secondKd );
    return material;
}

otk::pbrt::PbrtMaterial mixMaterial( const BoundMdlColor& firstKd, const BoundMdlColor& secondKd, const BoundMdlColor& amount )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", "front" );
    addString( material.params, "namedmaterial2", "back" );
    addRgbSpectrum( material.params, "amount", amount.red, amount.green, amount.blue );
    material.graph.namedMaterials["front"] = namedMatteMaterial( "front", firstKd );
    material.graph.namedMaterials["back"]  = namedMatteMaterial( "back", secondKd );
    return material;
}

otk::pbrt::PbrtMaterial mixMaterial()
{
    return mixMaterial( BoundMdlColor{ 0.2f, 0.2f, 0.2f }, BoundMdlColor{ 0.8f, 0.8f, 0.8f }, 0.25f );
}

otk::pbrt::PbrtMaterial mixMaterialWithNamedUberIndex( const BoundMdlColor& firstKd, const BoundMdlColor& secondKd, float amount )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", "front" );
    addString( material.params, "namedmaterial2", "back" );
    addFloat( material.params, "amount", amount );
    material.graph.namedMaterials["front"] = namedUberMaterialWithIndex( "front", firstKd );
    material.graph.namedMaterials["back"]  = namedMatteMaterial( "back", secondKd );
    return material;
}

otk::pbrt::PbrtTexture constantFloatTexture( const std::string& name, float value )
{
    otk::pbrt::PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "float";
    texture.type      = "constant";
    addFloat( texture.params, "value", value );
    return texture;
}

otk::pbrt::PbrtMaterial mixMaterialWithAmountTexture( const BoundMdlColor& firstKd, const BoundMdlColor& secondKd, float amount )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", "front" );
    addString( material.params, "namedmaterial2", "back" );
    material.params.AddTexture( "amount", "weight" );
    material.graph.namedMaterials["front"]  = namedMatteMaterial( "front", firstKd );
    material.graph.namedMaterials["back"]   = namedMatteMaterial( "back", secondKd );
    material.graph.textures["float:weight"] = constantFloatTexture( "weight", amount );
    return material;
}

otk::pbrt::PbrtMaterial mixMaterialWithAmountTexture()
{
    return mixMaterialWithAmountTexture( BoundMdlColor{ 0.2f, 0.2f, 0.2f }, BoundMdlColor{ 0.8f, 0.8f, 0.8f }, 0.25f );
}

std::string describeGeneratedSource( const demandPbrtScene::GeneratedMdlSource& source, const demandPbrtScene::MdlShaderKey& key )
{
    return "module=" + source.moduleName + ", material=" + source.materialName + ", key=" + demandPbrtScene::toString( key );
}

CompiledMaterialHandle compileGeneratedMaterialWithBoundParameters(
    mi::neuraylib::INeuray*                                        neuray,
    mi::neuraylib::ITransaction*                                   transaction,
    mi::neuraylib::IMdl_execution_context*                         context,
    const demandPbrtScene::GeneratedMdlSource&                     source,
    const demandPbrtScene::MdlShaderKey&                           key,
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter>& parameters )
{
    const std::string sourceDescription{ describeGeneratedSource( source, key ) };
    MdlFactoryHandle mdlFactory( neuray->get_api_component<mi::neuraylib::IMdl_factory>() );
    EXPECT_TRUE( mdlFactory.is_valid_interface() ) << sourceDescription;
    if( !mdlFactory.is_valid_interface() )
        return {};

    MdlImpexpApiHandle mdlImpexpApi( neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>() );
    EXPECT_TRUE( mdlImpexpApi.is_valid_interface() ) << sourceDescription;
    if( !mdlImpexpApi.is_valid_interface() )
        return {};

    ConstStringHandle moduleDbName( mdlFactory->get_db_module_name( source.moduleName.c_str() ) );
    EXPECT_TRUE( moduleDbName.is_valid_interface() ) << sourceDescription;
    if( !moduleDbName.is_valid_interface() )
        return {};

    ModuleHandle module( transaction->access<mi::neuraylib::IModule>( moduleDbName->get_c_str() ) );
    if( !module.is_valid_interface() )
    {
        context->clear_messages();
        const mi::Sint32 loadResult =
            mdlImpexpApi->load_module_from_string( transaction, source.moduleName.c_str(), source.source.c_str(), context );
        EXPECT_EQ( 0, loadResult ) << sourceDescription << '\n' << describeContextMessages( context );
        if( loadResult != 0 )
            return {};

        module = transaction->access<mi::neuraylib::IModule>( moduleDbName->get_c_str() );
    }
    EXPECT_TRUE( module.is_valid_interface() ) << sourceDescription;
    if( !module.is_valid_interface() )
        return {};

    EXPECT_EQ( 1U, module->get_material_count() ) << sourceDescription;
    const char* const materialDbName = module->get_material( 0 );
    EXPECT_NE( nullptr, materialDbName ) << sourceDescription;
    if( !materialDbName )
        return {};

    FunctionDefinitionHandle materialDefinition(
        transaction->access<mi::neuraylib::IFunction_definition>( materialDbName ) );
    EXPECT_TRUE( materialDefinition.is_valid_interface() ) << sourceDescription;
    if( !materialDefinition.is_valid_interface() )
        return {};

    mi::Sint32 callResult = 0;
    FunctionCallHandle materialCall( materialDefinition->create_function_call( nullptr, &callResult ) );
    EXPECT_EQ( 0, callResult ) << sourceDescription;
    EXPECT_TRUE( materialCall.is_valid_interface() ) << sourceDescription;
    if( !materialCall.is_valid_interface() )
        return {};

    ValueFactoryHandle valueFactory( mdlFactory->create_value_factory( transaction ) );
    EXPECT_TRUE( valueFactory.is_valid_interface() ) << sourceDescription;
    if( !valueFactory.is_valid_interface() )
        return {};

    ExpressionFactoryHandle expressionFactory( mdlFactory->create_expression_factory( transaction ) );
    EXPECT_TRUE( expressionFactory.is_valid_interface() ) << sourceDescription;
    if( !expressionFactory.is_valid_interface() )
        return {};

    for( const demandPbrtScene::MdlBoundMaterialParameter& parameter : parameters )
    {
        if( parameter.type == demandPbrtScene::MdlBoundParameterType::COLOR )
        {
            ColorValueHandle value(
                valueFactory->create_color( parameter.red, parameter.green, parameter.blue ) );
            EXPECT_TRUE( value.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !value.is_valid_interface() )
                return {};

            ExpressionConstantHandle expression( expressionFactory->create_constant( value.get() ) );
            EXPECT_TRUE( expression.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !expression.is_valid_interface() )
                return {};

            EXPECT_EQ( 0, materialCall->set_argument( parameter.name.c_str(), expression.get() ) )
                << sourceDescription << ", parameter=" << parameter.name;
        }
        else
        {
            FloatValueHandle value( valueFactory->create_float( parameter.value ) );
            EXPECT_TRUE( value.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !value.is_valid_interface() )
                return {};

            ExpressionConstantHandle expression( expressionFactory->create_constant( value.get() ) );
            EXPECT_TRUE( expression.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !expression.is_valid_interface() )
                return {};

            EXPECT_EQ( 0, materialCall->set_argument( parameter.name.c_str(), expression.get() ) )
                << sourceDescription << ", parameter=" << parameter.name;
        }
    }

    MaterialInstanceHandle materialInstance(
        materialCall->get_interface<mi::neuraylib::IMaterial_instance>() );
    EXPECT_TRUE( materialInstance.is_valid_interface() ) << sourceDescription;
    if( !materialInstance.is_valid_interface() )
        return {};

    TypeFactoryHandle typeFactory( mdlFactory->create_type_factory( transaction ) );
    EXPECT_TRUE( typeFactory.is_valid_interface() ) << sourceDescription;
    if( !typeFactory.is_valid_interface() )
        return {};

    TypeHandle standardMaterialType(
        typeFactory->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL ) );
    EXPECT_TRUE( standardMaterialType.is_valid_interface() ) << sourceDescription;
    if( !standardMaterialType.is_valid_interface() )
        return {};

    context->clear_messages();
    const mi::Sint32 targetTypeResult = context->set_option( "target_type", standardMaterialType.get() );
    EXPECT_EQ( 0, targetTypeResult ) << sourceDescription << '\n' << describeContextMessages( context );
    if( targetTypeResult != 0 )
        return {};

    CompiledMaterialHandle compiledMaterial(
        materialInstance->create_compiled_material( mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context ) );
    EXPECT_TRUE( compiledMaterial.is_valid_interface() ) << sourceDescription << '\n'
                                                         << describeContextMessages( context );
    return compiledMaterial;
}

void expectColorExpressionMatches( const mi::neuraylib::ICompiled_material* compiledMaterial,
                                   const char*                              expressionPath,
                                   const BoundMdlColor&                     expected )
{
    ConstExpressionHandle tintExpression( compiledMaterial->lookup_sub_expression( expressionPath ) );
    ASSERT_TRUE( tintExpression.is_valid_interface() );
    ASSERT_EQ( mi::neuraylib::IExpression::EK_CONSTANT, tintExpression->get_kind() );

    ConstExpressionConstantHandle tintConstant(
        tintExpression->get_interface<mi::neuraylib::IExpression_constant>() );
    ASSERT_TRUE( tintConstant.is_valid_interface() );

    ConstColorValueHandle tintValue( tintConstant->get_value<mi::neuraylib::IValue_color>() );
    ASSERT_TRUE( tintValue.is_valid_interface() );

    ConstFloatValueHandle red( tintValue->get_value( 0 ) );
    ConstFloatValueHandle green( tintValue->get_value( 1 ) );
    ConstFloatValueHandle blue( tintValue->get_value( 2 ) );
    ASSERT_TRUE( red.is_valid_interface() );
    ASSERT_TRUE( green.is_valid_interface() );
    ASSERT_TRUE( blue.is_valid_interface() );

    EXPECT_NEAR( expected.red, red->get_value(), 1.0e-6f );
    EXPECT_NEAR( expected.green, green->get_value(), 1.0e-6f );
    EXPECT_NEAR( expected.blue, blue->get_value(), 1.0e-6f );
}

void expectTintMatchesColor( const mi::neuraylib::ICompiled_material* compiledMaterial, const BoundMdlColor& expected )
{
    expectColorExpressionMatches( compiledMaterial, "surface.scattering.tint", expected );
}

void expectTintMatchesPbrtKd( const mi::neuraylib::ICompiled_material* compiledMaterial, const BoundMdlColor& kd )
{
    expectTintMatchesColor( compiledMaterial, kd );
}

void expectFloatExpressionMatches( const mi::neuraylib::ICompiled_material* compiledMaterial, const char* expressionPath, float expected )
{
    ConstExpressionHandle expression( compiledMaterial->lookup_sub_expression( expressionPath ) );
    ASSERT_TRUE( expression.is_valid_interface() );
    ASSERT_EQ( mi::neuraylib::IExpression::EK_CONSTANT, expression->get_kind() );

    ConstExpressionConstantHandle constant(
        expression->get_interface<mi::neuraylib::IExpression_constant>() );
    ASSERT_TRUE( constant.is_valid_interface() );

    ConstFloatValueHandle value( constant->get_value<mi::neuraylib::IValue_float>() );
    ASSERT_TRUE( value.is_valid_interface() );

    EXPECT_NEAR( expected, value->get_value(), 1.0e-6f );
}

void expectIorMatchesFloat( const mi::neuraylib::ICompiled_material* compiledMaterial, float index )
{
    expectColorExpressionMatches( compiledMaterial, "ior", BoundMdlColor{ index, index, index } );
}

const char* findPreviewColorExpressionPath( const mi::neuraylib::ICompiled_material* compiledMaterial )
{
    static const char* const paths[] = { "surface.scattering.tint", "ior" };
    for( const char* path : paths )
    {
        ConstExpressionHandle expression( compiledMaterial->lookup_sub_expression( path ) );
        if( expression.is_valid_interface() )
        {
            return path;
        }
    }
    ADD_FAILURE() << "Compiled material has no preview color expression";
    return "";
}

std::string translateTintExpressionToPtx( mi::neuraylib::INeuray*                  neuray,
                                          mi::neuraylib::ITransaction*             transaction,
                                          const mi::neuraylib::ICompiled_material* compiledMaterial,
                                          mi::neuraylib::IMdl_execution_context*   context )
{
    BackendApiHandle backendApi( neuray->get_api_component<mi::neuraylib::IMdl_backend_api>() );
    EXPECT_TRUE( backendApi.is_valid_interface() );
    if( !backendApi.is_valid_interface() )
        return {};

    BackendHandle ptxBackend( backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
    EXPECT_TRUE( ptxBackend.is_valid_interface() );
    if( !ptxBackend.is_valid_interface() )
        return {};

    context->clear_messages();
    const char* const previewColorExpressionPath{ findPreviewColorExpressionPath( compiledMaterial ) };
    if( previewColorExpressionPath[0] == '\0' )
    {
        return {};
    }
    TargetCodeHandle targetCode( ptxBackend->translate_material_expression(
        transaction, compiledMaterial, previewColorExpressionPath, "evaluate_tint", context ) );
    EXPECT_TRUE( targetCode.is_valid_interface() ) << describeContextMessages( context );
    if( !targetCode.is_valid_interface() )
        return {};
    EXPECT_GT( targetCode->get_code_size(), 0U );
    EXPECT_EQ( 1U, targetCode->get_callable_function_count() );
    if( targetCode->get_callable_function_count() != 1U )
        return {};
    EXPECT_STREQ( "evaluate_tint", targetCode->get_callable_function( 0 ) );
    return std::string{ targetCode->get_code(), static_cast<std::size_t>( targetCode->get_code_size() ) };
}

std::string translateNormalExpressionToPtx( mi::neuraylib::INeuray*                  neuray,
                                            mi::neuraylib::ITransaction*             transaction,
                                            const mi::neuraylib::ICompiled_material* compiledMaterial,
                                            mi::neuraylib::IMdl_execution_context*   context )
{
    BackendApiHandle backendApi( neuray->get_api_component<mi::neuraylib::IMdl_backend_api>() );
    EXPECT_TRUE( backendApi.is_valid_interface() );
    if( !backendApi.is_valid_interface() )
        return {};

    BackendHandle ptxBackend( backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
    EXPECT_TRUE( ptxBackend.is_valid_interface() );
    if( !ptxBackend.is_valid_interface() )
        return {};

    context->clear_messages();
    TargetCodeHandle targetCode(
        ptxBackend->translate_material_expression( transaction, compiledMaterial, "geometry.normal", "evaluate_normal", context ) );
    EXPECT_TRUE( targetCode.is_valid_interface() ) << describeContextMessages( context );
    if( !targetCode.is_valid_interface() )
        return {};
    EXPECT_GT( targetCode->get_code_size(), 0U );
    EXPECT_EQ( 1U, targetCode->get_callable_function_count() );
    if( targetCode->get_callable_function_count() != 1U )
        return {};
    EXPECT_STREQ( "evaluate_normal", targetCode->get_callable_function( 0 ) );
    return std::string{ targetCode->get_code(), static_cast<std::size_t>( targetCode->get_code_size() ) };
}

class TestMdlSdk : public testing::Test
{
  protected:
    void SetUp() override
    {
        ASSERT_TRUE( session.isStarted() ) << session.error();

        database = session.neuray()->get_api_component<mi::neuraylib::IDatabase>();
        ASSERT_TRUE( database.is_valid_interface() );

        scope = database->get_global_scope();
        ASSERT_TRUE( scope.is_valid_interface() );

        transaction = scope->create_transaction();
        ASSERT_TRUE( transaction.is_valid_interface() );

        mdlFactory = session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>();
        ASSERT_TRUE( mdlFactory.is_valid_interface() );

        context = mdlFactory->create_execution_context();
        ASSERT_TRUE( context.is_valid_interface() );
    }

    void TearDown() override
    {
        context.reset();
        mdlFactory.reset();
        if( transaction.is_valid_interface() )
        {
            EXPECT_EQ( 0, transaction->commit() );
            transaction.reset();
        }
        scope.reset();
        database.reset();
        if( session.isStarted() )
        {
            EXPECT_EQ( 0, session.shutdown() );
        }
    }

    CompiledMaterialHandle compileMaterial(
        const demandPbrtScene::GeneratedMdlSource&                     source,
        const demandPbrtScene::MdlShaderKey&                           key,
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter>& parameters )
    {
        return compileGeneratedMaterialWithBoundParameters( session.neuray(), transaction.get(), context.get(), source, key,
                                                            parameters );
    }

    MdlSdkSession          session;
    DatabaseHandle         database;
    ScopeHandle            scope;
    TransactionHandle      transaction;
    MdlFactoryHandle       mdlFactory;
    ExecutionContextHandle context;
};

}  // namespace

TEST( TestMdlSdkHeaders, headerProvidesVersionMetadata )
{
    EXPECT_GT( std::strlen( MI_NEURAYLIB_PRODUCT_VERSION_STRING ), 0U );
    EXPECT_GT( MI_NEURAYLIB_API_VERSION, 0 );
}

TEST( TestMdlSdkHeaders, headerProvidesNeurayInterfaceId )
{
    const mi::base::Uuid id = mi::neuraylib::INeuray::IID();

    EXPECT_NE( 0U, id.m_id1 | id.m_id2 | id.m_id3 | id.m_id4 );
}

TEST_F( TestMdlSdk, rejectsPbrtFourierFixtureAsMdlMeasuredBsdfResource )
{
    const std::filesystem::path fixture{ pbrtReferenceDir() / "bsdfs" / "roughgold_alpha_0.2.bsdf" };
    const demandPbrtScene::FourierBsdfTableLoadResult table{ demandPbrtScene::loadFourierBsdfTable( fixture.string() ) };
    ASSERT_TRUE( table ) << table.diagnostic;
    BsdfMeasurementHandle measurement(
        transaction->create<mi::neuraylib::IBsdf_measurement>( "Bsdf_measurement" ) );
    ASSERT_TRUE( measurement.is_valid_interface() );

    const auto resetResult = measurement->reset_file( fixture.string().c_str() );
    const demandPbrtScene::FourierMdlMeasuredBsdfCapability capability{ demandPbrtScene::fourierMdlMeasuredBsdfCapability() };

    EXPECT_EQ( -3, resetResult );
    EXPECT_FALSE( capability.acceptsPbrtBsdfTables );
    EXPECT_FALSE( capability.exposesSampleEvaluatePdfCallables );
    EXPECT_EQ( demandPbrtScene::FourierGpuEvaluationPath::PBRT_FOURIER_CALLABLE, capability.selectedPath );
    EXPECT_THAT( capability.reason, testing::HasSubstr( ".mbsdf" ) );
    EXPECT_THAT( capability.reason, testing::HasSubstr( ".bsdf" ) );

    measurement.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedMatteMaterialWithBoundKd )
{
    const otk::pbrt::PbrtMaterial       sourceMaterial{ matteMaterial( PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE ) };
    const demandPbrtScene::MdlShaderKey key{ demandPbrtScene::makeMdlShaderKey( sourceMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( sourceMaterial ) };
    const BoundMdlColor firstKd{ PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE };
    const BoundMdlColor secondKd{ PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( sourceMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{ demandPbrtScene::makeMdlBoundMaterialParameters(
        matteMaterial( PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE ) ) };

    CompiledMaterialHandle compiledMaterial( compileMaterial( generated, key, firstParameters ) );
    ASSERT_TRUE( compiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, key, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    const std::string firstPtx{
        translateTintExpressionToPtx( session.neuray(), transaction.get(), compiledMaterial.get(), context.get() ) };
    const std::string secondPtx{ translateTintExpressionToPtx( session.neuray(), transaction.get(),
                                                               secondCompiledMaterial.get(), context.get() ) };

    expectTintMatchesPbrtKd( compiledMaterial.get(), firstKd );
    expectTintMatchesPbrtKd( secondCompiledMaterial.get(), secondKd );
    EXPECT_FALSE( firstPtx.empty() );
    EXPECT_FALSE( secondPtx.empty() );
    EXPECT_NE( firstPtx, secondPtx );

    secondCompiledMaterial.reset();
    compiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedMatteMaterialWithFoldedKdTexture )
{
    const BoundMdlColor                        firstKd{ PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE };
    const BoundMdlColor                        secondKd{ PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE };
    const otk::pbrt::PbrtMaterial              sourceMaterial{ matteMaterialWithKdTexture( firstKd ) };
    const demandPbrtScene::MdlShaderKey        key{ demandPbrtScene::makeMdlShaderKey( sourceMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( sourceMaterial ) };
    const demandPbrtScene::GeneratedMdlSource& secondGenerated{ sourceCache.getSource( matteMaterialWithKdTexture( secondKd ) ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( sourceMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( matteMaterialWithKdTexture( secondKd ) ) };

    CompiledMaterialHandle compiledMaterial( compileMaterial( generated, key, firstParameters ) );
    ASSERT_TRUE( compiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, key, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );

    EXPECT_EQ( generated.moduleName, secondGenerated.moduleName );
    EXPECT_EQ( generated.materialName, secondGenerated.materialName );
    EXPECT_EQ( generated.source, secondGenerated.source );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input Kd: Kd" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "// pbrt texture node:" ) ) );
    expectTintMatchesPbrtKd( compiledMaterial.get(), firstKd );
    expectTintMatchesPbrtKd( secondCompiledMaterial.get(), secondKd );

    compiledMaterial.reset();
    secondCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedMaterialWithRuntimeBumpmap )
{
    const otk::pbrt::PbrtMaterial              sourceMaterial{ matteMaterialWithBumpmap() };
    const demandPbrtScene::MdlShaderKey        key{ demandPbrtScene::makeMdlShaderKey( sourceMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( sourceMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, key ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> parameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( sourceMaterial ) };

    CompiledMaterialHandle compiledMaterial( compileMaterial( generated, key, parameters ) );
    ASSERT_TRUE( compiledMaterial.is_valid_interface() ) << sourceDescription;
    const std::string normalPtx{ translateNormalExpressionToPtx( session.neuray(), transaction.get(),
                                                                 compiledMaterial.get(), context.get() ) };

    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "pbrt_bump_normal" ) ) );
    EXPECT_THAT( generated.source,
                 testing::HasSubstr( "// pbrt material implementation: bumpmap is evaluated with runtime finite differences" ) );
    EXPECT_FALSE( normalPtx.empty() ) << sourceDescription;
    EXPECT_THAT( normalPtx, testing::HasSubstr( "evaluate_normal" ) ) << sourceDescription;

    compiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedMatteBsdfCallablesWithBoundKd )
{
    const otk::pbrt::PbrtMaterial firstMaterial{ matteMaterial( PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE ) };
    const otk::pbrt::PbrtMaterial secondMaterial{ matteMaterial( PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE ) };
    const otk::pbrt::PbrtMaterial roughMaterial{ matteMaterialWithSigma( PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE, 45.0f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string                          sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle roughCompiledMaterial( compileMaterial( generated, firstKey, roughParameters ) );
    ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_matte_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_matte_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), roughCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_matte_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_EQ( "pbrt_matte_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_matte_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_matte_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_matte_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( roughBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    roughCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedMirrorBsdfCallablesWithBoundKr )
{
    const otk::pbrt::PbrtMaterial       material{ mirrorMaterial() };
    const demandPbrtScene::MdlShaderKey key{ demandPbrtScene::makeMdlShaderKey( material ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( material ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, key ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> parameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( material ) };

    CompiledMaterialHandle compiledMaterial( compileMaterial( generated, key, parameters ) );
    ASSERT_TRUE( compiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx bsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), compiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_mirror_bsdf" ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::specular_bsdf" ) );
    EXPECT_EQ( "pbrt_mirror_bsdf_init", bsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_mirror_bsdf_sample", bsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_mirror_bsdf_evaluate", bsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_mirror_bsdf_pdf", bsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.initFunctionName ) );
    EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.sampleFunctionName ) );
    EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.evaluateFunctionName ) );
    EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.pdfFunctionName ) );

    compiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedPlasticBsdfCallablesWithBoundDiffuseAndGlossyInputs )
{
    const BoundMdlColor                 firstKd{ 0.2f, 0.3f, 0.4f };
    const BoundMdlColor                 firstKs{ 0.5f, 0.6f, 0.7f };
    const BoundMdlColor                 secondKd{ 0.6f, 0.2f, 0.1f };
    const BoundMdlColor                 secondKs{ 0.1f, 0.2f, 0.3f };
    const otk::pbrt::PbrtMaterial       firstMaterial{ plasticMaterial( firstKd, firstKs, 0.25f ) };
    const otk::pbrt::PbrtMaterial       secondMaterial{ plasticMaterial( secondKd, secondKs, 0.25f ) };
    const otk::pbrt::PbrtMaterial       roughMaterial{ plasticMaterial( firstKd, firstKs, 0.45f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle roughCompiledMaterial( compileMaterial( generated, firstKey, roughParameters ) );
    ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_plastic_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_plastic_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), roughCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_plastic_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "component: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_EQ( "pbrt_plastic_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_plastic_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_plastic_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_plastic_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( roughBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    roughCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedUberBsdfCallablesWithBoundDiffuseAndGlossyInputs )
{
    const BoundMdlColor firstKd{ 0.2f, 0.3f, 0.4f };
    const BoundMdlColor firstKs{ 0.5f, 0.6f, 0.7f };
    const BoundMdlColor firstKr{ 0.2f, 0.1f, 0.0f };
    const BoundMdlColor firstKt{ 0.1f, 0.2f, 0.3f };
    const BoundMdlColor secondKd{ 0.6f, 0.2f, 0.1f };
    const BoundMdlColor secondKs{ 0.1f, 0.2f, 0.3f };
    const BoundMdlColor secondKr{ 0.4f, 0.2f, 0.1f };
    const BoundMdlColor secondKt{ 0.3f, 0.1f, 0.2f };
    const BoundMdlColor spectrumOpacity{ 0.2f, 0.5f, 0.8f };
    const otk::pbrt::PbrtMaterial firstMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.25f, 0.8f, 0.7f ) };
    const otk::pbrt::PbrtMaterial secondMaterial{ uberMaterial( secondKd, secondKs, secondKr, secondKt, 0.25f, 0.8f, 0.7f ) };
    const otk::pbrt::PbrtMaterial roughMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.45f, 0.8f, 0.7f ) };
    const otk::pbrt::PbrtMaterial opacityMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.25f, 0.8f, 0.35f ) };
    const otk::pbrt::PbrtMaterial spectrumOpacityMaterial{
        uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.25f, 0.8f, spectrumOpacity ) };
    const otk::pbrt::PbrtMaterial alphaMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.25f, 0.35f, 0.7f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    const demandPbrtScene::MdlShaderKey opacityKey{ demandPbrtScene::makeMdlShaderKey( opacityMaterial ) };
    const demandPbrtScene::MdlShaderKey spectrumOpacityKey{ demandPbrtScene::makeMdlShaderKey( spectrumOpacityMaterial ) };
    const demandPbrtScene::MdlShaderKey alphaKey{ demandPbrtScene::makeMdlShaderKey( alphaMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> opacityParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( opacityMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> spectrumOpacityParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( spectrumOpacityMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> alphaParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( alphaMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle roughCompiledMaterial( compileMaterial( generated, firstKey, roughParameters ) );
    ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle opacityCompiledMaterial( compileMaterial( generated, firstKey, opacityParameters ) );
    ASSERT_TRUE( opacityCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle spectrumOpacityCompiledMaterial( compileMaterial( generated, firstKey, spectrumOpacityParameters ) );
    ASSERT_TRUE( spectrumOpacityCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle alphaCompiledMaterial( compileMaterial( generated, firstKey, alphaParameters ) );
    ASSERT_TRUE( alphaCompiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_uber_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_uber_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), roughCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_uber_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx opacityBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), opacityCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_uber_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx spectrumOpacityBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), spectrumOpacityCompiledMaterial.get(), context.get(),
        "surface.scattering", "pbrt_uber_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( opacityKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( spectrumOpacityKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( alphaKey ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::specular_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_transmit" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "component: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_uber_resolved_roughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color opacity = color(1.0, 1.0, 1.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_uber_opacity_weight" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_uber_transparency_weight" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "cutout_opacity: alpha" ) );
    expectIorMatchesFloat( firstCompiledMaterial.get(), 1.4f );
    expectFloatExpressionMatches( firstCompiledMaterial.get(), "geometry.cutout_opacity", 0.8f );
    expectFloatExpressionMatches( opacityCompiledMaterial.get(), "geometry.cutout_opacity", 0.8f );
    expectFloatExpressionMatches( alphaCompiledMaterial.get(), "geometry.cutout_opacity", 0.35f );
    EXPECT_EQ( "pbrt_uber_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_uber_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_uber_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_uber_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( roughBsdf.ptx.empty() );
    EXPECT_FALSE( opacityBsdf.ptx.empty() );
    EXPECT_FALSE( spectrumOpacityBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, opacityBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, spectrumOpacityBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    roughCompiledMaterial.reset();
    opacityCompiledMaterial.reset();
    spectrumOpacityCompiledMaterial.reset();
    alphaCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedSubstrateBsdfCallablesWithBoundLayeredInputs )
{
    const BoundMdlColor                 firstKd{ 0.2f, 0.3f, 0.4f };
    const BoundMdlColor                 firstKs{ 0.5f, 0.6f, 0.7f };
    const BoundMdlColor                 secondKd{ 0.6f, 0.2f, 0.1f };
    const BoundMdlColor                 secondKs{ 0.1f, 0.2f, 0.3f };
    const otk::pbrt::PbrtMaterial       firstMaterial{ substrateMaterial( firstKd, firstKs, 0.25f, 0.2f, 0.3f ) };
    const otk::pbrt::PbrtMaterial       secondMaterial{ substrateMaterial( secondKd, secondKs, 0.25f, 0.2f, 0.3f ) };
    const otk::pbrt::PbrtMaterial       roughMaterial{ substrateMaterial( firstKd, firstKs, 0.45f, 0.35f, 0.4f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle roughCompiledMaterial( compileMaterial( generated, firstKey, roughParameters ) );
    ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), firstCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_substrate_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), secondCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_substrate_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), roughCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_substrate_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_weighted_layer" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "base: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_substrate_resolved_roughness" ) );
    EXPECT_EQ( "pbrt_substrate_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_substrate_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_substrate_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_substrate_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( roughBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    roughCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedGlassBsdfCallablesWithBoundDielectricInputs )
{
    const otk::pbrt::PbrtMaterial       firstMaterial{ glassMaterial( 1.5f, 0.0f, 0.0f, 0.0f ) };
    const otk::pbrt::PbrtMaterial       secondMaterial{ glassMaterial( 1.1f, 0.0f, 0.0f, 0.0f ) };
    const otk::pbrt::PbrtMaterial       roughMaterial{ glassMaterial( 1.5f, 0.2f, 0.3f, 0.4f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle roughCompiledMaterial( compileMaterial( generated, firstKey, roughParameters ) );
    ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );
    const std::string tintPtx{ translateTintExpressionToPtx( session.neuray(), transaction.get(),
                                                             firstCompiledMaterial.get(), context.get() ) };
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_glass_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_glass_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), roughCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_glass_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::microfacet_ggx_smith_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_glass_resolved_roughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_reflect_transmit" ) );
    expectIorMatchesFloat( firstCompiledMaterial.get(), 1.5f );
    expectIorMatchesFloat( secondCompiledMaterial.get(), 1.1f );
    expectIorMatchesFloat( roughCompiledMaterial.get(), 1.5f );
    EXPECT_FALSE( tintPtx.empty() );
    EXPECT_EQ( "pbrt_glass_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_glass_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_glass_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_glass_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( roughBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    roughCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedMetalBsdfCallablesWithBoundConductorInputs )
{
    const BoundMdlColor                 firstEta{ 0.2f, 0.3f, 0.45f };
    const BoundMdlColor                 firstK{ 2.2f, 2.8f, 3.4f };
    const BoundMdlColor                 secondEta{ 0.4f, 0.5f, 0.6f };
    const BoundMdlColor                 secondK{ 1.5f, 2.0f, 2.5f };
    const otk::pbrt::PbrtMaterial       firstMaterial{ metalMaterial( firstEta, firstK, 0.18f, 0.16f, 0.2f ) };
    const otk::pbrt::PbrtMaterial       secondMaterial{ metalMaterial( secondEta, secondK, 0.35f, 0.22f, 0.31f ) };
    const otk::pbrt::PbrtMaterial       roughMaterial{ metalMaterial( firstEta, firstK, 0.18f, 0.33f, 0.41f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle roughCompiledMaterial( compileMaterial( generated, firstKey, roughParameters ) );
    ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_metal_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_metal_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), roughCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_metal_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::microfacet_ggx_smith_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_metal_conductor_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_metal_resolved_roughness" ) );
    expectTintMatchesColor( firstCompiledMaterial.get(), conductorNormalReflectance( firstEta, firstK ) );
    expectTintMatchesColor( secondCompiledMaterial.get(), conductorNormalReflectance( secondEta, secondK ) );
    expectTintMatchesColor( roughCompiledMaterial.get(), conductorNormalReflectance( firstEta, firstK ) );
    EXPECT_EQ( "pbrt_metal_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_metal_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_metal_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_metal_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( roughBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    roughCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedTranslucentBsdfCallablesWithBoundReflectionTransmissionAndOpacity )
{
    const BoundMdlColor firstKd{ 0.2f, 0.3f, 0.4f };
    const BoundMdlColor firstKs{ 0.5f, 0.6f, 0.7f };
    const BoundMdlColor firstReflect{ 0.8f, 0.6f, 0.4f };
    const BoundMdlColor firstTransmit{ 0.2f, 0.4f, 0.6f };
    const BoundMdlColor secondKd{ 0.6f, 0.2f, 0.1f };
    const BoundMdlColor secondKs{ 0.1f, 0.2f, 0.3f };
    const BoundMdlColor secondReflect{ 0.3f, 0.5f, 0.7f };
    const BoundMdlColor secondTransmit{ 0.7f, 0.4f, 0.2f };
    const BoundMdlColor spectrumOpacity{ 0.2f, 0.5f, 0.8f };
    const otk::pbrt::PbrtMaterial firstMaterial{ translucentMaterial( firstKd, firstKs, firstReflect, firstTransmit, 0.25f, 0.7f ) };
    const otk::pbrt::PbrtMaterial secondMaterial{
        translucentMaterial( secondKd, secondKs, secondReflect, secondTransmit, 0.25f, 0.7f ) };
    const otk::pbrt::PbrtMaterial roughMaterial{ translucentMaterial( firstKd, firstKs, firstReflect, firstTransmit, 0.45f, 0.7f ) };
    const otk::pbrt::PbrtMaterial opacityMaterial{ translucentMaterial( firstKd, firstKs, firstReflect, firstTransmit, 0.25f, 0.35f ) };
    const otk::pbrt::PbrtMaterial spectrumOpacityMaterial{
        translucentMaterial( firstKd, firstKs, firstReflect, firstTransmit, 0.25f, spectrumOpacity ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    const demandPbrtScene::MdlShaderKey opacityKey{ demandPbrtScene::makeMdlShaderKey( opacityMaterial ) };
    const demandPbrtScene::MdlShaderKey spectrumOpacityKey{ demandPbrtScene::makeMdlShaderKey( spectrumOpacityMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> opacityParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( opacityMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> spectrumOpacityParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( spectrumOpacityMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle roughCompiledMaterial( compileMaterial( generated, firstKey, roughParameters ) );
    ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle opacityCompiledMaterial( compileMaterial( generated, firstKey, opacityParameters ) );
    ASSERT_TRUE( opacityCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle spectrumOpacityCompiledMaterial( compileMaterial( generated, firstKey, spectrumOpacityParameters ) );
    ASSERT_TRUE( spectrumOpacityCompiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), firstCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_translucent_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), secondCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_translucent_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), roughCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_translucent_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx opacityBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), opacityCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_translucent_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx spectrumOpacityBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), spectrumOpacityCompiledMaterial.get(), context.get(),
        "surface.scattering", "pbrt_translucent_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( opacityKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( spectrumOpacityKey ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::diffuse_transmission_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_reflect" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_transmit" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color opacity = color(1.0, 1.0, 1.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_translucent_opacity_weight" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_translucent_transparency_weight" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "cutout_opacity: opacity" ) ) );
    expectIorMatchesFloat( firstCompiledMaterial.get(), 1.5f );
    EXPECT_EQ( "pbrt_translucent_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_translucent_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_translucent_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_translucent_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( roughBsdf.ptx.empty() );
    EXPECT_FALSE( opacityBsdf.ptx.empty() );
    EXPECT_FALSE( spectrumOpacityBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, opacityBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, spectrumOpacityBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    roughCompiledMaterial.reset();
    opacityCompiledMaterial.reset();
    spectrumOpacityCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesGeneratedSubsurfaceApproximationBsdfCallables )
{
    const otk::pbrt::PbrtMaterial subsurface{
        subsurfaceMaterial( BoundMdlColor{ 0.7f, 0.6f, 0.5f }, BoundMdlColor{ 0.2f, 0.3f, 0.4f },
                            BoundMdlColor{ 0.01f, 0.02f, 0.03f }, BoundMdlColor{ 0.6f, 0.5f, 0.4f }, 2.0f, 1.4f ) };
    const otk::pbrt::PbrtMaterial kdSubsurface{
        kdSubsurfaceMaterial( BoundMdlColor{ 0.2f, 0.3f, 0.4f }, BoundMdlColor{ 0.8f, 0.7f, 0.6f },
                              BoundMdlColor{ 0.4f, 0.5f, 0.6f }, BoundMdlColor{ 0.1f, 0.2f, 0.3f }, 1.5f, 1.33f ) };
    demandPbrtScene::MdlGeneratedSourceCache sourceCache;
    const auto expectCompiledBsdf = [&]( const otk::pbrt::PbrtMaterial& material, const char* callablePrefix, float expectedEta ) {
        const demandPbrtScene::MdlShaderKey        key{ demandPbrtScene::makeMdlShaderKey( material ) };
        const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( material ) };
        const std::string                          sourceDescription{ describeGeneratedSource( generated, key ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> parameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( material ) };

        CompiledMaterialHandle compiledMaterial( compileMaterial( generated, key, parameters ) );
        ASSERT_TRUE( compiledMaterial.is_valid_interface() ) << sourceDescription;
        const demandPbrtScene::MdlBsdfCallablePtx bsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), compiledMaterial.get(),
                                                           context.get(), "surface.scattering", callablePrefix ) };

        EXPECT_TRUE( generated.unsupportedReasons.empty() ) << sourceDescription;
        EXPECT_THAT( generated.source, testing::HasSubstr( "::df::diffuse_reflection_bsdf" ) );
        EXPECT_THAT( generated.source, testing::HasSubstr( "::df::diffuse_transmission_bsdf" ) );
        expectIorMatchesFloat( compiledMaterial.get(), expectedEta );
        EXPECT_EQ( std::string{ callablePrefix } + "_init", bsdf.initFunctionName ) << sourceDescription;
        EXPECT_EQ( std::string{ callablePrefix } + "_sample", bsdf.sampleFunctionName ) << sourceDescription;
        EXPECT_EQ( std::string{ callablePrefix } + "_evaluate", bsdf.evaluateFunctionName ) << sourceDescription;
        EXPECT_EQ( std::string{ callablePrefix } + "_pdf", bsdf.pdfFunctionName ) << sourceDescription;
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.initFunctionName ) );
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.sampleFunctionName ) );
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.evaluateFunctionName ) );
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.pdfFunctionName ) );
    };

    expectCompiledBsdf( subsurface, "pbrt_subsurface_bsdf", 1.4f );
    expectCompiledBsdf( kdSubsurface, "pbrt_kdsubsurface_bsdf", 1.33f );
}

TEST_F( TestMdlSdk, compilesGeneratedMixBsdfCallablesWithBoundNamedMaterialClosures )
{
    const BoundMdlColor           firstFront{ 0.8f, 0.2f, 0.1f };
    const BoundMdlColor           firstBack{ 0.1f, 0.3f, 0.9f };
    const BoundMdlColor           secondFront{ 0.2f, 0.7f, 0.4f };
    const BoundMdlColor           secondBack{ 0.8f, 0.1f, 0.3f };
    const otk::pbrt::PbrtMaterial firstMaterial{ mixMaterial( firstFront, firstBack, 0.35f ) };
    const otk::pbrt::PbrtMaterial secondMaterial{ mixMaterial( secondFront, secondBack, 0.35f ) };
    const otk::pbrt::PbrtMaterial amountMaterial{ mixMaterial( firstFront, firstBack, 0.75f ) };
    const otk::pbrt::PbrtMaterial rgbAmountMaterial{ mixMaterial( firstFront, firstBack, BoundMdlColor{ 0.2f, 0.4f, 0.6f } ) };
    const otk::pbrt::PbrtMaterial amountTextureMaterial{ mixMaterialWithAmountTexture( firstFront, firstBack, 0.35f ) };
    const otk::pbrt::PbrtMaterial namedUberMaterial{ mixMaterialWithNamedUberIndex( firstFront, firstBack, 0.35f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey amountKey{ demandPbrtScene::makeMdlShaderKey( amountMaterial ) };
    const demandPbrtScene::MdlShaderKey rgbAmountKey{ demandPbrtScene::makeMdlShaderKey( rgbAmountMaterial ) };
    const demandPbrtScene::MdlShaderKey amountTextureKey{ demandPbrtScene::makeMdlShaderKey( amountTextureMaterial ) };
    const demandPbrtScene::MdlShaderKey namedUberKey{ demandPbrtScene::makeMdlShaderKey( namedUberMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const demandPbrtScene::GeneratedMdlSource& amountTextureGenerated{ sourceCache.getSource( amountTextureMaterial ) };
    const demandPbrtScene::GeneratedMdlSource& namedUberGenerated{ sourceCache.getSource( namedUberMaterial ) };
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> amountParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( amountMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> rgbAmountParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( rgbAmountMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> amountTextureParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( amountTextureMaterial ) };
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter> namedUberParameters{
        demandPbrtScene::makeMdlBoundMaterialParameters( namedUberMaterial ) };

    CompiledMaterialHandle firstCompiledMaterial( compileMaterial( generated, firstKey, firstParameters ) );
    ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle secondCompiledMaterial( compileMaterial( generated, firstKey, secondParameters ) );
    ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle amountCompiledMaterial( compileMaterial( generated, firstKey, amountParameters ) );
    ASSERT_TRUE( amountCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle rgbAmountCompiledMaterial( compileMaterial( generated, firstKey, rgbAmountParameters ) );
    ASSERT_TRUE( rgbAmountCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle amountTextureCompiledMaterial(
        compileMaterial( amountTextureGenerated, amountTextureKey, amountTextureParameters ) );
    ASSERT_TRUE( amountTextureCompiledMaterial.is_valid_interface() );
    CompiledMaterialHandle namedUberCompiledMaterial( compileMaterial( namedUberGenerated, namedUberKey, namedUberParameters ) );
    ASSERT_TRUE( namedUberCompiledMaterial.is_valid_interface() );
    const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_mix_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_mix_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx amountBsdf{
        demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), amountCompiledMaterial.get(),
                                                       context.get(), "surface.scattering", "pbrt_mix_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx rgbAmountBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), rgbAmountCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_mix_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx amountTextureBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), amountTextureCompiledMaterial.get(), context.get(),
        "surface.scattering", "pbrt_mix_bsdf" ) };
    const demandPbrtScene::MdlBsdfCallablePtx namedUberBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
        session.neuray(), transaction.get(), namedUberCompiledMaterial.get(), context.get(), "surface.scattering",
        "pbrt_mix_uber_bsdf" ) };

    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( amountKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( rgbAmountKey ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_bsdf_component[]" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "weight: amount" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "weight: color(1.0, 1.0, 1.0) - amount" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "component: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "pbrt_mix_approximation_tint" ) ) );
    EXPECT_THAT( namedUberGenerated.source, testing::Not( testing::HasSubstr( "named_0_index" ) ) );
    EXPECT_EQ( "pbrt_mix_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_mix_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_mix_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
    EXPECT_EQ( "pbrt_mix_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
    EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
    EXPECT_FALSE( secondBsdf.ptx.empty() );
    EXPECT_FALSE( amountBsdf.ptx.empty() );
    EXPECT_FALSE( rgbAmountBsdf.ptx.empty() );
    EXPECT_FALSE( amountTextureBsdf.ptx.empty() );
    EXPECT_FALSE( namedUberBsdf.ptx.empty() );
    EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, amountBsdf.ptx );
    EXPECT_NE( firstBsdf.ptx, rgbAmountBsdf.ptx );

    firstCompiledMaterial.reset();
    secondCompiledMaterial.reset();
    amountCompiledMaterial.reset();
    rgbAmountCompiledMaterial.reset();
    amountTextureCompiledMaterial.reset();
    namedUberCompiledMaterial.reset();
}

TEST_F( TestMdlSdk, compilesOpaqueGeneratedMaterialsWithBoundConstants )
{
    demandPbrtScene::MdlGeneratedSourceCache sourceCache;
    const auto expectCompiledTint = [&]( const otk::pbrt::PbrtMaterial& material, const BoundMdlColor& expected ) {
        const demandPbrtScene::MdlShaderKey        key{ demandPbrtScene::makeMdlShaderKey( material ) };
        const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( material ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> parameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( material ) };

        CompiledMaterialHandle compiledMaterial( compileMaterial( generated, key, parameters ) );
        ASSERT_TRUE( compiledMaterial.is_valid_interface() );

        expectTintMatchesColor( compiledMaterial.get(), expected );
    };

    expectCompiledTint( mirrorMaterial(), BoundMdlColor{ 0.2f, 0.3f, 0.4f } );
}
