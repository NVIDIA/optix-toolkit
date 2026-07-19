// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <gmock/gmock.h>

#include "DemandPbrtScene/MdlBsdfCompiler.h"
#include "DemandPbrtScene/MdlShaderCache.h"

#include <mi/mdl_sdk.h>

#ifdef _WIN32
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif

#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr mi::Float32 PBRT_KD_RED       = 0.25f;
constexpr mi::Float32 PBRT_KD_GREEN     = 0.50f;
constexpr mi::Float32 PBRT_KD_BLUE      = 0.75f;
constexpr mi::Float32 PBRT_KD_ALT_RED   = 0.75f;
constexpr mi::Float32 PBRT_KD_ALT_GREEN = 0.20f;
constexpr mi::Float32 PBRT_KD_ALT_BLUE  = 0.10f;

struct BoundMdlColor
{
    mi::Float32 red{};
    mi::Float32 green{};
    mi::Float32 blue{};
};

std::string describeContextMessages( const mi::neuraylib::IMdl_execution_context* context )
{
    if( !context )
        return {};

    std::ostringstream out;
    for( mi::Size i = 0; i < context->get_messages_count(); ++i )
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message( context->get_message( i ) );
        if( message.is_valid_interface() )
            out << message->get_string() << '\n';
    }
    return out.str();
}

#ifdef _WIN32

using MdlLibraryHandle = HMODULE;

std::string lastLibraryError()
{
    std::ostringstream out;
    out << "Windows error " << GetLastError();
    return out.str();
}

MdlLibraryHandle loadMdlSdkLibrary( std::string& error )
{
    const char* const libraryName = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
    MdlLibraryHandle  handle      = LoadLibraryA( libraryName );
    if( handle )
        return handle;

    const std::string fallback = std::string( "../../../bin/" ) + libraryName;
    handle                     = LoadLibraryA( fallback.c_str() );
    if( handle )
        return handle;

    error = "Failed to load " + std::string( libraryName ) + ": " + lastLibraryError();
    return nullptr;
}

void* loadMdlFactorySymbol( MdlLibraryHandle handle, std::string& error )
{
    void* symbol = GetProcAddress( handle, "mi_factory" );
    if( !symbol )
        error = "Failed to find mi_factory: " + lastLibraryError();
    return symbol;
}

void unloadMdlSdkLibrary( MdlLibraryHandle handle )
{
    if( handle )
        FreeLibrary( handle );
}

#else

using MdlLibraryHandle = void*;

MdlLibraryHandle loadMdlSdkLibrary( std::string& error )
{
    const char* const libraryName = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
    MdlLibraryHandle  handle      = dlopen( libraryName, RTLD_LAZY );
    if( !handle )
        error = dlerror();
    return handle;
}

void* loadMdlFactorySymbol( MdlLibraryHandle handle, std::string& error )
{
    void* symbol = dlsym( handle, "mi_factory" );
    if( !symbol )
        error = dlerror();
    return symbol;
}

void unloadMdlSdkLibrary( MdlLibraryHandle handle )
{
    if( handle )
        dlclose( handle );
}

#endif

class MdlSdkSession
{
  public:
    MdlSdkSession()
        : m_library( loadMdlSdkLibrary( m_error ) )
    {
        if( !m_library )
            return;

        void* symbol = loadMdlFactorySymbol( m_library, m_error );
        if( !symbol )
            return;

        m_neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>( symbol );
        if( !m_neuray.is_valid_interface() )
        {
            mi::base::Handle<const mi::neuraylib::IVersion> version( mi::neuraylib::mi_factory<mi::neuraylib::IVersion>( symbol ) );
            m_error = version.is_valid_interface() ? "MDL SDK library version does not match header version "
                                                         + std::string( MI_NEURAYLIB_PRODUCT_VERSION_STRING ) :
                                                     "MDL SDK library is incompatible with this header";
            return;
        }

        const mi::Sint32 startResult = m_neuray->start( true );
        if( startResult != 0 )
        {
            std::ostringstream out;
            out << "Failed to start MDL SDK: " << startResult;
            m_error = out.str();
            return;
        }

        m_started = true;
    }

    ~MdlSdkSession()
    {
        shutdown();
        unloadMdlSdkLibrary( m_library );
    }

    bool isStarted() const { return m_started; }

    const std::string& error() const { return m_error; }

    mi::neuraylib::INeuray* neuray() const { return m_neuray.get(); }

    mi::Sint32 shutdown()
    {
        mi::Sint32 result = 0;
        if( m_started )
        {
            result    = m_neuray->shutdown( true );
            m_started = false;
        }
        m_neuray.reset();
        return result;
    }

  private:
    MdlLibraryHandle                         m_library{};
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    std::string                              m_error;
    bool                                     m_started{ false };
};

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

otk::pbrt::PbrtMaterial matteMaterial( float red, float green, float blue )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "matte";
    addRgbSpectrum( material.params, "Kd", red, green, blue );
    return material;
}

otk::pbrt::PbrtMaterial matteMaterialWithSigma( float red, float green, float blue, float sigma )
{
    otk::pbrt::PbrtMaterial material{ matteMaterial( red, green, blue ) };
    addFloat( material.params, "sigma", sigma );
    return material;
}

otk::pbrt::PbrtMaterial plasticMaterial()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "plastic";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addFloat( material.params, "roughness", 0.25f );
    return material;
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

otk::pbrt::PbrtMaterial substrateMaterial()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "substrate";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "uroughness", 0.2f );
    addFloat( material.params, "vroughness", 0.3f );
    return material;
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

otk::pbrt::PbrtMaterial metalMaterial()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "metal";
    addRgbSpectrum( material.params, "eta", 0.2f, 0.3f, 0.45f );
    addRgbSpectrum( material.params, "k", 2.2f, 2.8f, 3.4f );
    addFloat( material.params, "roughness", 0.18f );
    addFloat( material.params, "uroughness", 0.16f );
    addFloat( material.params, "vroughness", 0.2f );
    return material;
}

otk::pbrt::PbrtMaterial translucentMaterial()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "translucent";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addRgbSpectrum( material.params, "reflect", 0.8f, 0.6f, 0.4f );
    addRgbSpectrum( material.params, "transmit", 0.2f, 0.4f, 0.6f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "opacity", 0.7f );
    return material;
}

otk::pbrt::PbrtNamedMaterial namedMatteMaterial( const std::string& name, float kd )
{
    otk::pbrt::PbrtNamedMaterial material;
    material.name = name;
    material.type = "matte";
    addString( material.params, "type", "matte" );
    addRgbSpectrum( material.params, "Kd", kd, kd, kd );
    return material;
}

otk::pbrt::PbrtMaterial mixMaterial()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", "front" );
    addString( material.params, "namedmaterial2", "back" );
    addFloat( material.params, "amount", 0.25f );
    material.graph.namedMaterials["front"] = namedMatteMaterial( "front", 0.2f );
    material.graph.namedMaterials["back"]  = namedMatteMaterial( "back", 0.8f );
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

otk::pbrt::PbrtMaterial mixMaterialWithAmountTexture()
{
    otk::pbrt::PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", "front" );
    addString( material.params, "namedmaterial2", "back" );
    material.params.AddTexture( "amount", "weight" );
    material.graph.namedMaterials["front"]  = namedMatteMaterial( "front", 0.2f );
    material.graph.namedMaterials["back"]   = namedMatteMaterial( "back", 0.8f );
    material.graph.textures["float:weight"] = constantFloatTexture( "weight", 0.25f );
    return material;
}

std::string describeGeneratedSource( const demandPbrtScene::GeneratedMdlSource& source, const demandPbrtScene::MdlShaderKey& key )
{
    return "module=" + source.moduleName + ", material=" + source.materialName + ", key=" + demandPbrtScene::toString( key );
}

mi::base::Handle<mi::neuraylib::ICompiled_material> compileGeneratedMaterialWithBoundParameters(
    mi::neuraylib::INeuray*                                        neuray,
    mi::neuraylib::ITransaction*                                   transaction,
    mi::neuraylib::IMdl_execution_context*                         context,
    const demandPbrtScene::GeneratedMdlSource&                     source,
    const demandPbrtScene::MdlShaderKey&                           key,
    const std::vector<demandPbrtScene::MdlBoundMaterialParameter>& parameters )
{
    const std::string sourceDescription{ describeGeneratedSource( source, key ) };
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( neuray->get_api_component<mi::neuraylib::IMdl_factory>() );
    EXPECT_TRUE( mdlFactory.is_valid_interface() ) << sourceDescription;
    if( !mdlFactory.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdlImpexpApi( neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>() );
    EXPECT_TRUE( mdlImpexpApi.is_valid_interface() ) << sourceDescription;
    if( !mdlImpexpApi.is_valid_interface() )
        return {};

    mi::base::Handle<const mi::IString> moduleDbName( mdlFactory->get_db_module_name( source.moduleName.c_str() ) );
    EXPECT_TRUE( moduleDbName.is_valid_interface() ) << sourceDescription;
    if( !moduleDbName.is_valid_interface() )
        return {};

    mi::base::Handle<const mi::neuraylib::IModule> module( transaction->access<mi::neuraylib::IModule>( moduleDbName->get_c_str() ) );
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

    mi::base::Handle<const mi::neuraylib::IFunction_definition> materialDefinition(
        transaction->access<mi::neuraylib::IFunction_definition>( materialDbName ) );
    EXPECT_TRUE( materialDefinition.is_valid_interface() ) << sourceDescription;
    if( !materialDefinition.is_valid_interface() )
        return {};

    mi::Sint32 callResult = 0;
    mi::base::Handle<mi::neuraylib::IFunction_call> materialCall( materialDefinition->create_function_call( nullptr, &callResult ) );
    EXPECT_EQ( 0, callResult ) << sourceDescription;
    EXPECT_TRUE( materialCall.is_valid_interface() ) << sourceDescription;
    if( !materialCall.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IValue_factory> valueFactory( mdlFactory->create_value_factory( transaction ) );
    EXPECT_TRUE( valueFactory.is_valid_interface() ) << sourceDescription;
    if( !valueFactory.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IExpression_factory> expressionFactory( mdlFactory->create_expression_factory( transaction ) );
    EXPECT_TRUE( expressionFactory.is_valid_interface() ) << sourceDescription;
    if( !expressionFactory.is_valid_interface() )
        return {};

    for( const demandPbrtScene::MdlBoundMaterialParameter& parameter : parameters )
    {
        if( parameter.type == demandPbrtScene::MdlBoundParameterType::COLOR )
        {
            mi::base::Handle<mi::neuraylib::IValue_color> value(
                valueFactory->create_color( parameter.red, parameter.green, parameter.blue ) );
            EXPECT_TRUE( value.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !value.is_valid_interface() )
                return {};

            mi::base::Handle<mi::neuraylib::IExpression_constant> expression( expressionFactory->create_constant( value.get() ) );
            EXPECT_TRUE( expression.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !expression.is_valid_interface() )
                return {};

            EXPECT_EQ( 0, materialCall->set_argument( parameter.name.c_str(), expression.get() ) )
                << sourceDescription << ", parameter=" << parameter.name;
        }
        else
        {
            mi::base::Handle<mi::neuraylib::IValue_float> value( valueFactory->create_float( parameter.value ) );
            EXPECT_TRUE( value.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !value.is_valid_interface() )
                return {};

            mi::base::Handle<mi::neuraylib::IExpression_constant> expression( expressionFactory->create_constant( value.get() ) );
            EXPECT_TRUE( expression.is_valid_interface() ) << sourceDescription << ", parameter=" << parameter.name;
            if( !expression.is_valid_interface() )
                return {};

            EXPECT_EQ( 0, materialCall->set_argument( parameter.name.c_str(), expression.get() ) )
                << sourceDescription << ", parameter=" << parameter.name;
        }
    }

    mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance(
        materialCall->get_interface<mi::neuraylib::IMaterial_instance>() );
    EXPECT_TRUE( materialInstance.is_valid_interface() ) << sourceDescription;
    if( !materialInstance.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IType_factory> typeFactory( mdlFactory->create_type_factory( transaction ) );
    EXPECT_TRUE( typeFactory.is_valid_interface() ) << sourceDescription;
    if( !typeFactory.is_valid_interface() )
        return {};

    mi::base::Handle<const mi::neuraylib::IType> standardMaterialType(
        typeFactory->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL ) );
    EXPECT_TRUE( standardMaterialType.is_valid_interface() ) << sourceDescription;
    if( !standardMaterialType.is_valid_interface() )
        return {};

    context->clear_messages();
    const mi::Sint32 targetTypeResult = context->set_option( "target_type", standardMaterialType.get() );
    EXPECT_EQ( 0, targetTypeResult ) << sourceDescription << '\n' << describeContextMessages( context );
    if( targetTypeResult != 0 )
        return {};

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(
        materialInstance->create_compiled_material( mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context ) );
    EXPECT_TRUE( compiledMaterial.is_valid_interface() ) << sourceDescription << '\n'
                                                         << describeContextMessages( context );
    return compiledMaterial;
}

void expectColorExpressionMatches( const mi::neuraylib::ICompiled_material* compiledMaterial,
                                   const char*                              expressionPath,
                                   const BoundMdlColor&                     expected )
{
    mi::base::Handle<const mi::neuraylib::IExpression> tintExpression( compiledMaterial->lookup_sub_expression( expressionPath ) );
    ASSERT_TRUE( tintExpression.is_valid_interface() );
    ASSERT_EQ( mi::neuraylib::IExpression::EK_CONSTANT, tintExpression->get_kind() );

    mi::base::Handle<const mi::neuraylib::IExpression_constant> tintConstant(
        tintExpression->get_interface<mi::neuraylib::IExpression_constant>() );
    ASSERT_TRUE( tintConstant.is_valid_interface() );

    mi::base::Handle<const mi::neuraylib::IValue_color> tintValue( tintConstant->get_value<mi::neuraylib::IValue_color>() );
    ASSERT_TRUE( tintValue.is_valid_interface() );

    mi::base::Handle<const mi::neuraylib::IValue_float> red( tintValue->get_value( 0 ) );
    mi::base::Handle<const mi::neuraylib::IValue_float> green( tintValue->get_value( 1 ) );
    mi::base::Handle<const mi::neuraylib::IValue_float> blue( tintValue->get_value( 2 ) );
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

void expectIorMatchesFloat( const mi::neuraylib::ICompiled_material* compiledMaterial, float index )
{
    expectColorExpressionMatches( compiledMaterial, "ior", BoundMdlColor{ index, index, index } );
}

const char* findPreviewColorExpressionPath( const mi::neuraylib::ICompiled_material* compiledMaterial )
{
    static const char* const paths[] = { "surface.scattering.tint", "ior" };
    for( const char* path : paths )
    {
        mi::base::Handle<const mi::neuraylib::IExpression> expression( compiledMaterial->lookup_sub_expression( path ) );
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
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi( neuray->get_api_component<mi::neuraylib::IMdl_backend_api>() );
    EXPECT_TRUE( backendApi.is_valid_interface() );
    if( !backendApi.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IMdl_backend> ptxBackend( backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
    EXPECT_TRUE( ptxBackend.is_valid_interface() );
    if( !ptxBackend.is_valid_interface() )
        return {};

    context->clear_messages();
    const char* const previewColorExpressionPath{ findPreviewColorExpressionPath( compiledMaterial ) };
    if( previewColorExpressionPath[0] == '\0' )
    {
        return {};
    }
    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode( ptxBackend->translate_material_expression(
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

}  // namespace

TEST( TestMdlSdk, headerProvidesVersionMetadata )
{
    EXPECT_GT( std::strlen( MI_NEURAYLIB_PRODUCT_VERSION_STRING ), 0U );
    EXPECT_GT( MI_NEURAYLIB_API_VERSION, 0 );
}

TEST( TestMdlSdk, headerProvidesNeurayInterfaceId )
{
    const mi::base::Uuid id = mi::neuraylib::INeuray::IID();

    EXPECT_NE( 0U, id.m_id1 | id.m_id2 | id.m_id3 | id.m_id4 );
}

TEST( TestMdlSdk, compilesGeneratedMatteMaterialWithBoundKd )
{
    const otk::pbrt::PbrtMaterial       sourceMaterial{ matteMaterial( PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE ) };
    const demandPbrtScene::MdlShaderKey key{ demandPbrtScene::makeMdlShaderKey( sourceMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( sourceMaterial ) };
    const std::string                          sourceDescription{ describeGeneratedSource( generated, key ) };
    EXPECT_THAT( sourceDescription, testing::HasSubstr( generated.moduleName ) );
    EXPECT_THAT( sourceDescription, testing::HasSubstr( generated.materialName ) );
    EXPECT_THAT( sourceDescription, testing::HasSubstr( demandPbrtScene::toString( key ) ) );
    EXPECT_THAT( sourceDescription, testing::Not( testing::HasSubstr( ":\\" ) ) );

    MdlSdkSession session;
    ASSERT_TRUE( session.isStarted() ) << session.error();

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database( session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
        ASSERT_TRUE( database.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
        ASSERT_TRUE( scope.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
        ASSERT_TRUE( transaction.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        ASSERT_TRUE( mdlFactory.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context( mdlFactory->create_execution_context() );
        ASSERT_TRUE( context.is_valid_interface() );

        const BoundMdlColor firstKd{ PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE };
        const BoundMdlColor secondKd{ PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( sourceMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{ demandPbrtScene::makeMdlBoundMaterialParameters(
            matteMaterial( PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE ) ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, key, firstParameters ) );
        ASSERT_TRUE( compiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> secondCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, key, secondParameters ) );
        ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );

        expectTintMatchesPbrtKd( compiledMaterial.get(), firstKd );
        expectTintMatchesPbrtKd( secondCompiledMaterial.get(), secondKd );
        const std::string firstPtx{
            translateTintExpressionToPtx( session.neuray(), transaction.get(), compiledMaterial.get(), context.get() ) };
        const std::string secondPtx{ translateTintExpressionToPtx( session.neuray(), transaction.get(),
                                                                   secondCompiledMaterial.get(), context.get() ) };
        EXPECT_FALSE( firstPtx.empty() );
        EXPECT_FALSE( secondPtx.empty() );
        EXPECT_NE( firstPtx, secondPtx );

        secondCompiledMaterial.reset();
        compiledMaterial.reset();
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedMatteBsdfCallablesWithBoundKd )
{
    const otk::pbrt::PbrtMaterial firstMaterial{ matteMaterial( PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE ) };
    const otk::pbrt::PbrtMaterial secondMaterial{ matteMaterial( PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE ) };
    const otk::pbrt::PbrtMaterial roughMaterial{ matteMaterialWithSigma( PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE, 45.0f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const std::string                          sourceDescription{ describeGeneratedSource( generated, firstKey ) };

    MdlSdkSession session;
    ASSERT_TRUE( session.isStarted() ) << session.error();

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database( session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
        ASSERT_TRUE( database.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
        ASSERT_TRUE( scope.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
        ASSERT_TRUE( transaction.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        ASSERT_TRUE( mdlFactory.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context( mdlFactory->create_execution_context() );
        ASSERT_TRUE( context.is_valid_interface() );

        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> firstCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, firstParameters ) );
        ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> secondCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, secondParameters ) );
        ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> roughCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, roughParameters ) );
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
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedMirrorBsdfCallablesWithBoundKr )
{
    const otk::pbrt::PbrtMaterial       material{ mirrorMaterial() };
    const demandPbrtScene::MdlShaderKey key{ demandPbrtScene::makeMdlShaderKey( material ) };

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( material ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::specular_bsdf" ) );
    const std::string sourceDescription{ describeGeneratedSource( generated, key ) };

    MdlSdkSession session;
    ASSERT_TRUE( session.isStarted() ) << session.error();

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database( session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
        ASSERT_TRUE( database.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
        ASSERT_TRUE( scope.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
        ASSERT_TRUE( transaction.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        ASSERT_TRUE( mdlFactory.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context( mdlFactory->create_execution_context() );
        ASSERT_TRUE( context.is_valid_interface() );

        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> parameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( material ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, key, parameters ) );
        ASSERT_TRUE( compiledMaterial.is_valid_interface() );

        const demandPbrtScene::MdlBsdfCallablePtx bsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), compiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_mirror_bsdf" ) };

        EXPECT_EQ( "pbrt_mirror_bsdf_init", bsdf.initFunctionName ) << sourceDescription;
        EXPECT_EQ( "pbrt_mirror_bsdf_sample", bsdf.sampleFunctionName ) << sourceDescription;
        EXPECT_EQ( "pbrt_mirror_bsdf_evaluate", bsdf.evaluateFunctionName ) << sourceDescription;
        EXPECT_EQ( "pbrt_mirror_bsdf_pdf", bsdf.pdfFunctionName ) << sourceDescription;
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.initFunctionName ) );
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.sampleFunctionName ) );
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.evaluateFunctionName ) );
        EXPECT_THAT( bsdf.ptx, testing::HasSubstr( bsdf.pdfFunctionName ) );

        compiledMaterial.reset();
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedGlassBsdfCallablesWithBoundDielectricInputs )
{
    const otk::pbrt::PbrtMaterial       firstMaterial{ glassMaterial() };
    const otk::pbrt::PbrtMaterial       secondMaterial{ glassMaterial( 1.1f, 0.2f, 0.3f, 0.4f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::specular_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_reflect_transmit" ) );
    const std::string sourceDescription{ describeGeneratedSource( generated, firstKey ) };

    MdlSdkSession session;
    ASSERT_TRUE( session.isStarted() ) << session.error();

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database( session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
        ASSERT_TRUE( database.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
        ASSERT_TRUE( scope.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
        ASSERT_TRUE( transaction.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        ASSERT_TRUE( mdlFactory.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context( mdlFactory->create_execution_context() );
        ASSERT_TRUE( context.is_valid_interface() );

        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> firstParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( firstMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( secondMaterial ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> firstCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, firstParameters ) );
        ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> secondCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, secondParameters ) );
        ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );
        expectIorMatchesFloat( firstCompiledMaterial.get(), 1.5f );
        expectIorMatchesFloat( secondCompiledMaterial.get(), 1.1f );

        const std::string tintPtx{ translateTintExpressionToPtx( session.neuray(), transaction.get(),
                                                                 firstCompiledMaterial.get(), context.get() ) };
        EXPECT_FALSE( tintPtx.empty() );

        const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_glass_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_glass_bsdf" ) };

        EXPECT_EQ( "pbrt_glass_bsdf_init", firstBsdf.initFunctionName ) << sourceDescription;
        EXPECT_EQ( "pbrt_glass_bsdf_sample", firstBsdf.sampleFunctionName ) << sourceDescription;
        EXPECT_EQ( "pbrt_glass_bsdf_evaluate", firstBsdf.evaluateFunctionName ) << sourceDescription;
        EXPECT_EQ( "pbrt_glass_bsdf_pdf", firstBsdf.pdfFunctionName ) << sourceDescription;
        EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.initFunctionName ) );
        EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.sampleFunctionName ) );
        EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.evaluateFunctionName ) );
        EXPECT_THAT( firstBsdf.ptx, testing::HasSubstr( firstBsdf.pdfFunctionName ) );
        EXPECT_FALSE( secondBsdf.ptx.empty() );

        firstCompiledMaterial.reset();
        secondCompiledMaterial.reset();
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesOpaqueGeneratedMaterialsWithBoundConstants )
{
    MdlSdkSession session;
    ASSERT_TRUE( session.isStarted() ) << session.error();

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database( session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
        ASSERT_TRUE( database.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
        ASSERT_TRUE( scope.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
        ASSERT_TRUE( transaction.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        ASSERT_TRUE( mdlFactory.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context( mdlFactory->create_execution_context() );
        ASSERT_TRUE( context.is_valid_interface() );

        demandPbrtScene::MdlGeneratedSourceCache sourceCache;
        const auto expectCompiledTint = [&]( const otk::pbrt::PbrtMaterial& material, const BoundMdlColor& expected ) {
            const demandPbrtScene::MdlShaderKey        key{ demandPbrtScene::makeMdlShaderKey( material ) };
            const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( material ) };
            const std::vector<demandPbrtScene::MdlBoundMaterialParameter> parameters{
                demandPbrtScene::makeMdlBoundMaterialParameters( material ) };
            mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial( compileGeneratedMaterialWithBoundParameters(
                session.neuray(), transaction.get(), context.get(), generated, key, parameters ) );
            ASSERT_TRUE( compiledMaterial.is_valid_interface() );
            expectTintMatchesColor( compiledMaterial.get(), expected );
        };

        expectCompiledTint( plasticMaterial(), BoundMdlColor{ 0.2f, 0.3f, 0.4f } );
        expectCompiledTint( uberMaterial(), BoundMdlColor{ 0.2f, 0.3f, 0.4f } );
        expectCompiledTint( substrateMaterial(), BoundMdlColor{ 0.2f, 0.3f, 0.4f } );
        expectCompiledTint( mirrorMaterial(), BoundMdlColor{ 0.2f, 0.3f, 0.4f } );
        expectCompiledTint( metalMaterial(), BoundMdlColor{ 2.2f / 2.4f, 2.8f / 3.1f, 3.4f / 3.85f } );
        expectCompiledTint( translucentMaterial(), BoundMdlColor{ 0.1f, 0.15f, 0.2f } );
        expectCompiledTint( mixMaterial(), BoundMdlColor{ 0.35f, 0.35f, 0.35f } );
        expectCompiledTint( mixMaterialWithAmountTexture(), BoundMdlColor{ 0.35f, 0.35f, 0.35f } );

        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}
