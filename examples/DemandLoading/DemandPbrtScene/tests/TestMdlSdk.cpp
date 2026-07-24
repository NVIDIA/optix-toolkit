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

void expectFloatExpressionMatches( const mi::neuraylib::ICompiled_material* compiledMaterial, const char* expressionPath, float expected )
{
    mi::base::Handle<const mi::neuraylib::IExpression> expression( compiledMaterial->lookup_sub_expression( expressionPath ) );
    ASSERT_TRUE( expression.is_valid_interface() );
    ASSERT_EQ( mi::neuraylib::IExpression::EK_CONSTANT, expression->get_kind() );

    mi::base::Handle<const mi::neuraylib::IExpression_constant> constant(
        expression->get_interface<mi::neuraylib::IExpression_constant>() );
    ASSERT_TRUE( constant.is_valid_interface() );

    mi::base::Handle<const mi::neuraylib::IValue_float> value( constant->get_value<mi::neuraylib::IValue_float>() );
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

std::string translateNormalExpressionToPtx( mi::neuraylib::INeuray*                  neuray,
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
    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode(
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

TEST( TestMdlSdk, compilesGeneratedMatteMaterialWithFoldedKdTexture )
{
    const BoundMdlColor                        firstKd{ PBRT_KD_RED, PBRT_KD_GREEN, PBRT_KD_BLUE };
    const BoundMdlColor                        secondKd{ PBRT_KD_ALT_RED, PBRT_KD_ALT_GREEN, PBRT_KD_ALT_BLUE };
    const otk::pbrt::PbrtMaterial              sourceMaterial{ matteMaterialWithKdTexture( firstKd ) };
    const demandPbrtScene::MdlShaderKey        key{ demandPbrtScene::makeMdlShaderKey( sourceMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( sourceMaterial ) };
    const demandPbrtScene::GeneratedMdlSource& secondGenerated{ sourceCache.getSource( matteMaterialWithKdTexture( secondKd ) ) };
    EXPECT_EQ( generated.moduleName, secondGenerated.moduleName );
    EXPECT_EQ( generated.materialName, secondGenerated.materialName );
    EXPECT_EQ( generated.source, secondGenerated.source );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input Kd: Kd" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "// pbrt texture node:" ) ) );

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
            demandPbrtScene::makeMdlBoundMaterialParameters( sourceMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> secondParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( matteMaterialWithKdTexture( secondKd ) ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, key, firstParameters ) );
        ASSERT_TRUE( compiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> secondCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, key, secondParameters ) );
        ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );

        expectTintMatchesPbrtKd( compiledMaterial.get(), firstKd );
        expectTintMatchesPbrtKd( secondCompiledMaterial.get(), secondKd );

        compiledMaterial.reset();
        secondCompiledMaterial.reset();
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedMaterialWithRuntimeBumpmap )
{
    const otk::pbrt::PbrtMaterial              sourceMaterial{ matteMaterialWithBumpmap() };
    const demandPbrtScene::MdlShaderKey        key{ demandPbrtScene::makeMdlShaderKey( sourceMaterial ) };
    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( sourceMaterial ) };
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "pbrt_bump_normal" ) ) );
    EXPECT_THAT( generated.source,
                 testing::HasSubstr( "// pbrt material implementation: bumpmap is evaluated with runtime finite differences" ) );
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
            demandPbrtScene::makeMdlBoundMaterialParameters( sourceMaterial ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, key, parameters ) );
        ASSERT_TRUE( compiledMaterial.is_valid_interface() ) << sourceDescription;

        const std::string normalPtx{ translateNormalExpressionToPtx( session.neuray(), transaction.get(),
                                                                     compiledMaterial.get(), context.get() ) };
        EXPECT_FALSE( normalPtx.empty() ) << sourceDescription;
        EXPECT_THAT( normalPtx, testing::HasSubstr( "evaluate_normal" ) ) << sourceDescription;

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

TEST( TestMdlSdk, compilesGeneratedPlasticBsdfCallablesWithBoundDiffuseAndGlossyInputs )
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
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "component: ::df::diffuse_reflection_bsdf" ) );
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
                                                           context.get(), "surface.scattering", "pbrt_plastic_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_plastic_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), roughCompiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_plastic_bsdf" ) };

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
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedUberBsdfCallablesWithBoundDiffuseAndGlossyInputs )
{
    const BoundMdlColor firstKd{ 0.2f, 0.3f, 0.4f };
    const BoundMdlColor firstKs{ 0.5f, 0.6f, 0.7f };
    const BoundMdlColor firstKr{ 0.2f, 0.1f, 0.0f };
    const BoundMdlColor firstKt{ 0.1f, 0.2f, 0.3f };
    const BoundMdlColor secondKd{ 0.6f, 0.2f, 0.1f };
    const BoundMdlColor secondKs{ 0.1f, 0.2f, 0.3f };
    const BoundMdlColor secondKr{ 0.4f, 0.2f, 0.1f };
    const BoundMdlColor secondKt{ 0.3f, 0.1f, 0.2f };
    const otk::pbrt::PbrtMaterial firstMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.25f, 0.8f, 0.7f ) };
    const otk::pbrt::PbrtMaterial secondMaterial{ uberMaterial( secondKd, secondKs, secondKr, secondKt, 0.25f, 0.8f, 0.7f ) };
    const otk::pbrt::PbrtMaterial roughMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.45f, 0.8f, 0.7f ) };
    const otk::pbrt::PbrtMaterial opacityMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.25f, 0.8f, 0.35f ) };
    const otk::pbrt::PbrtMaterial alphaMaterial{ uberMaterial( firstKd, firstKs, firstKr, firstKt, 0.25f, 0.35f, 0.7f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    const demandPbrtScene::MdlShaderKey opacityKey{ demandPbrtScene::makeMdlShaderKey( opacityMaterial ) };
    const demandPbrtScene::MdlShaderKey alphaKey{ demandPbrtScene::makeMdlShaderKey( alphaMaterial ) };
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( opacityKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( alphaKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::specular_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_transmit" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "component: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_uber_resolved_roughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_uber_opacity_weight" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_uber_transparency_weight" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "cutout_opacity: alpha" ) );
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
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> opacityParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( opacityMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> alphaParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( alphaMaterial ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> firstCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, firstParameters ) );
        ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> secondCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, secondParameters ) );
        ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> roughCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, roughParameters ) );
        ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> opacityCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, opacityParameters ) );
        ASSERT_TRUE( opacityCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> alphaCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, alphaParameters ) );
        ASSERT_TRUE( alphaCompiledMaterial.is_valid_interface() );

        expectIorMatchesFloat( firstCompiledMaterial.get(), 1.4f );
        expectFloatExpressionMatches( firstCompiledMaterial.get(), "geometry.cutout_opacity", 0.8f );
        expectFloatExpressionMatches( opacityCompiledMaterial.get(), "geometry.cutout_opacity", 0.8f );
        expectFloatExpressionMatches( alphaCompiledMaterial.get(), "geometry.cutout_opacity", 0.35f );

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
        EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
        EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );
        EXPECT_NE( firstBsdf.ptx, opacityBsdf.ptx );

        firstCompiledMaterial.reset();
        secondCompiledMaterial.reset();
        roughCompiledMaterial.reset();
        opacityCompiledMaterial.reset();
        alphaCompiledMaterial.reset();
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedSubstrateBsdfCallablesWithBoundLayeredInputs )
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
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_weighted_layer" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "base: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_substrate_resolved_roughness" ) );
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

        const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), firstCompiledMaterial.get(), context.get(), "surface.scattering",
            "pbrt_substrate_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), secondCompiledMaterial.get(), context.get(), "surface.scattering",
            "pbrt_substrate_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), roughCompiledMaterial.get(), context.get(), "surface.scattering",
            "pbrt_substrate_bsdf" ) };

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

TEST( TestMdlSdk, compilesGeneratedMetalBsdfCallablesWithBoundConductorInputs )
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
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::microfacet_ggx_smith_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_metal_conductor_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_metal_resolved_roughness" ) );
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

        expectTintMatchesColor( firstCompiledMaterial.get(), conductorNormalReflectance( firstEta, firstK ) );
        expectTintMatchesColor( secondCompiledMaterial.get(), conductorNormalReflectance( secondEta, secondK ) );
        expectTintMatchesColor( roughCompiledMaterial.get(), conductorNormalReflectance( firstEta, firstK ) );

        const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), firstCompiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_metal_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), secondCompiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_metal_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{
            demandPbrtScene::compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), roughCompiledMaterial.get(),
                                                           context.get(), "surface.scattering", "pbrt_metal_bsdf" ) };

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
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedTranslucentBsdfCallablesWithBoundReflectionTransmissionAndOpacity )
{
    const BoundMdlColor firstKd{ 0.2f, 0.3f, 0.4f };
    const BoundMdlColor firstKs{ 0.5f, 0.6f, 0.7f };
    const BoundMdlColor firstReflect{ 0.8f, 0.6f, 0.4f };
    const BoundMdlColor firstTransmit{ 0.2f, 0.4f, 0.6f };
    const BoundMdlColor secondKd{ 0.6f, 0.2f, 0.1f };
    const BoundMdlColor secondKs{ 0.1f, 0.2f, 0.3f };
    const BoundMdlColor secondReflect{ 0.3f, 0.5f, 0.7f };
    const BoundMdlColor secondTransmit{ 0.7f, 0.4f, 0.2f };
    const otk::pbrt::PbrtMaterial firstMaterial{ translucentMaterial( firstKd, firstKs, firstReflect, firstTransmit, 0.25f, 0.7f ) };
    const otk::pbrt::PbrtMaterial secondMaterial{
        translucentMaterial( secondKd, secondKs, secondReflect, secondTransmit, 0.25f, 0.7f ) };
    const otk::pbrt::PbrtMaterial roughMaterial{ translucentMaterial( firstKd, firstKs, firstReflect, firstTransmit, 0.45f, 0.7f ) };
    const otk::pbrt::PbrtMaterial opacityMaterial{ translucentMaterial( firstKd, firstKs, firstReflect, firstTransmit, 0.25f, 0.35f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey roughKey{ demandPbrtScene::makeMdlShaderKey( roughMaterial ) };
    const demandPbrtScene::MdlShaderKey opacityKey{ demandPbrtScene::makeMdlShaderKey( opacityMaterial ) };
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( roughKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( opacityKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::diffuse_transmission_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_reflect" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::scatter_transmit" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "cutout_opacity: opacity" ) );
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
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> roughParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( roughMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> opacityParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( opacityMaterial ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> firstCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, firstParameters ) );
        ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> secondCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, secondParameters ) );
        ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> roughCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, roughParameters ) );
        ASSERT_TRUE( roughCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> opacityCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, opacityParameters ) );
        ASSERT_TRUE( opacityCompiledMaterial.is_valid_interface() );

        expectIorMatchesFloat( firstCompiledMaterial.get(), 1.5f );
        expectFloatExpressionMatches( firstCompiledMaterial.get(), "geometry.cutout_opacity", 0.7f );
        expectFloatExpressionMatches( opacityCompiledMaterial.get(), "geometry.cutout_opacity", 0.35f );

        const demandPbrtScene::MdlBsdfCallablePtx firstBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), firstCompiledMaterial.get(), context.get(), "surface.scattering",
            "pbrt_translucent_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx secondBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), secondCompiledMaterial.get(), context.get(), "surface.scattering",
            "pbrt_translucent_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx roughBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), roughCompiledMaterial.get(), context.get(), "surface.scattering",
            "pbrt_translucent_bsdf" ) };

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
        EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
        EXPECT_NE( firstBsdf.ptx, roughBsdf.ptx );

        firstCompiledMaterial.reset();
        secondCompiledMaterial.reset();
        roughCompiledMaterial.reset();
        opacityCompiledMaterial.reset();
        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}

TEST( TestMdlSdk, compilesGeneratedMixBsdfCallablesWithBoundNamedMaterialClosures )
{
    const BoundMdlColor           firstFront{ 0.8f, 0.2f, 0.1f };
    const BoundMdlColor           firstBack{ 0.1f, 0.3f, 0.9f };
    const BoundMdlColor           secondFront{ 0.2f, 0.7f, 0.4f };
    const BoundMdlColor           secondBack{ 0.8f, 0.1f, 0.3f };
    const otk::pbrt::PbrtMaterial firstMaterial{ mixMaterial( firstFront, firstBack, 0.35f ) };
    const otk::pbrt::PbrtMaterial secondMaterial{ mixMaterial( secondFront, secondBack, 0.35f ) };
    const otk::pbrt::PbrtMaterial amountMaterial{ mixMaterial( firstFront, firstBack, 0.75f ) };
    const otk::pbrt::PbrtMaterial amountTextureMaterial{ mixMaterialWithAmountTexture( firstFront, firstBack, 0.35f ) };
    const otk::pbrt::PbrtMaterial namedUberMaterial{ mixMaterialWithNamedUberIndex( firstFront, firstBack, 0.35f ) };
    const demandPbrtScene::MdlShaderKey firstKey{ demandPbrtScene::makeMdlShaderKey( firstMaterial ) };
    const demandPbrtScene::MdlShaderKey secondKey{ demandPbrtScene::makeMdlShaderKey( secondMaterial ) };
    const demandPbrtScene::MdlShaderKey amountKey{ demandPbrtScene::makeMdlShaderKey( amountMaterial ) };
    const demandPbrtScene::MdlShaderKey amountTextureKey{ demandPbrtScene::makeMdlShaderKey( amountTextureMaterial ) };
    const demandPbrtScene::MdlShaderKey namedUberKey{ demandPbrtScene::makeMdlShaderKey( namedUberMaterial ) };
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( secondKey ) );
    EXPECT_EQ( demandPbrtScene::toString( firstKey ), demandPbrtScene::toString( amountKey ) );

    demandPbrtScene::MdlGeneratedSourceCache   sourceCache;
    const demandPbrtScene::GeneratedMdlSource& generated{ sourceCache.getSource( firstMaterial ) };
    const demandPbrtScene::GeneratedMdlSource& amountTextureGenerated{ sourceCache.getSource( amountTextureMaterial ) };
    const demandPbrtScene::GeneratedMdlSource& namedUberGenerated{ sourceCache.getSource( namedUberMaterial ) };
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_bsdf_component[]" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "weight: color(amount, amount, amount)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "weight: color(1.0 - amount, 1.0 - amount, 1.0 - amount)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "component: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "pbrt_mix_approximation_tint" ) ) );
    EXPECT_THAT( namedUberGenerated.source, testing::Not( testing::HasSubstr( "named_0_index" ) ) );
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
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> amountParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( amountMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> amountTextureParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( amountTextureMaterial ) };
        const std::vector<demandPbrtScene::MdlBoundMaterialParameter> namedUberParameters{
            demandPbrtScene::makeMdlBoundMaterialParameters( namedUberMaterial ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> firstCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, firstParameters ) );
        ASSERT_TRUE( firstCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> secondCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, secondParameters ) );
        ASSERT_TRUE( secondCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> amountCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), generated, firstKey, amountParameters ) );
        ASSERT_TRUE( amountCompiledMaterial.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> amountTextureCompiledMaterial(
            compileGeneratedMaterialWithBoundParameters( session.neuray(), transaction.get(), context.get(),
                                                         amountTextureGenerated, amountTextureKey, amountTextureParameters ) );
        ASSERT_TRUE( amountTextureCompiledMaterial.is_valid_interface() );
        mi::base::Handle<mi::neuraylib::ICompiled_material> namedUberCompiledMaterial( compileGeneratedMaterialWithBoundParameters(
            session.neuray(), transaction.get(), context.get(), namedUberGenerated, namedUberKey, namedUberParameters ) );
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
        const demandPbrtScene::MdlBsdfCallablePtx amountTextureBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), amountTextureCompiledMaterial.get(), context.get(),
            "surface.scattering", "pbrt_mix_bsdf" ) };
        const demandPbrtScene::MdlBsdfCallablePtx namedUberBsdf{ demandPbrtScene::compileMdlBsdfCallablesToPtx(
            session.neuray(), transaction.get(), namedUberCompiledMaterial.get(), context.get(), "surface.scattering",
            "pbrt_mix_uber_bsdf" ) };

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
        EXPECT_FALSE( amountTextureBsdf.ptx.empty() );
        EXPECT_FALSE( namedUberBsdf.ptx.empty() );
        EXPECT_NE( firstBsdf.ptx, secondBsdf.ptx );
        EXPECT_NE( firstBsdf.ptx, amountBsdf.ptx );

        firstCompiledMaterial.reset();
        secondCompiledMaterial.reset();
        amountCompiledMaterial.reset();
        amountTextureCompiledMaterial.reset();
        namedUberCompiledMaterial.reset();
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

        expectCompiledTint( mirrorMaterial(), BoundMdlColor{ 0.2f, 0.3f, 0.4f } );

        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}
