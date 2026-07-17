// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <mi/mdl_sdk.h>

#ifdef _WIN32
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif

#include <gtest/gtest.h>

#include <cstring>
#include <sstream>
#include <string>

namespace {

constexpr mi::Float32 PBRT_KD_RED   = 0.25f;
constexpr mi::Float32 PBRT_KD_GREEN = 0.50f;
constexpr mi::Float32 PBRT_KD_BLUE  = 0.75f;

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
    MdlLibraryHandle handle       = LoadLibraryA( libraryName );
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
    MdlLibraryHandle handle      = dlopen( libraryName, RTLD_LAZY );
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
            mi::base::Handle<const mi::neuraylib::IVersion> version(
                mi::neuraylib::mi_factory<mi::neuraylib::IVersion>( symbol ) );
            m_error = version.is_valid_interface()
                          ? "MDL SDK library version does not match header version "
                                + std::string( MI_NEURAYLIB_PRODUCT_VERSION_STRING )
                          : "MDL SDK library is incompatible with this header";
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
    MdlLibraryHandle m_library{};
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    std::string m_error;
    bool m_started{ false };
};

std::string makeDiffuseMdlModuleFromPbrtKd()
{
    std::ostringstream out;
    out << "mdl 1.6;\n"
        << "import ::df::*;\n"
        << "export material pbrt_diffuse() = material(\n"
        << "    surface: material_surface(\n"
        << "        scattering: ::df::diffuse_reflection_bsdf(\n"
        << "            tint: color(" << PBRT_KD_RED << ", " << PBRT_KD_GREEN << ", " << PBRT_KD_BLUE
        << "))));\n";
    return out.str();
}

mi::base::Handle<mi::neuraylib::ICompiled_material> compileGeneratedDiffuseMaterial(
    mi::neuraylib::INeuray* neuray,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_execution_context* context )
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>() );
    EXPECT_TRUE( mdlFactory.is_valid_interface() );
    if( !mdlFactory.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdlImpexpApi(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>() );
    EXPECT_TRUE( mdlImpexpApi.is_valid_interface() );
    if( !mdlImpexpApi.is_valid_interface() )
        return {};

    const std::string moduleSource = makeDiffuseMdlModuleFromPbrtKd();
    const char* const moduleName   = "::otk::pbrt_generated_test";
    context->clear_messages();
    const mi::Sint32 loadResult =
        mdlImpexpApi->load_module_from_string( transaction, moduleName, moduleSource.c_str(), context );
    EXPECT_EQ( 0, loadResult ) << describeContextMessages( context );
    if( loadResult != 0 )
        return {};

    mi::base::Handle<const mi::IString> moduleDbName( mdlFactory->get_db_module_name( moduleName ) );
    EXPECT_TRUE( moduleDbName.is_valid_interface() );
    if( !moduleDbName.is_valid_interface() )
        return {};

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>( moduleDbName->get_c_str() ) );
    EXPECT_TRUE( module.is_valid_interface() );
    if( !module.is_valid_interface() )
        return {};

    EXPECT_EQ( 1U, module->get_material_count() );
    const char* const materialDbName = module->get_material( 0 );
    EXPECT_NE( nullptr, materialDbName );
    if( !materialDbName )
        return {};

    mi::base::Handle<const mi::neuraylib::IFunction_definition> materialDefinition(
        transaction->access<mi::neuraylib::IFunction_definition>( materialDbName ) );
    EXPECT_TRUE( materialDefinition.is_valid_interface() );
    if( !materialDefinition.is_valid_interface() )
        return {};

    mi::Sint32 callResult = 0;
    mi::base::Handle<mi::neuraylib::IFunction_call> materialCall(
        materialDefinition->create_function_call( nullptr, &callResult ) );
    EXPECT_EQ( 0, callResult );
    EXPECT_TRUE( materialCall.is_valid_interface() );
    if( !materialCall.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance(
        materialCall->get_interface<mi::neuraylib::IMaterial_instance>() );
    EXPECT_TRUE( materialInstance.is_valid_interface() );
    if( !materialInstance.is_valid_interface() )
        return {};

    mi::base::Handle<mi::neuraylib::IType_factory> typeFactory( mdlFactory->create_type_factory( transaction ) );
    EXPECT_TRUE( typeFactory.is_valid_interface() );
    if( !typeFactory.is_valid_interface() )
        return {};

    mi::base::Handle<const mi::neuraylib::IType> standardMaterialType(
        typeFactory->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL ) );
    EXPECT_TRUE( standardMaterialType.is_valid_interface() );
    if( !standardMaterialType.is_valid_interface() )
        return {};

    context->clear_messages();
    const mi::Sint32 targetTypeResult = context->set_option( "target_type", standardMaterialType.get() );
    EXPECT_EQ( 0, targetTypeResult ) << describeContextMessages( context );
    if( targetTypeResult != 0 )
        return {};

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(
        materialInstance->create_compiled_material( mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context ) );
    EXPECT_TRUE( compiledMaterial.is_valid_interface() ) << describeContextMessages( context );
    return compiledMaterial;
}

void expectTintMatchesPbrtKd( const mi::neuraylib::ICompiled_material* compiledMaterial )
{
    mi::base::Handle<const mi::neuraylib::IExpression> tintExpression(
        compiledMaterial->lookup_sub_expression( "surface.scattering.tint" ) );
    ASSERT_TRUE( tintExpression.is_valid_interface() );
    ASSERT_EQ( mi::neuraylib::IExpression::EK_CONSTANT, tintExpression->get_kind() );

    mi::base::Handle<const mi::neuraylib::IExpression_constant> tintConstant(
        tintExpression->get_interface<mi::neuraylib::IExpression_constant>() );
    ASSERT_TRUE( tintConstant.is_valid_interface() );

    mi::base::Handle<const mi::neuraylib::IValue_color> tintValue(
        tintConstant->get_value<mi::neuraylib::IValue_color>() );
    ASSERT_TRUE( tintValue.is_valid_interface() );

    mi::base::Handle<const mi::neuraylib::IValue_float> red( tintValue->get_value( 0 ) );
    mi::base::Handle<const mi::neuraylib::IValue_float> green( tintValue->get_value( 1 ) );
    mi::base::Handle<const mi::neuraylib::IValue_float> blue( tintValue->get_value( 2 ) );
    ASSERT_TRUE( red.is_valid_interface() );
    ASSERT_TRUE( green.is_valid_interface() );
    ASSERT_TRUE( blue.is_valid_interface() );

    EXPECT_FLOAT_EQ( PBRT_KD_RED, red->get_value() );
    EXPECT_FLOAT_EQ( PBRT_KD_GREEN, green->get_value() );
    EXPECT_FLOAT_EQ( PBRT_KD_BLUE, blue->get_value() );
}

void expectPtxGeneratedForTintExpression(
    mi::neuraylib::INeuray* neuray,
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* compiledMaterial,
    mi::neuraylib::IMdl_execution_context* context )
{
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>() );
    ASSERT_TRUE( backendApi.is_valid_interface() );

    mi::base::Handle<mi::neuraylib::IMdl_backend> ptxBackend(
        backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
    ASSERT_TRUE( ptxBackend.is_valid_interface() );

    context->clear_messages();
    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode(
        ptxBackend->translate_material_expression(
            transaction, compiledMaterial, "surface.scattering.tint", "evaluate_tint", context ) );
    ASSERT_TRUE( targetCode.is_valid_interface() ) << describeContextMessages( context );
    EXPECT_GT( targetCode->get_code_size(), 0U );
    ASSERT_EQ( 1U, targetCode->get_callable_function_count() );
    EXPECT_STREQ( "evaluate_tint", targetCode->get_callable_function( 0 ) );
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

TEST( TestMdlSdk, compilesGeneratedDiffuseMaterial )
{
    MdlSdkSession session;
    ASSERT_TRUE( session.isStarted() ) << session.error();

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
        ASSERT_TRUE( database.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
        ASSERT_TRUE( scope.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
        ASSERT_TRUE( transaction.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory(
            session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        ASSERT_TRUE( mdlFactory.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdlFactory->create_execution_context() );
        ASSERT_TRUE( context.is_valid_interface() );

        mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(
            compileGeneratedDiffuseMaterial( session.neuray(), transaction.get(), context.get() ) );
        ASSERT_TRUE( compiledMaterial.is_valid_interface() );

        expectTintMatchesPbrtKd( compiledMaterial.get() );
        expectPtxGeneratedForTintExpression(
            session.neuray(), transaction.get(), compiledMaterial.get(), context.get() );

        EXPECT_EQ( 0, transaction->commit() );
    }

    EXPECT_EQ( 0, session.shutdown() );
}
