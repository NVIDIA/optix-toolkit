// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/ProgramGroups.h"

#include "DemandPbrtScene/Config.h"
#include "DemandPbrtScene/DemandTextureCache.h"
#include "DemandPbrtScene/IdRangePrinter.h"
#include "DemandPbrtScene/ImageSourceFactory.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Renderer.h"
#include "DemandPbrtScene/SceneAdapters.h"
#include "DemandPbrtScene/SceneProxy.h"
#include "DemandPbrtScene/Stopwatch.h"

#include <DemandPbrtSceneKernelCuda.h>

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>

#include <optix_stubs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iterator>
#ifdef OTK_USE_MDL
#include <map>
#endif
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef OTK_USE_MDL
#include <mi/mdl_sdk.h>

#ifdef _WIN32
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif
#endif

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

namespace demandPbrtScene {

static OptixModuleCompileOptions getCompileOptions()
{
    OptixModuleCompileOptions compileOptions{};
    compileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    otk::configModuleCompileOptions( compileOptions );

    return compileOptions;
}

namespace {

#ifdef OTK_USE_MDL

constexpr const char* MDL_SMOKE_TINT_FUNCTION_NAME = "__direct_callable__mdlSmokeTint";

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

[[noreturn]] void failMdl( const std::string& message, const mi::neuraylib::IMdl_execution_context* context = nullptr )
{
    const std::string contextMessages{ describeContextMessages( context ) };
    throw std::runtime_error( contextMessages.empty() ? message : message + ":\n" + contextMessages );
}

void requireMdl( bool condition, const std::string& message, const mi::neuraylib::IMdl_execution_context* context = nullptr )
{
    if( !condition )
    {
        failMdl( message, context );
    }
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

std::string makeDiffuseMdlModuleFromPbrtKd( const float3& kd )
{
    std::ostringstream out;
    out << std::setprecision( 9 ) << "mdl 1.6;\n"
        << "import ::df::*;\n"
        << "export material pbrt_diffuse() = material(\n"
        << "    surface: material_surface(\n"
        << "        scattering: ::df::diffuse_reflection_bsdf(\n"
        << "            tint: color(" << kd.x << ", " << kd.y << ", " << kd.z << "))));\n";
    return out.str();
}

mi::base::Handle<mi::neuraylib::ICompiled_material> compileGeneratedDiffuseMaterial( mi::neuraylib::INeuray* neuray,
                                                                                     mi::neuraylib::ITransaction* transaction,
                                                                                     mi::neuraylib::IMdl_execution_context* context,
                                                                                     const float3& kd )
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( neuray->get_api_component<mi::neuraylib::IMdl_factory>() );
    requireMdl( mdlFactory.is_valid_interface(), "Failed to get MDL factory" );

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdlImpexpApi( neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>() );
    requireMdl( mdlImpexpApi.is_valid_interface(), "Failed to get MDL import/export API" );

    const std::string moduleSource = makeDiffuseMdlModuleFromPbrtKd( kd );
    const char* const moduleName   = "::otk::demand_pbrt_scene_smoke";
    context->clear_messages();
    const mi::Sint32 loadResult = mdlImpexpApi->load_module_from_string( transaction, moduleName, moduleSource.c_str(), context );
    requireMdl( loadResult == 0, "Failed to load generated MDL module", context );

    mi::base::Handle<const mi::IString> moduleDbName( mdlFactory->get_db_module_name( moduleName ) );
    requireMdl( moduleDbName.is_valid_interface(), "Failed to get generated MDL module DB name" );

    mi::base::Handle<const mi::neuraylib::IModule> module( transaction->access<mi::neuraylib::IModule>( moduleDbName->get_c_str() ) );
    requireMdl( module.is_valid_interface(), "Failed to access generated MDL module" );
    requireMdl( module->get_material_count() == 1U, "Generated MDL module did not contain exactly one material" );

    const char* const materialDbName = module->get_material( 0 );
    requireMdl( materialDbName != nullptr, "Generated MDL module had no material definition" );

    mi::base::Handle<const mi::neuraylib::IFunction_definition> materialDefinition(
        transaction->access<mi::neuraylib::IFunction_definition>( materialDbName ) );
    requireMdl( materialDefinition.is_valid_interface(), "Failed to access generated MDL material definition" );

    mi::Sint32 callResult = 0;
    mi::base::Handle<mi::neuraylib::IFunction_call> materialCall( materialDefinition->create_function_call( nullptr, &callResult ) );
    requireMdl( callResult == 0 && materialCall.is_valid_interface(), "Failed to create generated MDL material call" );

    mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance(
        materialCall->get_interface<mi::neuraylib::IMaterial_instance>() );
    requireMdl( materialInstance.is_valid_interface(), "Failed to create generated MDL material instance" );

    mi::base::Handle<mi::neuraylib::IType_factory> typeFactory( mdlFactory->create_type_factory( transaction ) );
    requireMdl( typeFactory.is_valid_interface(), "Failed to create MDL type factory" );

    mi::base::Handle<const mi::neuraylib::IType> standardMaterialType(
        typeFactory->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL ) );
    requireMdl( standardMaterialType.is_valid_interface(), "Failed to get MDL material type" );

    context->clear_messages();
    const mi::Sint32 targetTypeResult = context->set_option( "target_type", standardMaterialType.get() );
    requireMdl( targetTypeResult == 0, "Failed to set MDL target material type", context );

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(
        materialInstance->create_compiled_material( mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context ) );
    requireMdl( compiledMaterial.is_valid_interface(), "Failed to compile generated MDL material", context );
    return compiledMaterial;
}

std::string compileMdlSmokeTintPtx( const PhongMaterial& material )
{
    MdlSdkSession session;
    requireMdl( session.isStarted(), session.error() );

    mi::base::Handle<mi::neuraylib::IDatabase> database( session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
    requireMdl( database.is_valid_interface(), "Failed to get MDL database" );

    mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
    requireMdl( scope.is_valid_interface(), "Failed to get MDL global scope" );

    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
    requireMdl( transaction.is_valid_interface(), "Failed to create MDL transaction" );

    std::string ptx;
    {
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        requireMdl( mdlFactory.is_valid_interface(), "Failed to get MDL factory" );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context( mdlFactory->create_execution_context() );
        requireMdl( context.is_valid_interface(), "Failed to create MDL execution context" );

        mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(
            compileGeneratedDiffuseMaterial( session.neuray(), transaction.get(), context.get(), material.Kd ) );

        mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi(
            session.neuray()->get_api_component<mi::neuraylib::IMdl_backend_api>() );
        requireMdl( backendApi.is_valid_interface(), "Failed to get MDL backend API" );

        mi::base::Handle<mi::neuraylib::IMdl_backend> ptxBackend(
            backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
        requireMdl( ptxBackend.is_valid_interface(), "Failed to get MDL CUDA PTX backend" );
        requireMdl( ptxBackend->set_option( "sm_version", "50" ) == 0,
                    "Failed to set MDL CUDA PTX target architecture" );

        context->clear_messages();
        mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode( ptxBackend->translate_material_expression(
            transaction.get(), compiledMaterial.get(), "surface.scattering.tint", MDL_SMOKE_TINT_FUNCTION_NAME, context.get() ) );
        requireMdl( targetCode.is_valid_interface(), "Failed to translate MDL tint expression to PTX", context.get() );
        requireMdl( targetCode->get_code_size() > 0U, "MDL generated empty PTX target code" );
        requireMdl( targetCode->get_callable_function_count() == 1U,
                    "MDL generated unexpected callable function count" );
        requireMdl( std::string( targetCode->get_callable_function( 0 ) ) == MDL_SMOKE_TINT_FUNCTION_NAME,
                    "MDL generated unexpected callable function name" );

        ptx.assign( targetCode->get_code(), static_cast<size_t>( targetCode->get_code_size() ) );
    }
    requireMdl( transaction->commit() == 0, "Failed to commit MDL transaction" );
    transaction.reset();
    scope.reset();
    database.reset();
    requireMdl( session.shutdown() == 0, "Failed to shut down MDL SDK" );
    return ptx;
}

#endif  // OTK_USE_MDL

class PbrtProgramGroups : public ProgramGroups
{
  public:
    PbrtProgramGroups( const Options& options, GeometryLoaderPtr geometryLoader, MaterialLoaderPtr materialLoader, RendererPtr renderer )
        : m_options( options )
        , m_geometryLoader( std::move( geometryLoader ) )
        , m_materialLoader( std::move( materialLoader ) )
        , m_renderer( std::move( renderer ) )
    {
    }

    void initialize() override;
    void cleanup() override;

    uint_t getRealizedMaterialSbtOffset( const GeometryInstance& instance ) override;
#ifdef OTK_USE_MDL
    uint_t            getFallbackMaterialSbtOffset( const GeometryInstance& instance ) override;
    uint_t            getMdlMaterialSbtOffset( const GeometryInstance& instance ) override;
    MdlMaterialShader realizeMdlMaterialShader( const GeometryInstance& instance, uint_t shaderKeyId ) override;
#endif

  private:
    OptixModule createModule( const char* optixir, size_t optixirSize );
    void        createModules();
    void        createProgramGroups();
    void        ensurePhongModule();
    uint_t      getTriangleRealizedMaterialSbtOffset( const GeometryInstance& instance );
    uint_t      getTriangleFallbackRealizedMaterialSbtOffset( MaterialFlags flags );
#ifdef OTK_USE_MDL
    uint_t            getTriangleMdlSmokeMaterialSbtOffset();
    MdlMaterialShader realizeTriangleMdlSmokeMaterialShader( const PhongMaterial& material, uint_t shaderKeyId );
#endif
    uint_t getSphereRealizedMaterialSbtOffset();

    // Dependencies
    const Options&    m_options;
    GeometryLoaderPtr m_geometryLoader;
    MaterialLoaderPtr m_materialLoader;
    RendererPtr       m_renderer;

    OptixModule                    m_sceneModule{};
    OptixModule                    m_phongModule{};
#ifdef OTK_USE_MDL
    OptixModule                         m_mdlSmokeClosestHitModule{};
    std::vector<OptixModule>            m_mdlSmokeTintModules;
#endif
    OptixModule                    m_triangleModule{};
    OptixModule                    m_sphereModule{};
    std::vector<OptixProgramGroup> m_programGroups;
#ifdef OTK_USE_MDL
    std::vector<OptixProgramGroup>      m_callableProgramGroups;
    std::map<uint_t, MdlMaterialShader> m_mdlMaterialShaders;
#endif
    size_t                         m_triangleHitGroupIndex{};
#ifdef OTK_USE_MDL
    size_t                         m_triangleMdlSmokeHitGroupIndex{};
#endif
    size_t                         m_triangleAlphaMapHitGroupIndex{};
    size_t                         m_triangleDiffuseMapHitGroupIndex{};
    size_t                         m_triangleAlphaDiffuseMapHitGroupIndex{};
    size_t                         m_sphereHitGroupIndex{};
};

OptixModule PbrtProgramGroups::createModule( const char* optixir, size_t optixirSize )
{
    const OptixModuleCompileOptions    compileOptions{ getCompileOptions() };
    const OptixPipelineCompileOptions& pipelineCompileOptions{ m_renderer->getPipelineCompileOptions() };

    OptixModule        module;
    OptixDeviceContext context = m_renderer->getDeviceContext();
    OTK_ERROR_CHECK_LOG( optixModuleCreate( context, &compileOptions, &pipelineCompileOptions, optixir, optixirSize,
                                            LOG, &LOG_SIZE, &module ) );
    return module;
}

void PbrtProgramGroups::createModules()
{
    const OptixModuleCompileOptions    compileOptions{ getCompileOptions() };
    const OptixPipelineCompileOptions& pipelineCompileOptions{ m_renderer->getPipelineCompileOptions() };

    OptixDeviceContext context          = m_renderer->getDeviceContext();
    auto               getBuiltinModule = [&]( OptixPrimitiveType type ) {
        OptixModule           module;
        OptixBuiltinISOptions builtinOptions{};
        builtinOptions.builtinISModuleType = type;
        builtinOptions.buildFlags          = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        OTK_ERROR_CHECK_LOG( optixBuiltinISModuleGet( context, &compileOptions, &pipelineCompileOptions, &builtinOptions, &module ) );
        return module;
    };
    m_sceneModule    = createModule( DemandPbrtSceneCudaText(), DemandPbrtSceneCudaSize );
    m_triangleModule = getBuiltinModule( OPTIX_PRIMITIVE_TYPE_TRIANGLE );
    m_sphereModule   = getBuiltinModule( OPTIX_PRIMITIVE_TYPE_SPHERE );
}

void PbrtProgramGroups::createProgramGroups()
{
    OptixProgramGroupOptions options{};
    m_programGroups.resize( +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS );
    OptixProgramGroupDesc descs[+ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS]{};
    const char* const     proxyMaterialCHFunctionName = m_materialLoader->getCHFunctionName();
    otk::ProgramGroupDescBuilder( descs, m_sceneModule )
        .raygen( "__raygen__perspectiveCamera" )
        .miss( "__miss__backgroundColor" )
        .hitGroupISCH( m_sceneModule, m_geometryLoader->getISFunctionName(), m_sceneModule, m_geometryLoader->getCHFunctionName() )
        .hitGroupISCH( m_triangleModule, nullptr, m_sceneModule, proxyMaterialCHFunctionName )
        .hitGroupISAHCH( m_triangleModule, nullptr, m_sceneModule, "__anyhit__alphaCutOutPartialMesh", m_sceneModule, proxyMaterialCHFunctionName )
        .hitGroupISCH( m_sphereModule, nullptr, m_sceneModule, proxyMaterialCHFunctionName )
        .hitGroupISAHCH( m_sphereModule, nullptr, m_sceneModule, "__anyhit__sphere", m_sceneModule, proxyMaterialCHFunctionName );
    OptixDeviceContext context = m_renderer->getDeviceContext();
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, descs, m_programGroups.size(), &options, LOG, &LOG_SIZE,
                                                  m_programGroups.data() ) );
}

void PbrtProgramGroups::initialize()
{
    createModules();
    createProgramGroups();
    m_renderer->setProgramGroups( m_programGroups );
#ifdef OTK_USE_MDL
    if( !m_callableProgramGroups.empty() )
    {
        m_renderer->setCallableProgramGroups( m_callableProgramGroups );
    }
#endif
}

void PbrtProgramGroups::cleanup()
{
#ifdef OTK_USE_MDL
    for( OptixProgramGroup group : m_callableProgramGroups )
    {
        OTK_ERROR_CHECK( optixProgramGroupDestroy( group ) );
    }
#endif
    for( OptixProgramGroup group : m_programGroups )
    {
        OTK_ERROR_CHECK( optixProgramGroupDestroy( group ) );
    }
#ifdef OTK_USE_MDL
    for( OptixModule module : m_mdlSmokeTintModules )
    {
        OTK_ERROR_CHECK( optixModuleDestroy( module ) );
    }
    if( m_mdlSmokeClosestHitModule )
    {
        OTK_ERROR_CHECK( optixModuleDestroy( m_mdlSmokeClosestHitModule ) );
    }
#endif
    if( m_phongModule )
    {
        OTK_ERROR_CHECK( optixModuleDestroy( m_phongModule ) );
    }
    OTK_ERROR_CHECK( optixModuleDestroy( m_sceneModule ) );
}

uint_t PbrtProgramGroups::getTriangleFallbackRealizedMaterialSbtOffset( MaterialFlags flags )
{
    OptixDeviceContext       context = m_renderer->getDeviceContext();
    OptixProgramGroupOptions options{};
    OptixProgramGroup        group{};
    OptixProgramGroupDesc    groupDesc[1]{};

    // triangles with alpha map and diffuse map texture
    if( flagSet( flags, MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP ) )
    {
        if( m_triangleAlphaDiffuseMapHitGroupIndex == 0 )
        {
            otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )             //
                .hitGroupISAHCH( m_triangleModule, nullptr,                      //
                                 m_sceneModule, "__anyhit__alphaCutOutMesh",     //
                                 m_sceneModule, "__closesthit__texturedMesh" );  //
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
            m_triangleAlphaDiffuseMapHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
            m_programGroups.push_back( group );
            m_renderer->setProgramGroups( m_programGroups );
        }
        return m_triangleAlphaDiffuseMapHitGroupIndex;
    }

    // triangles with alpha map texture
    if( flagSet( flags, MaterialFlags::ALPHA_MAP ) )
    {
        if( m_triangleAlphaMapHitGroupIndex == 0 )
        {
            otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )          //
                .hitGroupISAHCH( m_triangleModule, nullptr,                   //
                                 m_sceneModule, "__anyhit__alphaCutOutMesh",  //
                                 m_phongModule, "__closesthit__mesh" );       //
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
            m_triangleAlphaMapHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
            m_programGroups.push_back( group );
            m_renderer->setProgramGroups( m_programGroups );
        }
        return m_triangleAlphaMapHitGroupIndex;
    }

    // triangles with diffuse map texture
    if( flagSet( flags, MaterialFlags::DIFFUSE_MAP ) )
    {
        if( m_triangleDiffuseMapHitGroupIndex == 0 )
        {
            otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )           //
                .hitGroupISCH( m_triangleModule, nullptr,                      //
                               m_sceneModule, "__closesthit__texturedMesh" );  //
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
            m_triangleDiffuseMapHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
            m_programGroups.push_back( group );
            m_renderer->setProgramGroups( m_programGroups );
        }
        return m_triangleDiffuseMapHitGroupIndex;
    }

    // untextured triangles
    if( m_triangleHitGroupIndex == 0 )
    {
        otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )   //
            .hitGroupISCH( m_triangleModule, nullptr,              //
                           m_phongModule, "__closesthit__mesh" );  //
        OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
        m_triangleHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
        m_programGroups.push_back( group );
        m_renderer->setProgramGroups( m_programGroups );
    }
    return m_triangleHitGroupIndex;
}

#ifdef OTK_USE_MDL
uint_t PbrtProgramGroups::getTriangleMdlSmokeMaterialSbtOffset()
{
    if( m_triangleMdlSmokeHitGroupIndex == 0 )
    {
        const Stopwatch optixTimer;
        m_mdlSmokeClosestHitModule = createModule( MdlSmokeMaterialCudaText(), MdlSmokeMaterialCudaSize );

        OptixProgramGroupOptions options{};
        OptixDeviceContext       context = m_renderer->getDeviceContext();

        OptixProgramGroupDesc groupDesc[1]{};
        OptixProgramGroup     group{};
        otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )  //
            .hitGroupISCH( m_triangleModule, nullptr,             //
                           m_mdlSmokeClosestHitModule, "__closesthit__mdlMesh" );
        OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
        m_triangleMdlSmokeHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
        m_programGroups.push_back( group );
        m_renderer->setProgramGroups( m_programGroups );

        const double optixTime{ optixTimer.elapsed() };
        std::cout << "MDL smoke material hit group setup: " << optixTime << " s\n";
    }
    return m_triangleMdlSmokeHitGroupIndex;
}

MdlMaterialShader PbrtProgramGroups::realizeTriangleMdlSmokeMaterialShader( const PhongMaterial& material, uint_t shaderKeyId )
{
    std::map<uint_t, MdlMaterialShader>::const_iterator it = m_mdlMaterialShaders.find( shaderKeyId );
    if( it != m_mdlMaterialShaders.end() )
    {
        return it->second;
    }

    const Stopwatch   compileTimer;
    const std::string ptx{ compileMdlSmokeTintPtx( material ) };
    const double      compileTime{ compileTimer.elapsed() };

    const Stopwatch optixTimer;
    OptixModule     tintModule{ createModule( ptx.data(), ptx.size() ) };

    OptixProgramGroupOptions options{};
    OptixDeviceContext       context = m_renderer->getDeviceContext();

    OptixProgramGroupDesc callableDesc{};
    callableDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callableDesc.callables.moduleDC            = tintModule;
    callableDesc.callables.entryFunctionNameDC = MDL_SMOKE_TINT_FUNCTION_NAME;
    OptixProgramGroup callableGroup{};
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, &callableDesc, 1, &options, LOG, &LOG_SIZE, &callableGroup ) );
    const MdlMaterialShader shader{ static_cast<uint_t>( m_callableProgramGroups.size() ), 1U };
    m_mdlSmokeTintModules.push_back( tintModule );
    m_callableProgramGroups.push_back( callableGroup );
    m_mdlMaterialShaders[shaderKeyId] = shader;
    m_renderer->setCallableProgramGroups( m_callableProgramGroups );

    const double optixTime{ optixTimer.elapsed() };
    std::cout << "Synchronous MDL smoke material compile: " << compileTime << " s, OptiX link setup: " << optixTime << " s\n";
    return shader;
}
#endif

uint_t PbrtProgramGroups::getTriangleRealizedMaterialSbtOffset( const GeometryInstance& instance )
{
    const MaterialFlags flags{ instance.groups[0].material.flags };
#ifdef OTK_USE_MDL
    if( m_options.mdlSmokeMaterial && instance.groups.size() == 1 && flags == MaterialFlags::NONE )
    {
        return getTriangleMdlSmokeMaterialSbtOffset();
    }
#endif

    return getTriangleFallbackRealizedMaterialSbtOffset( flags );
}

uint_t PbrtProgramGroups::getSphereRealizedMaterialSbtOffset()
{
    // untextured sphere
    if( m_sphereHitGroupIndex == 0 )
    {
        const OptixDeviceContext context = m_renderer->getDeviceContext();
        OptixProgramGroupOptions options{};
        OptixProgramGroup        group{};
        OptixProgramGroupDesc    groupDesc[1]{};

        otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )
            .hitGroupISCH( m_sphereModule, nullptr, m_phongModule, "__closesthit__sphere" );
        OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
        m_sphereHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
        m_programGroups.push_back( group );
        m_renderer->setProgramGroups( m_programGroups );
    }
    return m_sphereHitGroupIndex;
}

void PbrtProgramGroups::ensurePhongModule()
{
    if( m_phongModule == nullptr )
    {
        m_phongModule = createModule( PhongMaterialCudaText(), PhongMaterialCudaSize );
    }
}

#ifdef OTK_USE_MDL
uint_t PbrtProgramGroups::getFallbackMaterialSbtOffset( const GeometryInstance& instance )
{
    ensurePhongModule();

    if( instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        return getTriangleFallbackRealizedMaterialSbtOffset( instance.groups[0].material.flags );
    }
    if( instance.primitive == GeometryPrimitive::SPHERE )
    {
        return getSphereRealizedMaterialSbtOffset();
    }
    throw std::runtime_error( "Unimplemented primitive type " + std::to_string( +instance.primitive ) );
}
#endif

#ifdef OTK_USE_MDL
uint_t PbrtProgramGroups::getMdlMaterialSbtOffset( const GeometryInstance& instance )
{
    ensurePhongModule();

    if( instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        return getTriangleMdlSmokeMaterialSbtOffset();
    }
    throw std::runtime_error( "MDL materials are only implemented for triangle primitives" );
}

MdlMaterialShader PbrtProgramGroups::realizeMdlMaterialShader( const GeometryInstance& instance, uint_t shaderKeyId )
{
    if( instance.primitive != GeometryPrimitive::TRIANGLE || instance.groups.empty() )
    {
        throw std::runtime_error( "MDL materials are only implemented for triangle primitives" );
    }

    getTriangleMdlSmokeMaterialSbtOffset();
    return realizeTriangleMdlSmokeMaterialShader( instance.groups[0].material, shaderKeyId );
}
#endif

uint_t PbrtProgramGroups::getRealizedMaterialSbtOffset( const GeometryInstance& instance )
{
    ensurePhongModule();

    if( instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        return getTriangleRealizedMaterialSbtOffset( instance );
    }
    if( instance.primitive == GeometryPrimitive::SPHERE )
    {
        return getSphereRealizedMaterialSbtOffset();
    }
    throw std::runtime_error( "Unimplemented primitive type " + std::to_string( +instance.primitive ) );
}

}  // namespace

ProgramGroupsPtr createProgramGroups( const Options& options, GeometryLoaderPtr geometryLoader, MaterialLoaderPtr materialLoader, RendererPtr renderer )
{
    return std::make_shared<PbrtProgramGroups>( options, std::move( geometryLoader ), std::move( materialLoader ),
                                                std::move( renderer ) );
}

}  // namespace demandPbrtScene
