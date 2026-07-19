// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/ProgramGroups.h"

#include "DemandPbrtScene/Config.h"
#include "DemandPbrtScene/DemandTextureCache.h"
#include "DemandPbrtScene/IdRangePrinter.h"
#include "DemandPbrtScene/ImageSourceFactory.h"
#ifdef OTK_USE_MDL
#include "DemandPbrtScene/MdlBsdfCompiler.h"
#include "DemandPbrtScene/MdlShaderCache.h"
#endif
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
#ifdef OTK_USE_MDL
#include <OptiXToolkit/Error/cuErrorCheck.h>
#endif
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>

#ifdef OTK_USE_MDL
#include <optix_stack_size.h>
#endif
#include <optix_stubs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#ifdef OTK_USE_MDL
#include <condition_variable>
#include <deque>
#endif
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iterator>
#ifdef OTK_USE_MDL
#include <map>
#endif
#include <memory>
#ifdef OTK_USE_MDL
#include <mutex>
#include <set>
#endif
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef OTK_USE_MDL
#include <thread>
#endif
#include <utility>
#include <vector>

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

OptixModule createOptixModule( OptixDeviceContext                 context,
                               const OptixPipelineCompileOptions& pipelineCompileOptions,
                               const char*                        optixir,
                               size_t                             optixirSize )
{
    const OptixModuleCompileOptions compileOptions{ getCompileOptions() };
    OptixModule                     module{};
    OTK_ERROR_CHECK_LOG( optixModuleCreate( context, &compileOptions, &pipelineCompileOptions, optixir, optixirSize,
                                            LOG, &LOG_SIZE, &module ) );
    return module;
}

#ifdef OTK_USE_MDL
OptixPipeline createOptixPipeline( OptixDeviceContext                    context,
                                   const OptixPipelineCompileOptions&    pipelineCompileOptions,
                                   const std::vector<OptixProgramGroup>& programGroups,
                                   const std::vector<OptixProgramGroup>& callableProgramGroups )
{
    const uint_t             maxTraceDepth{ 1 };
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = maxTraceDepth;
    std::vector<OptixProgramGroup> pipelineProgramGroups{ programGroups };
    std::copy( callableProgramGroups.cbegin(), callableProgramGroups.cend(), std::back_inserter( pipelineProgramGroups ) );

    OptixPipeline pipeline{};
    OTK_ERROR_CHECK_LOG( optixPipelineCreate( context, &pipelineCompileOptions, &linkOptions, pipelineProgramGroups.data(),
                                              pipelineProgramGroups.size(), LOG, &LOG_SIZE, &pipeline ) );

    OptixStackSizes stackSizes{};
    for( OptixProgramGroup group : pipelineProgramGroups )
    {
#if OPTIX_VERSION < 70700
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes ) );
#else
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes, pipeline ) );
#endif
    }
    uint_t       directCallableTraversalStackSize{};
    uint_t       directCallableStateStackSize{};
    uint_t       continuationStackSize{};
    const uint_t maxDirectCallableDepth{ callableProgramGroups.empty() ? 0U : 1U };
    OTK_ERROR_CHECK( optixUtilComputeStackSizes( &stackSizes, maxTraceDepth, 0, maxDirectCallableDepth, &directCallableTraversalStackSize,
                                                 &directCallableStateStackSize, &continuationStackSize ) );
    const uint_t maxTraversableDepth{ 3 };
    OTK_ERROR_CHECK( optixPipelineSetStackSize( pipeline, directCallableTraversalStackSize, directCallableStateStackSize,
                                                continuationStackSize, maxTraversableDepth ) );
    return pipeline;
}
#endif

namespace {

#ifdef OTK_USE_MDL

constexpr const char* MDL_MATERIAL_TINT_FUNCTION_NAME = "__direct_callable__mdlMaterialTint";
constexpr const char* MDL_MATERIAL_BSDF_FUNCTION_NAME = "__direct_callable__mdlMaterialBsdf";

struct MdlMaterialTargetCode
{
    std::string        tintPtx;
    MdlBsdfCallablePtx bsdfPtx;
    bool               hasBsdfCallables{};
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

std::string generatedMdlContext( const GeneratedMdlSource& source, const MdlShaderKey& key )
{
    return "module=" + source.moduleName + ", material=" + source.materialName + ", key=" + toString( key );
}

std::string generatedMdlMessage( const std::string& message, const GeneratedMdlSource& source, const MdlShaderKey& key )
{
    return message + " (" + generatedMdlContext( source, key ) + ")";
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

otk::pbrt::PbrtMaterial makeSyntheticMatteMaterial( const float3& kd )
{
    otk::pbrt::PbrtMaterial material;
    material.type = "matte";

    std::unique_ptr<::pbrt::Float[]> values{ new ::pbrt::Float[3] };
    values[0] = kd.x;
    values[1] = kd.y;
    values[2] = kd.z;
    material.params.AddRGBSpectrum( "Kd", std::move( values ), 3 );
    return material;
}

void bindGeneratedColorParameter( mi::neuraylib::IFunction_call*      materialCall,
                                  mi::neuraylib::IValue_factory*      valueFactory,
                                  mi::neuraylib::IExpression_factory* expressionFactory,
                                  const GeneratedMdlSource&           source,
                                  const MdlShaderKey&                 key,
                                  const MdlBoundMaterialParameter&    parameter )
{
    mi::base::Handle<mi::neuraylib::IValue_color> value(
        valueFactory->create_color( parameter.red, parameter.green, parameter.blue ) );
    requireMdl( value.is_valid_interface(),
                generatedMdlMessage( "Failed to create generated MDL " + parameter.name + " value", source, key ) );

    mi::base::Handle<mi::neuraylib::IExpression_constant> expression( expressionFactory->create_constant( value.get() ) );
    requireMdl( expression.is_valid_interface(),
                generatedMdlMessage( "Failed to create generated MDL " + parameter.name + " expression", source, key ) );

    const mi::Sint32 bindResult = materialCall->set_argument( parameter.name.c_str(), expression.get() );
    requireMdl( bindResult == 0,
                generatedMdlMessage( "Failed to bind generated MDL " + parameter.name + " parameter", source, key ) );
}

void bindGeneratedFloatParameter( mi::neuraylib::IFunction_call*      materialCall,
                                  mi::neuraylib::IValue_factory*      valueFactory,
                                  mi::neuraylib::IExpression_factory* expressionFactory,
                                  const GeneratedMdlSource&           source,
                                  const MdlShaderKey&                 key,
                                  const MdlBoundMaterialParameter&    parameter )
{
    mi::base::Handle<mi::neuraylib::IValue_float> value( valueFactory->create_float( parameter.value ) );
    requireMdl( value.is_valid_interface(),
                generatedMdlMessage( "Failed to create generated MDL " + parameter.name + " value", source, key ) );

    mi::base::Handle<mi::neuraylib::IExpression_constant> expression( expressionFactory->create_constant( value.get() ) );
    requireMdl( expression.is_valid_interface(),
                generatedMdlMessage( "Failed to create generated MDL " + parameter.name + " expression", source, key ) );

    const mi::Sint32 bindResult = materialCall->set_argument( parameter.name.c_str(), expression.get() );
    requireMdl( bindResult == 0,
                generatedMdlMessage( "Failed to bind generated MDL " + parameter.name + " parameter", source, key ) );
}

void bindGeneratedMaterialParameters( mi::neuraylib::IFunction_call*                materialCall,
                                      mi::neuraylib::IValue_factory*                valueFactory,
                                      mi::neuraylib::IExpression_factory*           expressionFactory,
                                      const GeneratedMdlSource&                     source,
                                      const MdlShaderKey&                           key,
                                      const std::vector<MdlBoundMaterialParameter>& parameters )
{
    for( std::vector<MdlBoundMaterialParameter>::const_iterator it = parameters.begin(); it != parameters.end(); ++it )
    {
        if( it->type == MdlBoundParameterType::COLOR )
        {
            bindGeneratedColorParameter( materialCall, valueFactory, expressionFactory, source, key, *it );
        }
        else
        {
            bindGeneratedFloatParameter( materialCall, valueFactory, expressionFactory, source, key, *it );
        }
    }
}

mi::base::Handle<mi::neuraylib::ICompiled_material> compileGeneratedMaterial( mi::neuraylib::INeuray*      neuray,
                                                                              mi::neuraylib::ITransaction* transaction,
                                                                              mi::neuraylib::IMdl_execution_context* context,
                                                                              const GeneratedMdlSource& source,
                                                                              const MdlShaderKey&       key,
                                                                              const std::vector<MdlBoundMaterialParameter>& parameters )
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( neuray->get_api_component<mi::neuraylib::IMdl_factory>() );
    requireMdl( mdlFactory.is_valid_interface(), "Failed to get MDL factory" );

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdlImpexpApi( neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>() );
    requireMdl( mdlImpexpApi.is_valid_interface(), "Failed to get MDL import/export API" );

    mi::base::Handle<const mi::IString> moduleDbName( mdlFactory->get_db_module_name( source.moduleName.c_str() ) );
    requireMdl( moduleDbName.is_valid_interface(), generatedMdlMessage( "Failed to get generated MDL module DB name", source, key ) );

    mi::base::Handle<const mi::neuraylib::IModule> module( transaction->access<mi::neuraylib::IModule>( moduleDbName->get_c_str() ) );
    if( !module.is_valid_interface() )
    {
        context->clear_messages();
        const mi::Sint32 loadResult =
            mdlImpexpApi->load_module_from_string( transaction, source.moduleName.c_str(), source.source.c_str(), context );
        requireMdl( loadResult == 0, generatedMdlMessage( "Failed to load generated MDL module", source, key ), context );
        module = transaction->access<mi::neuraylib::IModule>( moduleDbName->get_c_str() );
    }
    requireMdl( module.is_valid_interface(), generatedMdlMessage( "Failed to access generated MDL module", source, key ) );
    requireMdl( module->get_material_count() == 1U,
                generatedMdlMessage( "Generated MDL module did not contain exactly one material", source, key ) );

    const char* const materialDbName = module->get_material( 0 );
    requireMdl( materialDbName != nullptr, generatedMdlMessage( "Generated MDL module had no material definition", source, key ) );

    mi::base::Handle<const mi::neuraylib::IFunction_definition> materialDefinition(
        transaction->access<mi::neuraylib::IFunction_definition>( materialDbName ) );
    requireMdl( materialDefinition.is_valid_interface(),
                generatedMdlMessage( "Failed to access generated MDL material definition", source, key ) );

    mi::Sint32 callResult = 0;
    mi::base::Handle<mi::neuraylib::IFunction_call> materialCall( materialDefinition->create_function_call( nullptr, &callResult ) );
    requireMdl( callResult == 0 && materialCall.is_valid_interface(),
                generatedMdlMessage( "Failed to create generated MDL material call", source, key ) );

    mi::base::Handle<mi::neuraylib::IValue_factory> valueFactory( mdlFactory->create_value_factory( transaction ) );
    requireMdl( valueFactory.is_valid_interface(), generatedMdlMessage( "Failed to create MDL value factory", source, key ) );

    mi::base::Handle<mi::neuraylib::IExpression_factory> expressionFactory( mdlFactory->create_expression_factory( transaction ) );
    requireMdl( expressionFactory.is_valid_interface(),
                generatedMdlMessage( "Failed to create MDL expression factory", source, key ) );

    bindGeneratedMaterialParameters( materialCall.get(), valueFactory.get(), expressionFactory.get(), source, key, parameters );

    mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance(
        materialCall->get_interface<mi::neuraylib::IMaterial_instance>() );
    requireMdl( materialInstance.is_valid_interface(),
                generatedMdlMessage( "Failed to create generated MDL material instance", source, key ) );

    mi::base::Handle<mi::neuraylib::IType_factory> typeFactory( mdlFactory->create_type_factory( transaction ) );
    requireMdl( typeFactory.is_valid_interface(), generatedMdlMessage( "Failed to create MDL type factory", source, key ) );

    mi::base::Handle<const mi::neuraylib::IType> standardMaterialType(
        typeFactory->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL ) );
    requireMdl( standardMaterialType.is_valid_interface(), generatedMdlMessage( "Failed to get MDL material type", source, key ) );

    context->clear_messages();
    const mi::Sint32 targetTypeResult = context->set_option( "target_type", standardMaterialType.get() );
    requireMdl( targetTypeResult == 0, generatedMdlMessage( "Failed to set MDL target material type", source, key ), context );

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(
        materialInstance->create_compiled_material( mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context ) );
    requireMdl( compiledMaterial.is_valid_interface(),
                generatedMdlMessage( "Failed to compile generated MDL material", source, key ), context );
    return compiledMaterial;
}

const char* findMdlPreviewColorExpressionPath( const mi::neuraylib::ICompiled_material* compiledMaterial )
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
    failMdl( "Generated MDL material has no preview color expression" );
}

MdlMaterialTargetCode compileMdlMaterialTargetCode( const MaterialGroup& group, bool includeBsdfCallables )
{
    MdlSdkSession session;
    requireMdl( session.isStarted(), session.error() );

    mi::base::Handle<mi::neuraylib::IDatabase> database( session.neuray()->get_api_component<mi::neuraylib::IDatabase>() );
    requireMdl( database.is_valid_interface(), "Failed to get MDL database" );

    mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope() );
    requireMdl( scope.is_valid_interface(), "Failed to get MDL global scope" );

    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction() );
    requireMdl( transaction.is_valid_interface(), "Failed to create MDL transaction" );

    MdlMaterialTargetCode targetCode;
    {
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory( session.neuray()->get_api_component<mi::neuraylib::IMdl_factory>() );
        requireMdl( mdlFactory.is_valid_interface(), "Failed to get MDL factory" );

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context( mdlFactory->create_execution_context() );
        requireMdl( context.is_valid_interface(), "Failed to create MDL execution context" );

        const otk::pbrt::PbrtMaterial  syntheticMaterial{ makeSyntheticMatteMaterial( group.material.Kd ) };
        const otk::pbrt::PbrtMaterial& pbrtMaterial{
            group.pbrtMaterial && !group.pbrtMaterial->type.empty() ? *group.pbrtMaterial : syntheticMaterial };
        MdlGeneratedSourceCache                      sourceCache;
        const MdlShaderKey                           key{ makeMdlShaderKey( pbrtMaterial ) };
        const GeneratedMdlSource&                    source{ sourceCache.getSource( pbrtMaterial ) };
        const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( pbrtMaterial ) };
        mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial(
            compileGeneratedMaterial( session.neuray(), transaction.get(), context.get(), source, key, parameters ) );
        const char* const previewColorExpressionPath{ findMdlPreviewColorExpressionPath( compiledMaterial.get() ) };

        mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi(
            session.neuray()->get_api_component<mi::neuraylib::IMdl_backend_api>() );
        requireMdl( backendApi.is_valid_interface(), "Failed to get MDL backend API" );

        mi::base::Handle<mi::neuraylib::IMdl_backend> ptxBackend(
            backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
        requireMdl( ptxBackend.is_valid_interface(), "Failed to get MDL CUDA PTX backend" );
        requireMdl( ptxBackend->set_option( "sm_version", "50" ) == 0,
                    "Failed to set MDL CUDA PTX target architecture" );

        context->clear_messages();
        mi::base::Handle<const mi::neuraylib::ITarget_code> tintTargetCode(
            ptxBackend->translate_material_expression( transaction.get(), compiledMaterial.get(), previewColorExpressionPath,
                                                       MDL_MATERIAL_TINT_FUNCTION_NAME, context.get() ) );
        requireMdl( tintTargetCode.is_valid_interface(), "Failed to translate MDL tint expression to PTX", context.get() );
        requireMdl( tintTargetCode->get_code_size() > 0U, "MDL generated empty PTX target code" );
        requireMdl( tintTargetCode->get_callable_function_count() == 1U,
                    "MDL generated unexpected callable function count" );
        requireMdl( std::string( tintTargetCode->get_callable_function( 0 ) ) == MDL_MATERIAL_TINT_FUNCTION_NAME,
                    "MDL generated unexpected callable function name" );

        targetCode.tintPtx.assign( tintTargetCode->get_code(), static_cast<size_t>( tintTargetCode->get_code_size() ) );
        if( includeBsdfCallables )
        {
            targetCode.bsdfPtx =
                compileMdlBsdfCallablesToPtx( session.neuray(), transaction.get(), compiledMaterial.get(),
                                              context.get(), "surface.scattering", MDL_MATERIAL_BSDF_FUNCTION_NAME );
            targetCode.hasBsdfCallables = true;
        }
    }
    requireMdl( transaction->commit() == 0, "Failed to commit MDL transaction" );
    transaction.reset();
    scope.reset();
    database.reset();
    requireMdl( session.shutdown() == 0, "Failed to shut down MDL SDK" );
    return targetCode;
}

struct MdlMaterialBuildJob
{
    uint_t                         shaderKeyId{};
    MaterialGroup                  group{};
    CUcontext                      cudaContext{};
    OptixDeviceContext             optixContext{};
    OptixPipelineCompileOptions    pipelineCompileOptions{};
    std::vector<OptixProgramGroup> programGroups;
    std::vector<OptixProgramGroup> callableProgramGroups;
    bool                           includeBsdfCallables{};
};

struct MdlMaterialBuildResult
{
    uint_t                         shaderKeyId{};
    MdlMaterialShader              shader{};
    OptixModule                    tintModule{};
    OptixModule                    bsdfModule{};
    std::vector<OptixProgramGroup> createdCallableProgramGroups;
    OptixPipeline                  pipeline{};
    std::vector<OptixProgramGroup> programGroups;
    std::vector<OptixProgramGroup> callableProgramGroups;
    std::string                    diagnostics;
};

void buildMdlMaterialHitGroupDesc( OptixProgramGroupDesc ( &groupDesc )[1],
                                   OptixModule   sceneModule,
                                   OptixModule   triangleModule,
                                   OptixModule   closestHitModule,
                                   MaterialFlags flags )
{
    if( flagSet( flags, MaterialFlags::ALPHA_MAP ) )
    {
        otk::ProgramGroupDescBuilder( groupDesc, sceneModule )          //
            .hitGroupISAHCH( triangleModule, nullptr,                   //
                             sceneModule, "__anyhit__alphaCutOutMesh",  //
                             closestHitModule, "__closesthit__mdlMesh" );
        return;
    }

    otk::ProgramGroupDescBuilder( groupDesc, sceneModule )  //
        .hitGroupISCH( triangleModule, nullptr,             //
                       closestHitModule, "__closesthit__mdlMesh" );
}

OptixProgramGroup createDirectCallableProgramGroup( OptixDeviceContext context, OptixModule module, const std::string& functionName )
{
    OptixProgramGroupOptions options{};
    OptixProgramGroupDesc    callableDesc{};
    callableDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callableDesc.callables.moduleDC            = module;
    callableDesc.callables.entryFunctionNameDC = functionName.c_str();

    OptixProgramGroup group{};
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, &callableDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
    return group;
}

MdlMaterialShader appendMdlMaterialCallableProgramGroups( OptixDeviceContext              context,
                                                          OptixModule                     tintModule,
                                                          OptixModule                     bsdfModule,
                                                          const MdlMaterialTargetCode&    targetCode,
                                                          std::vector<OptixProgramGroup>& createdCallableProgramGroups,
                                                          std::vector<OptixProgramGroup>& callableProgramGroups )
{
    const MdlMaterialShader shader{ static_cast<uint_t>( callableProgramGroups.size() ), targetCode.hasBsdfCallables ? 5U : 1U };

    OptixProgramGroup tintGroup{ createDirectCallableProgramGroup( context, tintModule, MDL_MATERIAL_TINT_FUNCTION_NAME ) };
    createdCallableProgramGroups.push_back( tintGroup );
    callableProgramGroups.push_back( tintGroup );

    if( !targetCode.hasBsdfCallables )
    {
        return shader;
    }

    const std::string bsdfFunctions[] = {
        targetCode.bsdfPtx.initFunctionName,
        targetCode.bsdfPtx.sampleFunctionName,
        targetCode.bsdfPtx.evaluateFunctionName,
        targetCode.bsdfPtx.pdfFunctionName,
    };
    for( const std::string& functionName : bsdfFunctions )
    {
        OptixProgramGroup group{ createDirectCallableProgramGroup( context, bsdfModule, functionName ) };
        createdCallableProgramGroups.push_back( group );
        callableProgramGroups.push_back( group );
    }

    return shader;
}

MdlMaterialShader bindMdlMaterialResources( const MaterialGroup& group, MdlMaterialShader shader )
{
    shader.usesDiffuseTexture = group.pbrtMaterial && group.pbrtMaterial->type == "matte";
    return shader;
}

void destroyMdlMaterialBuildResultNoThrow( MdlMaterialBuildResult& result )
{
    if( result.pipeline )
    {
        OTK_ERROR_CHECK_NOTHROW( optixPipelineDestroy( result.pipeline ) );
        result.pipeline = nullptr;
    }
    for( OptixProgramGroup group : result.createdCallableProgramGroups )
    {
        if( group )
        {
            OTK_ERROR_CHECK_NOTHROW( optixProgramGroupDestroy( group ) );
        }
    }
    result.createdCallableProgramGroups.clear();
    if( result.tintModule )
    {
        OTK_ERROR_CHECK_NOTHROW( optixModuleDestroy( result.tintModule ) );
        result.tintModule = nullptr;
    }
    if( result.bsdfModule )
    {
        OTK_ERROR_CHECK_NOTHROW( optixModuleDestroy( result.bsdfModule ) );
        result.bsdfModule = nullptr;
    }
}

MdlMaterialBuildResult buildMdlMaterialPipelineState( const MdlMaterialBuildJob& job )
{
    MdlMaterialBuildResult result{};
    result.shaderKeyId = job.shaderKeyId;
    try
    {
        if( job.cudaContext )
        {
            OTK_ERROR_CHECK( cuCtxSetCurrent( job.cudaContext ) );
        }

        const Stopwatch             compileTimer;
        const MdlMaterialTargetCode targetCode{ compileMdlMaterialTargetCode( job.group, job.includeBsdfCallables ) };
        const double                compileTime{ compileTimer.elapsed() };

        const Stopwatch optixTimer;
        result.tintModule = createOptixModule( job.optixContext, job.pipelineCompileOptions, targetCode.tintPtx.data(),
                                               targetCode.tintPtx.size() );
        if( targetCode.hasBsdfCallables )
        {
            result.bsdfModule = createOptixModule( job.optixContext, job.pipelineCompileOptions,
                                                   targetCode.bsdfPtx.ptx.data(), targetCode.bsdfPtx.ptx.size() );
        }

        result.programGroups         = job.programGroups;
        result.callableProgramGroups = job.callableProgramGroups;
        result.shader = bindMdlMaterialResources(
            job.group,
            appendMdlMaterialCallableProgramGroups( job.optixContext, result.tintModule, result.bsdfModule, targetCode,
                                                    result.createdCallableProgramGroups, result.callableProgramGroups ) );
        result.pipeline = createOptixPipeline( job.optixContext, job.pipelineCompileOptions, result.programGroups,
                                               result.callableProgramGroups );

        const double optixTime{ optixTimer.elapsed() };
        std::cout << "Asynchronous MDL material compile: " << compileTime << " s, OptiX link setup: " << optixTime << " s\n";
    }
    catch( const std::exception& e )
    {
        result.diagnostics = e.what();
        destroyMdlMaterialBuildResultNoThrow( result );
    }
    catch( ... )
    {
        result.diagnostics = "Unknown asynchronous MDL material build failure";
        destroyMdlMaterialBuildResultNoThrow( result );
    }
    return result;
}

class MdlMaterialBuildWorker
{
  public:
    MdlMaterialBuildWorker()
        : m_thread( &MdlMaterialBuildWorker::run, this )
    {
    }

    ~MdlMaterialBuildWorker() { shutdown(); }

    bool enqueue( const MdlMaterialBuildJob& job )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        if( m_stop || m_queuedShaderKeys.count( job.shaderKeyId ) || m_inFlightShaderKeys.count( job.shaderKeyId )
            || m_results.count( job.shaderKeyId ) )
        {
            return false;
        }

        m_jobs.push_back( job );
        m_queuedShaderKeys.insert( job.shaderKeyId );
        m_condition.notify_one();
        return true;
    }

    bool takeResult( uint_t shaderKeyId, MdlMaterialBuildResult& result )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        auto                        it = m_results.find( shaderKeyId );
        if( it == m_results.end() )
        {
            return false;
        }

        result = std::move( it->second );
        m_results.erase( it );
        return true;
    }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock( m_mutex );
            m_stop = true;
            m_condition.notify_one();
        }
        if( m_thread.joinable() )
        {
            m_thread.join();
        }

        std::lock_guard<std::mutex> lock( m_mutex );
        for( auto& entry : m_results )
        {
            destroyMdlMaterialBuildResultNoThrow( entry.second );
        }
        m_results.clear();
    }

  private:
    void run()
    {
        for( ;; )
        {
            MdlMaterialBuildJob job;
            {
                std::unique_lock<std::mutex> lock( m_mutex );
                m_condition.wait( lock, [&]() { return m_stop || !m_jobs.empty(); } );
                if( m_stop && m_jobs.empty() )
                {
                    return;
                }

                job = m_jobs.front();
                m_jobs.pop_front();
                m_queuedShaderKeys.erase( job.shaderKeyId );
                m_inFlightShaderKeys.insert( job.shaderKeyId );
            }

            MdlMaterialBuildResult result{ buildMdlMaterialPipelineState( job ) };

            std::lock_guard<std::mutex> lock( m_mutex );
            m_inFlightShaderKeys.erase( job.shaderKeyId );
            m_results[job.shaderKeyId] = std::move( result );
        }
    }

    std::mutex                               m_mutex;
    std::condition_variable                  m_condition;
    std::deque<MdlMaterialBuildJob>          m_jobs;
    std::set<uint_t>                         m_queuedShaderKeys;
    std::set<uint_t>                         m_inFlightShaderKeys;
    std::map<uint_t, MdlMaterialBuildResult> m_results;
    bool                                     m_stop{};
    std::thread                              m_thread;
};

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
    OptixModule       createModule( const char* optixir, size_t optixirSize );
    void              createModules();
    void              createProgramGroups();
    void              ensurePhongModule();
    uint_t            getTriangleRealizedMaterialSbtOffset( const GeometryInstance& instance );
    uint_t            getTriangleFallbackRealizedMaterialSbtOffset( MaterialFlags flags );
#ifdef OTK_USE_MDL
    void              requestMdlMaterialBuild( const MaterialGroup& group, uint_t shaderKeyId );
    bool              installPendingMdlMaterialBuild( uint_t shaderKeyId, MdlMaterialShader& shader );
    uint_t            getTriangleMdlMaterialSbtOffset( MaterialFlags flags );
    MdlMaterialShader realizeTriangleMdlMaterialShader( const MaterialGroup& group, uint_t shaderKeyId );
#endif
    uint_t            getSphereRealizedMaterialSbtOffset();

    // Dependencies
    const Options&    m_options;
    GeometryLoaderPtr m_geometryLoader;
    MaterialLoaderPtr m_materialLoader;
    RendererPtr       m_renderer;

    OptixModule                    m_sceneModule{};
    OptixModule                    m_phongModule{};
#ifdef OTK_USE_MDL
    OptixModule                            m_mdlMaterialClosestHitModule{};
    std::vector<OptixModule>               m_mdlMaterialTintModules;
    std::vector<OptixModule>               m_mdlMaterialBsdfModules;
    std::unique_ptr<MdlMaterialBuildWorker> m_mdlMaterialBuildWorker;
#endif
    OptixModule                    m_triangleModule{};
    OptixModule                    m_sphereModule{};
    std::vector<OptixProgramGroup> m_programGroups;
#ifdef OTK_USE_MDL
    std::vector<OptixProgramGroup>      m_callableProgramGroups;
    std::map<uint_t, MdlMaterialShader> m_mdlMaterialShaders;
    size_t m_triangleMdlMaterialHitGroupIndex{};
    size_t m_triangleMdlAlphaMapHitGroupIndex{};
#endif
    size_t m_triangleHitGroupIndex{};
    size_t m_triangleAlphaMapHitGroupIndex{};
    size_t m_triangleDiffuseMapHitGroupIndex{};
    size_t m_triangleAlphaDiffuseMapHitGroupIndex{};
    size_t m_sphereHitGroupIndex{};
};

OptixModule PbrtProgramGroups::createModule( const char* optixir, size_t optixirSize )
{
    const OptixPipelineCompileOptions& pipelineCompileOptions{ m_renderer->getPipelineCompileOptions() };
    OptixDeviceContext                 context = m_renderer->getDeviceContext();
    return createOptixModule( context, pipelineCompileOptions, optixir, optixirSize );
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
    m_mdlMaterialBuildWorker.reset();
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
    for( OptixModule module : m_mdlMaterialTintModules )
    {
        OTK_ERROR_CHECK( optixModuleDestroy( module ) );
    }
    for( OptixModule module : m_mdlMaterialBsdfModules )
    {
        OTK_ERROR_CHECK( optixModuleDestroy( module ) );
    }
    if( m_mdlMaterialClosestHitModule )
    {
        OTK_ERROR_CHECK( optixModuleDestroy( m_mdlMaterialClosestHitModule ) );
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
uint_t PbrtProgramGroups::getTriangleMdlMaterialSbtOffset( MaterialFlags flags )
{
    size_t& hitGroupIndex{ flagSet( flags, MaterialFlags::ALPHA_MAP ) ? m_triangleMdlAlphaMapHitGroupIndex :
                                                                        m_triangleMdlMaterialHitGroupIndex };
    if( hitGroupIndex == 0 )
    {
        const Stopwatch optixTimer;
        if( m_mdlMaterialClosestHitModule == nullptr )
        {
            m_mdlMaterialClosestHitModule = createModule( MdlSmokeMaterialCudaText(), MdlSmokeMaterialCudaSize );
        }

        OptixProgramGroupOptions options{};
        OptixDeviceContext       context = m_renderer->getDeviceContext();

        OptixProgramGroupDesc groupDesc[1]{};
        OptixProgramGroup     group{};
        buildMdlMaterialHitGroupDesc( groupDesc, m_sceneModule, m_triangleModule, m_mdlMaterialClosestHitModule, flags );
        OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
        hitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
        m_programGroups.push_back( group );
        m_renderer->setProgramGroups( m_programGroups );

        const double optixTime{ optixTimer.elapsed() };
        std::cout << "MDL material hit group setup: " << optixTime << " s\n";
    }

    return hitGroupIndex;
}

MdlMaterialShader PbrtProgramGroups::realizeTriangleMdlMaterialShader( const MaterialGroup& group, uint_t shaderKeyId )
{
    std::map<uint_t, MdlMaterialShader>::const_iterator it = m_mdlMaterialShaders.find( shaderKeyId );
    if( it != m_mdlMaterialShaders.end() )
    {
        return it->second;
    }

    if( !m_options.mdlSynchronousCompilation )
    {
        requestMdlMaterialBuild( group, shaderKeyId );
        MdlMaterialShader shader{};
        if( installPendingMdlMaterialBuild( shaderKeyId, shader ) )
        {
            return shader;
        }
        throw MdlMaterialBuildPending( "MDL material build is still pending" );
    }

    const Stopwatch             compileTimer;
    const MdlMaterialTargetCode targetCode{
        compileMdlMaterialTargetCode( group, m_options.useMdlMaterials && shaderKeyId != 0U ) };
    const double                compileTime{ compileTimer.elapsed() };

    const Stopwatch optixTimer;
    OptixModule     tintModule{ createModule( targetCode.tintPtx.data(), targetCode.tintPtx.size() ) };
    OptixModule     bsdfModule{};
    if( targetCode.hasBsdfCallables )
    {
        bsdfModule = createModule( targetCode.bsdfPtx.ptx.data(), targetCode.bsdfPtx.ptx.size() );
    }

    OptixDeviceContext             context = m_renderer->getDeviceContext();
    std::vector<OptixProgramGroup> createdCallableProgramGroups;
    const MdlMaterialShader        shader{ bindMdlMaterialResources(
        group, appendMdlMaterialCallableProgramGroups( context, tintModule, bsdfModule, targetCode,
                                                       createdCallableProgramGroups, m_callableProgramGroups ) ) };
    m_mdlMaterialTintModules.push_back( tintModule );
    if( bsdfModule )
    {
        m_mdlMaterialBsdfModules.push_back( bsdfModule );
    }
    m_mdlMaterialShaders[shaderKeyId] = shader;
    m_renderer->setCallableProgramGroups( m_callableProgramGroups );

    const double optixTime{ optixTimer.elapsed() };
    std::cout << "Synchronous MDL material compile: " << compileTime << " s, OptiX link setup: " << optixTime << " s\n";
    return shader;
}
#endif

uint_t PbrtProgramGroups::getTriangleRealizedMaterialSbtOffset( const GeometryInstance& instance )
{
    const MaterialFlags flags{ instance.groups[0].material.flags };
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
void PbrtProgramGroups::requestMdlMaterialBuild( const MaterialGroup& group, uint_t shaderKeyId )
{
    if( m_options.mdlSynchronousCompilation )
    {
        return;
    }
    if( m_mdlMaterialShaders.count( shaderKeyId ) )
    {
        return;
    }

    CUcontext cudaContext{};
    OTK_ERROR_CHECK( cuCtxGetCurrent( &cudaContext ) );
    if( !m_mdlMaterialBuildWorker )
    {
        m_mdlMaterialBuildWorker.reset( new MdlMaterialBuildWorker{} );
    }

    MdlMaterialBuildJob job{};
    job.shaderKeyId            = shaderKeyId;
    job.group                  = group;
    job.cudaContext            = cudaContext;
    job.optixContext           = m_renderer->getDeviceContext();
    job.pipelineCompileOptions = m_renderer->getPipelineCompileOptions();
    job.programGroups          = m_programGroups;
    job.callableProgramGroups  = m_callableProgramGroups;
    job.includeBsdfCallables   = m_options.useMdlMaterials && shaderKeyId != 0U;
    m_mdlMaterialBuildWorker->enqueue( job );
}

bool PbrtProgramGroups::installPendingMdlMaterialBuild( uint_t shaderKeyId, MdlMaterialShader& shader )
{
    if( m_options.mdlSynchronousCompilation )
    {
        return false;
    }
    const std::map<uint_t, MdlMaterialShader>::const_iterator existing = m_mdlMaterialShaders.find( shaderKeyId );
    if( existing != m_mdlMaterialShaders.end() )
    {
        shader = existing->second;
        return true;
    }
    if( !m_mdlMaterialBuildWorker )
    {
        return false;
    }

    MdlMaterialBuildResult result{};
    if( !m_mdlMaterialBuildWorker->takeResult( shaderKeyId, result ) )
    {
        return false;
    }

    if( !result.diagnostics.empty() )
    {
        throw std::runtime_error( result.diagnostics );
    }

    shader                            = result.shader;
    m_mdlMaterialShaders[shaderKeyId] = shader;
    m_mdlMaterialTintModules.push_back( result.tintModule );
    if( result.bsdfModule )
    {
        m_mdlMaterialBsdfModules.push_back( result.bsdfModule );
    }
    m_programGroups         = std::move( result.programGroups );
    m_callableProgramGroups = std::move( result.callableProgramGroups );
    m_renderer->setPipelineState( result.pipeline, m_programGroups, m_callableProgramGroups );
    result.tintModule = nullptr;
    result.bsdfModule = nullptr;
    result.createdCallableProgramGroups.clear();
    result.pipeline = nullptr;
    return true;
}

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
        return getTriangleMdlMaterialSbtOffset( instance.groups[0].material.flags );
    }
    throw std::runtime_error( "MDL materials are only implemented for triangle primitives" );
}

MdlMaterialShader PbrtProgramGroups::realizeMdlMaterialShader( const GeometryInstance& instance, uint_t shaderKeyId )
{
    if( instance.primitive != GeometryPrimitive::TRIANGLE || instance.groups.empty() )
    {
        throw std::runtime_error( "MDL materials are only implemented for triangle primitives" );
    }

    getTriangleMdlMaterialSbtOffset( instance.groups[0].material.flags );
    return realizeTriangleMdlMaterialShader( instance.groups[0], shaderKeyId );
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
