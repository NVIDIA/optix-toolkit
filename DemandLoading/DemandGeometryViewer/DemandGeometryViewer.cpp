//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "DemandGeometryViewer.h"

// This is disgusting, but glad.h insists it must be included first.
#include <glad/glad.h>

#include "DemandGeometryViewerKernelPTX.h"
#include "SphereInstances.h"

#include <OptiXToolkit/DemandGeometry/ProxyInstances.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Gui/BufferMapper.h>
#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/Trackball.h>
#include <OptiXToolkit/Gui/TrackballCamera.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/OptiXMemory/Record.h>
#include <OptiXToolkit/OptiXMemory/SyncRecord.h>
#include <OptiXToolkit/Util/Exception.h>

#include <GLFW/glfw3.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

namespace demandGeometryViewer {

using RayGenSbtRecord   = otk::Record<CameraData>;
using MissSbtRecord     = otk::Record<MissData>;
using HitGroupSbtRecord = otk::Record<HitGroupData>;
using OutputBuffer      = otk::CUDAOutputBuffer<uchar4>;

const int    NUM_PAYLOAD_VALUES   = 4;
const int    NUM_ATTRIBUTE_VALUES = 3;
const uint_t PROXY_SBT_INDEX      = 0;

[[noreturn]] void printUsageAndExit( const char* argv0 )
{
    // clang-format off
    std::cerr <<
        "Usage  : " << argv0 << " [options]\n"
        "Options: --file | -f <filename>      Specify file for image output\n"
        "         --help | -h                 Print this usage message\n"
        "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n"
        "         --frames=<num>              Specify number of warmup frames before writing to file\n";
    // clang-format on
    exit( 1 );
}

void contextLog( uint_t level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << '\n';
}

struct Options
{
    std::string outFile;
    int         width{ 512 };
    int         height{ 512 };
    float3      background;
    int         warmup{};
};

bool hasOption( const std::string& arg, const std::string& flag, std::istringstream& value )
{
    if( arg.substr( 0, flag.length() ) == flag )
    {
        value.str( arg.substr( flag.length() ) );
        return true;
    }
    return false;
}

Options parseArguments( int argc, char* argv[] )
{
    Options            options{};
    std::istringstream value;
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
            printUsageAndExit( argv[0] );

        if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
                options.outFile = argv[++i];
            else
                printUsageAndExit( argv[0] );
        }
        else if( hasOption( arg, "--dim=", value ) )
        {
            otk::parseDimensions( value.str().c_str(), options.width, options.height );
        }
        else if( hasOption( arg, "--bg=", value ) )
        {
            char sep{};
            value >> options.background.x >> sep >> options.background.y >> sep >> options.background.z;
        }
        else if( hasOption( arg, "--frames=", value ) )
        {
            int warmup{};
            value >> warmup;
            if( !value || warmup < 0 )
                printUsageAndExit( argv[0] );
            options.warmup = warmup;
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }
    if( options.warmup > 0 && options.outFile.empty() )
        printUsageAndExit( argv[0] );

    return options;
}

class CudaContextPushPop
{
  public:
    CudaContextPushPop( CUcontext current ) { OTK_ERROR_CHECK( cuCtxPushCurrent( current ) ); }
    ~CudaContextPushPop()
    {
        CUcontext ignored{};
        cuCtxPopCurrent( &ignored );
    }
};

struct SceneProxy
{
    uint_t level;
    uint_t pageId;
    float3 center;
    float radius;
};

struct ScenePrimitive
{
    uint_t level;
    float3 center;
    float radius;
    int index;
};

class DemandLoaderDestroyer
{
  public:
    void operator()( demandLoading::DemandLoader* ptr ) { demandLoading::destroyDemandLoader( ptr ); }
};

using DemandLoaderPtr = std::unique_ptr<demandLoading::DemandLoader, DemandLoaderDestroyer>;

class Application
{
  public:
    Application( Options&& cliOptions );

    void initialize();
    void cleanup();
    void run();

  private:
    void createContext();
    void initPipelineOpts();

    OptixModule createModuleFromSource( OptixModuleCompileOptions compileOptions );

    void createModules();
    void createProgramGroups();
    void createPipeline();
    void addProxy( uint_t level, float3 center, float radius );
    void createPrimitives();
    void createAccels();
    void pushInstance( OptixTraversableHandle handle );
    void createTopLevelTraversable();
    void updateTopLevelTraversable();
    void writeRayGenRecords();
    void writeMissRecords();
    void writeHitGroupRecords();
    void writeSbt();
    void buildShaderBindingTable();
    void initLaunchParams();

    void createGeometry( uint_t pageId );
    void updateSbt();

    void increaseLevel();
    void decreaseLevel();

    void runInteractive();
    void updateState( OutputBuffer& output );
    void handleCameraUpdate();
    void handleResize( OutputBuffer& output );
    void displayOutput( OutputBuffer& output, const otk::GLDisplay& display, GLFWwindow* window );

    void runToFile();
    void launch( OutputBuffer& output, uint_t width, uint_t height );
    void updateScene();
    void saveOutput( OutputBuffer& output );

    static Application* self( void* context ) { return static_cast<Application*>( context ); }
    static void keyCallback( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods, void* context )
    {
        self( context )->key( window, key, scanCode, action, mods );
    }
    void key( GLFWwindow* window, int32_t key, int32_t, int32_t action, int32_t );

    void cleanupContext();
    void cleanupModule();
    void cleanupProgramGroups();
    void cleanupPipeline();
    void cleanupDemandLoader();

    Options                     m_options;
    uint_t                      m_deviceIndex{};
    CUcontext                   m_cudaContext{};
    OptixDeviceContext          m_context{};
    CUstream                    m_stream{};
    OptixPipelineCompileOptions m_pipelineOpts{};
    OptixModule                 m_sampleModule{};
    OptixModule                 m_sphereModule{};
    bool                        m_updateNeeded{};

    enum
    {
        GROUP_RAYGEN = 0,
        GROUP_MISS,
        GROUP_HIT_GROUP_PROXIES,
        GROUP_HIT_GROUP_PRIMITIVES,
        NUM_GROUPS
    };
    OptixProgramGroup m_groups[NUM_GROUPS]{};
    OptixPipeline     m_pipeline{};

    otk::SyncRecord<CameraData>   m_rayGenRecord{ 1 };
    otk::SyncRecord<MissData>     m_missRecord{ 1 };
    otk::SyncRecord<HitGroupData> m_hitGroupRecords{ 2 * RAYTYPE_COUNT };
    OptixShaderBindingTable       m_sbt{};

    DemandLoaderPtr                                 m_loader;
    std::unique_ptr<demandGeometry::ProxyInstances> m_proxies;

    OptixTraversableHandle m_proxyInstanceTraversable{};

    SphereInstances        m_spheres;
    OptixTraversableHandle m_sphereTraversable{};

    otk::DeviceBuffer              m_devTopLevelInstanceTempBuffer;
    otk::DeviceBuffer              m_devTopLevelInstanceAccelBuffer;
    otk::SyncVector<OptixInstance> m_topLevelInstances;
    OptixTraversableHandle         m_topLevelTraversable{};

    otk::SyncVector<Params> m_params{ 1 };

    demandLoading::Ticket m_ticket;

    GLFWwindow*          m_window{};
    otk::TrackballCamera m_trackballCamera;

    uint_t                      m_level{};
    std::vector<SceneProxy>     m_sceneProxies;
    std::vector<ScenePrimitive> m_scenePrimitives;
};

Application::Application( Options&& cliOptions )
    : m_options( std::move( cliOptions ) )
{
}

void Application::initialize()
{
    createContext();
    initPipelineOpts();
    createModules();
    createProgramGroups();
    createPipeline();
    createPrimitives();
    createAccels();
    buildShaderBindingTable();
    initLaunchParams();
}

void Application::createContext()
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    m_loader.reset( createDemandLoader( demandLoading::Options{} ) );

    std::vector<uint_t> devices = m_loader->getDevices();
    if( devices.empty() )
    {
        throw std::runtime_error( "No devices support demand loading." );
    }
    m_deviceIndex = devices[0];
    OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
    OTK_ERROR_CHECK( cuCtxGetCurrent( &m_cudaContext ) );

    OTK_ERROR_CHECK( optixInit() );
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = contextLog;
    options.logCallbackLevel    = 4;
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OTK_ERROR_CHECK( optixDeviceContextCreate( m_cudaContext, &options, &m_context ) );

    OTK_ERROR_CHECK( cudaStreamCreate( &m_stream ) );

    OTK_ERROR_CHECK( cuCtxSetCurrent( m_cudaContext ) );

    m_proxies.reset( new demandGeometry::ProxyInstances( m_loader.get() ) );
}

void Application::initPipelineOpts()
{
    m_pipelineOpts.usesMotionBlur         = 0;
    m_pipelineOpts.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    m_pipelineOpts.numPayloadValues       = NUM_PAYLOAD_VALUES;
    m_pipelineOpts.numAttributeValues     = std::max( NUM_ATTRIBUTE_VALUES, m_proxies->getNumAttributes() );
    m_pipelineOpts.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipelineOpts.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
    m_pipelineOpts.pipelineLaunchParamsVariableName = "params";
}

OptixModule Application::createModuleFromSource( OptixModuleCompileOptions compileOptions )
{
    OptixModule module;
    OPTIX_CHECK_LOG2( optixModuleCreate( m_context, &compileOptions, &m_pipelineOpts, DemandGeometryViewer_ptx_text(),
                                         DemandGeometryViewer_ptx_size, LOG, &LOG_SIZE, &module ) );
    return module;
}

void Application::createModules()
{
    OptixModuleCompileOptions compileOptions{};
    compileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    bool debugInfo{ false };
#else
    bool debugInfo{ true };
#endif
    compileOptions.optLevel   = debugInfo ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 : OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    compileOptions.debugLevel = debugInfo ? OPTIX_COMPILE_DEBUG_LEVEL_FULL : OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    m_sampleModule = createModuleFromSource( compileOptions );

    OptixBuiltinISOptions builtinOptions{};
    builtinOptions.usesMotionBlur      = false;
    builtinOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK_LOG2( optixBuiltinISModuleGet( m_context, &compileOptions, &m_pipelineOpts, &builtinOptions, &m_sphereModule ) );
}

void Application::createProgramGroups()
{
    OptixProgramGroupOptions options{};
    OptixProgramGroupDesc    descs[NUM_GROUPS]{};
    otk::ProgramGroupDescBuilder( descs, m_sampleModule )
        .raygen( "__raygen__pinHoleCamera" )
        .miss( "__miss__backgroundColor" )
        .hitGroupCHIS( m_sampleModule, m_proxies->getCHFunctionName(), m_sampleModule, m_proxies->getISFunctionName() )
        .hitGroupCHIS( m_sampleModule, "__closesthit__sphere", m_sphereModule, nullptr );
    OPTIX_CHECK_LOG2( optixProgramGroupCreate( m_context, descs, NUM_GROUPS, &options, LOG, &LOG_SIZE, m_groups ) );
}

void Application::createPipeline()
{
    const uint_t             maxTraceDepth = 1;
    OptixPipelineLinkOptions options;
    options.maxTraceDepth = maxTraceDepth;
    OPTIX_CHECK_LOG2( optixPipelineCreate( m_context, &m_pipelineOpts, &options, m_groups, NUM_GROUPS, LOG, &LOG_SIZE, &m_pipeline ) );

    OptixStackSizes stackSizes{};
    for( OptixProgramGroup group : m_groups )
    {
#if OPTIX_VERSION < 70700
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes ) );
#else
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes, m_pipeline ) );
#endif
    }
    uint_t directCallableTraversalStackSize{};
    uint_t directCallableStateStackSize{};
    uint_t continuationStackSize{};
    OTK_ERROR_CHECK( optixUtilComputeStackSizes( &stackSizes, maxTraceDepth, 0, 0, &directCallableTraversalStackSize,
                                             &directCallableStateStackSize, &continuationStackSize ) );
    const uint_t maxTraversableDepth = 3;
    OTK_ERROR_CHECK( optixPipelineSetStackSize( m_pipeline, directCallableTraversalStackSize, directCallableStateStackSize,
                                            continuationStackSize, maxTraversableDepth ) );
}

void Application::addProxy( uint_t level, float3 center, float radius )
{
    const float3    min = center - radius * make_float3( 1.0f, 1.0f, 1.0f );
    const float3    max = center + radius * make_float3( 1.0f, 1.0f, 1.0f );
    const OptixAabb bounds{ min.x, min.y, min.z, max.x, max.y, max.z };
    const uint_t    pageId = m_proxies->add( bounds );
    m_sceneProxies.push_back( { level, pageId, center, radius } );
}

void Application::createPrimitives()
{
    addProxy( 0, { 0.0f, 0.0f, 0.0f }, 1.0f );
    m_proxies->copyToDeviceAsync( m_stream );
    m_params[0].demandGeomContext.proxies = m_proxies->getTypedProxyDataDevicePtr();

    const float3        Ka       = { 0.0f, 0.0f, 0.0f };
    const float3        Kd       = { 0.5f * 1.0f, 0.5f * 0.9f, 0.5f * 0.7f };
    const float3        Ks       = { 0.5f * 1.0f, 0.5f * 0.9f, 0.5f * 0.7f };
    const float3        Kr       = { 0.5f, 0.5f, 0.5f };
    const float         phongExp = 128.0f;
    const PhongMaterial mat      = {
        Ka, Kd, Ks, Kr, phongExp,
    };
    m_params[0].proxyMaterial = mat;
}

void Application::createAccels()
{
    m_proxies->setSbtIndex( PROXY_SBT_INDEX );
    m_proxyInstanceTraversable = m_proxies->createTraversable( m_context, m_stream );
    CUDA_SYNC_CHECK();
    m_spheres.setSbtIndex( RAYTYPE_COUNT );
    m_sphereTraversable = m_spheres.createTraversable( m_context, m_stream );
    CUDA_SYNC_CHECK();
    createTopLevelTraversable();
    CUDA_SYNC_CHECK();
}

static void identity( float ( &result )[12] )
{
    // clang-format off
    const float matrix[12]{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    // clang-format on
    std::copy( std::begin( matrix ), std::end( matrix ), std::begin( result ) );
}

void Application::pushInstance( OptixTraversableHandle handle )
{
    OptixInstance instance;
    identity( instance.transform );
    instance.instanceId        = 0;
    instance.sbtOffset         = 0U;
    instance.visibilityMask    = 255U;
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = handle;
    m_topLevelInstances.push_back( instance );
}

void Application::createTopLevelTraversable()
{
    m_topLevelInstances.clear();
    pushInstance( m_proxyInstanceTraversable );
    pushInstance( m_sphereTraversable );
    m_topLevelInstances.copyToDeviceAsync( m_stream );
    CUDA_SYNC_CHECK();

    const uint_t    NUM_BUILD_INPUTS = 1;
    OptixBuildInput inputs[NUM_BUILD_INPUTS]{};
    otk::BuildInputBuilder( inputs ).instanceArray( m_topLevelInstances, 2U );

    const OptixAccelBuildOptions options = {
        OPTIX_BUILD_FLAG_ALLOW_UPDATE,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD     // operation
    };
    OptixAccelBufferSizes sizes{};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( m_context, &options, inputs, NUM_BUILD_INPUTS, &sizes ) );

    m_devTopLevelInstanceTempBuffer.resize( sizes.tempSizeInBytes );
    m_devTopLevelInstanceAccelBuffer.resize( sizes.outputSizeInBytes );
    CUDA_SYNC_CHECK();
    OTK_ERROR_CHECK( optixAccelBuild( m_context, m_stream, &options, inputs, NUM_BUILD_INPUTS, m_devTopLevelInstanceTempBuffer,
                                  sizes.tempSizeInBytes, m_devTopLevelInstanceAccelBuffer, sizes.outputSizeInBytes,
                                  &m_topLevelTraversable, nullptr, 0 ) );
}

void Application::updateTopLevelTraversable()
{
    m_topLevelInstances[0].traversableHandle = m_proxyInstanceTraversable;
    m_topLevelInstances[1].traversableHandle = m_sphereTraversable;
    m_topLevelInstances.copyToDeviceAsync( m_stream );

    const uint_t    NUM_BUILD_INPUTS = 1;
    OptixBuildInput inputs[NUM_BUILD_INPUTS]{};
    otk::BuildInputBuilder( inputs ).instanceArray( m_topLevelInstances, static_cast<uint_t>( m_topLevelInstances.size() ) );

    const OptixAccelBuildOptions options = {
        OPTIX_BUILD_FLAG_ALLOW_UPDATE,  // buildFlags
        OPTIX_BUILD_OPERATION_UPDATE    // operation
    };
    OptixAccelBufferSizes sizes{};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( m_context, &options, inputs, NUM_BUILD_INPUTS, &sizes ) );

    m_devTopLevelInstanceTempBuffer.resize( sizes.tempUpdateSizeInBytes );
    m_devTopLevelInstanceAccelBuffer.resize( sizes.outputSizeInBytes );
    OTK_ERROR_CHECK( optixAccelBuild( m_context, m_stream, &options, inputs, NUM_BUILD_INPUTS, m_devTopLevelInstanceTempBuffer,
                                  sizes.tempSizeInBytes, m_devTopLevelInstanceAccelBuffer, sizes.outputSizeInBytes,
                                  &m_topLevelTraversable, nullptr, 0 ) );
    CUDA_SYNC_CHECK();
}

void Application::writeRayGenRecords()
{
    m_rayGenRecord[0].eye = m_trackballCamera.getCameraEye();
    m_trackballCamera.cameraUVWFrame( m_rayGenRecord[0].U, m_rayGenRecord[0].V, m_rayGenRecord[0].W );
    m_rayGenRecord.packHeader( 0, m_groups[GROUP_RAYGEN] );
    m_rayGenRecord.copyToDeviceAsync( m_stream );
}

void Application::writeMissRecords()
{
    m_missRecord[0].background = m_options.background;
    m_missRecord.packHeader( 0, m_groups[GROUP_MISS] );
    m_missRecord.copyToDeviceAsync( m_stream );
}

void Application::writeHitGroupRecords()
{
    // HitGroup records for proxies.
    uint_t record = 0;
    for( size_t i = 0; i < RAYTYPE_COUNT; ++i )
    {
        m_hitGroupRecords.packHeader( record, m_groups[GROUP_HIT_GROUP_PROXIES] );
        ++record;
    }

    // HitGroup records for loaded geometries.
    // Use a blue material initially so we know that the proper SBT
    // hit group record was rewritten when the geometry is loaded.
    for( size_t i = 0; i < RAYTYPE_COUNT; ++i )
    {
        m_hitGroupRecords.packHeader( record, m_groups[GROUP_HIT_GROUP_PRIMITIVES] );
        HitGroupData& primData   = m_hitGroupRecords[record];
        primData.spheres.indices = m_spheres.getIndicesDevicePtr();
        const float3 Ka          = { 0.0f, 0.0f, 0.0f };
        const float3 Kd          = { 0.0f, 0.0f, 1.0f };  // blue
        const float3 Ks          = { 0.5f * 1.0f, 0.5f * 0.0f, 0.5f * 0.0f };
        const float3 Kr          = { 0.5f, 0.5f, 0.5f };
        const float  phongExp    = 128.0f;
        primData.material        = {
            Ka, Kd, Ks, Kr, phongExp,
        };
        ++record;
    }
    m_hitGroupRecords.copyToDeviceAsync( m_stream );
}

void Application::writeSbt()
{
    m_sbt.raygenRecord                = m_rayGenRecord;
    m_sbt.missRecordBase              = m_missRecord;
    m_sbt.missRecordStrideInBytes     = static_cast<uint_t>( sizeof( MissSbtRecord ) );
    m_sbt.missRecordCount             = 1;
    m_sbt.hitgroupRecordBase          = m_hitGroupRecords;
    m_sbt.hitgroupRecordCount         = static_cast<uint_t>( m_hitGroupRecords.size() );
    m_sbt.hitgroupRecordStrideInBytes = static_cast<uint_t>( sizeof( HitGroupSbtRecord ) );
}

void Application::buildShaderBindingTable()
{
    writeRayGenRecords();
    writeMissRecords();
    writeHitGroupRecords();
    writeSbt();
}

void Application::initLaunchParams()
{
    BasicLight lights[]{ {
                             make_float3( 4.0f, 2.0f, -3.0f ),  // pos
                             make_float3( 0.5f, 0.5f, 0.5f )    // color
                         },
                         {
                             make_float3( 1.0f, 4.0f, 4.0f ),  // pos
                             make_float3( 0.5f, 0.5f, 0.5f )   // color
                         },
                         {
                             make_float3( -3.0f, 5.0f, -1.0f ),  // pos
                             make_float3( 0.5f, 0.5f, 0.5f )     // color
                         } };
    // normalize the light directions to start
    for( BasicLight& light : lights )
        light.pos = normalize( light.pos );
    Params& params = m_params[0];
    std::copy( std::begin( lights ), std::end( lights ), std::begin( params.lights ) );
    params.ambientColor = make_float3( 0.4f, 0.4f, 0.4f );
    params.sceneEpsilon = 1.e-4f;
    params.traversable  = m_topLevelTraversable;
}

void Application::createGeometry( uint_t pageId )
{
    auto pos = std::find_if( m_sceneProxies.begin(), m_sceneProxies.end(),
                             [pageId]( const SceneProxy& prim ) { return prim.pageId == pageId; } );
    if( pos == m_sceneProxies.end() )
    {
        throw std::runtime_error("Unknown page id " + std::to_string(pageId));
    }

    m_proxies->remove( pageId );
    m_scenePrimitives.push_back( { pos->level, pos->center, pos->radius, static_cast<int>( m_scenePrimitives.size() ) } );
    m_spheres.add( pos->center, pos->radius, m_scenePrimitives.back().index );
    m_updateNeeded = true;
}

void Application::updateSbt()
{
    uint_t record = RAYTYPE_COUNT;  // First RAYTYPE_COUNT records are for proxies.
    for( size_t i = 0; i < RAYTYPE_COUNT; ++i )
    {
        m_hitGroupRecords.packHeader( record, m_groups[GROUP_HIT_GROUP_PRIMITIVES] );
        HitGroupData& primData   = m_hitGroupRecords[record];
        primData.spheres.indices = m_spheres.getIndicesDevicePtr();
        const float3 Ka          = { 0.0f, 0.0f, 0.0f };
        const float3 Kd          = { 1.0f, 0.0f, 0.0f };  // red
        const float3 Ks          = { 0.5f * 1.0f, 0.5f * 0.0f, 0.5f * 0.0f };
        const float3 Kr          = { 0.5f, 0.5f, 0.5f };
        const float  phongExp    = 128.0f;
        primData.material        = {
            Ka, Kd, Ks, Kr, phongExp,
        };
        ++record;
    }
    m_hitGroupRecords.copyToDeviceAsync( m_stream );
}

void Application::increaseLevel()
{
    m_level++;
    std::cout << "Level " << m_level << '\n';
}

void Application::decreaseLevel()
{
    if( m_level > 0 )
    {
        m_level--;
        std::cout << "Level " << m_level << '\n';
    }
}

void Application::run()
{
    if( m_options.outFile.empty() )
        runInteractive();
    else
        runToFile();
}

void Application::runInteractive()
{
    m_window = otk::initGLFW( "DemandGeometry", m_options.width, m_options.height );
    otk::initGL();
    m_trackballCamera.trackWindow( m_window );
    m_trackballCamera.setKeyHandler( &Application::keyCallback, this );

    {
        OutputBuffer output{ otk::CUDAOutputBufferType::CUDA_DEVICE, m_options.width, m_options.height };
        output.setStream( m_stream );
        otk::GLDisplay display;
        do
        {
            glfwPollEvents();
            updateState( output );
            launch( output, m_trackballCamera.getWidth(), m_trackballCamera.getHeight() );
            displayOutput( output, display, m_window );
            glfwSwapBuffers( m_window );
        } while( glfwWindowShouldClose( m_window ) == 0 );
    }
}

void Application::updateState( OutputBuffer& output )
{
    handleCameraUpdate();
    handleResize( output );
}

void Application::handleCameraUpdate()
{
    if( !m_trackballCamera.handleCameraUpdate() )
        return;

    m_rayGenRecord[0].eye = m_trackballCamera.getCameraEye();
    m_trackballCamera.cameraUVWFrame( m_rayGenRecord[0].U, m_rayGenRecord[0].V, m_rayGenRecord[0].W );
    m_rayGenRecord.copyToDeviceAsync( m_stream );
}

void Application::handleResize( OutputBuffer& output )
{
    if( !m_trackballCamera.handleResize() )
        return;

    int width  = m_trackballCamera.getWidth();
    int height = m_trackballCamera.getHeight();
    output.resize( width, height );
    m_params[0].width  = width;
    m_params[0].height = height;
}

void Application::displayOutput( OutputBuffer& output, const otk::GLDisplay& display, GLFWwindow* window )
{
    int frameBuffResX;
    int frameBuffResY;
    glfwGetFramebufferSize( window, &frameBuffResX, &frameBuffResY );
    display.display( output.width(), output.height(), frameBuffResX, frameBuffResY, output.getPBO() );
}

void Application::runToFile()
{
    OutputBuffer output{ otk::CUDAOutputBufferType::CUDA_DEVICE, m_options.width, m_options.height };
    output.setStream( m_stream );

    for( int i = 0; i < m_options.warmup; ++i )
        launch( output, m_options.width, m_options.height );

    launch( output, m_options.width, m_options.height );
    saveOutput( output );
}

void Application::launch( OutputBuffer& output, uint_t width, uint_t height )
{
    updateScene();

    otk::BufferMapper<uchar4> map( output );

    Params& params = m_params[0];

    OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
    m_loader->launchPrepare( m_stream, params.demandContext );
    CUDA_SYNC_CHECK();

    OTK_ERROR_CHECK( cuCtxSetCurrent( m_cudaContext ) );
    params.image  = map;
    params.width  = width;
    params.height = height;
    m_params.copyToDeviceAsync( m_stream );
    CUDA_SYNC_CHECK();

    const int launchDepth = 1;
    OTK_ERROR_CHECK( optixLaunch( m_pipeline, m_stream, reinterpret_cast<CUdeviceptr>( m_params.devicePtr() ),
                              sizeof( Params ), &m_sbt, params.width, params.height, launchDepth ) );
    CUDA_SYNC_CHECK();

    OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
    m_ticket = m_loader->processRequests( m_stream, params.demandContext );
    CUDA_SYNC_CHECK();

    OTK_ERROR_CHECK( cuCtxSetCurrent( m_cudaContext ) );
}

void Application::updateScene()
{
    m_ticket.wait();

    OTK_ERROR_CHECK( cuCtxSetCurrent( m_cudaContext ) );
    for( const uint_t pageId : m_proxies->requestedResources() )
    {
        createGeometry( pageId );
    }

    if( m_updateNeeded )
    {
        m_proxyInstanceTraversable = m_proxies->createTraversable( m_context, m_stream );
        m_sphereTraversable        = m_spheres.createTraversable( m_context, m_stream );
        updateSbt();
        updateTopLevelTraversable();
        m_updateNeeded = false;
    }
}

void Application::saveOutput( OutputBuffer& output )
{
    otk::ImageBuffer buffer;
    buffer.data         = output.getHostPointer();
    buffer.width        = m_options.width;
    buffer.height       = m_options.height;
    buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;
    saveImage( m_options.outFile.c_str(), buffer, false );
}

void Application::cleanup()
{
    cleanupDemandLoader();
    cleanupPipeline();
    cleanupProgramGroups();
    cleanupModule();
    cleanupContext();
}

void Application::cleanupContext()
{
    OTK_ERROR_CHECK( cudaStreamDestroy( m_stream ) );

    OTK_ERROR_CHECK( optixDeviceContextDestroy( m_context ) );
}

void Application::cleanupModule()
{
    OTK_ERROR_CHECK( optixModuleDestroy( m_sampleModule ) );
}

void Application::cleanupProgramGroups()
{
    for( OptixProgramGroup group : m_groups )
        OTK_ERROR_CHECK( optixProgramGroupDestroy( group ) );
}

void Application::cleanupPipeline()
{
    OTK_ERROR_CHECK( optixPipelineDestroy( m_pipeline ) );
}

void Application::cleanupDemandLoader()
{
    m_loader.reset();
}

void Application::key( GLFWwindow* window, int32_t key, int32_t /*scanCode*/, int32_t action, int32_t /*mods*/ )
{
    if( window != m_window )
        return;

    if( action == GLFW_PRESS )
        switch( key )
        {
            case GLFW_KEY_UP:
                increaseLevel();
                break;

            case GLFW_KEY_DOWN:
                decreaseLevel();
                break;

            case GLFW_KEY_Q:
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose( window, 1 );
                break;

            default:
                break;
        }
}

}  // namespace demandGeometryViewer

int main( int argc, char* argv[] )
{
    try
    {
        demandGeometryViewer::Application app( demandGeometryViewer::parseArguments( argc, argv ) );
        app.initialize();
        app.run();
        app.cleanup();
    }
    catch( const std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << '\n';
        return 1;
    }
    catch( ... )
    {
        std::cerr << "Unknown exception\n";
        return 2;
    }
    return 0;
}
