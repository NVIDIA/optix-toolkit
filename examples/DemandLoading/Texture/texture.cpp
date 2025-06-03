// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "SourceDir.h"  // generated from SourceDir.h.in
#include "TextureKernelCuda.h"
#include "textureKernel.h"

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/Window.h>
#include <OptiXToolkit/ImageSources/ImageSources.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/Util/Logger.h>

#ifdef OPTIX_SAMPLE_USE_CORE_EXR
#include <OptiXToolkit/ImageSource/CoreEXRReader.h>
#endif

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

using namespace demandLoading;
using namespace imageSource;
using namespace otk;  // for vec_math operators

int          g_numThreads       = 0;
int          g_totalLaunches    = 0;
double       g_totalLaunchTime  = 0.0;
unsigned int g_totalRequests    = 0;
float        g_mipLevelBias     = 0.0f;
bool         g_useCoreExr       = false;

int32_t g_width      = 768;
int32_t g_height     = 768;
int32_t g_bucketSize = 256;

int g_textureWidth  = 2048;
int g_textureHeight = 2048;

otk::Camera g_camera;

struct PerDeviceSampleState
{
    int32_t                     device_idx               = -1;
    OptixDeviceContext          context                  = 0;
    OptixTraversableHandle      gas_handle               = 0;  // Traversable handle for triangle AS
    CUdeviceptr                 d_gas_output_buffer      = 0;  // Triangle AS memory
    OptixModule                 optixir_module               = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline               pipeline                 = 0;
    OptixProgramGroup           raygen_prog_group        = 0;
    OptixProgramGroup           miss_prog_group          = 0;
    OptixProgramGroup           hitgroup_prog_group      = 0;
    OptixShaderBindingTable     sbt                      = {};
    Params                      params                   = {};
    Params*                     d_params                 = nullptr;
    CUstream                    stream                   = 0;

    // Only valid on the host
    std::shared_ptr<DemandLoader> demandLoader;
    demandLoading::Ticket ticket;
};


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


void printUsageAndExit( const char* argv0 )
{
    // clang-format off
    std::cerr
        << "\nUsage  : " << argv0 << " [options]\n"
        << "Options: --help | -h                         Print this usage message\n"
        << "         --file | -f <filename>              Specify file for image output\n"
        << "         --dim=<width>x<height>              Set image dimensions\n"
        << "         --texture | -t <filename>           Texture to render (path relative to data folder). Use checkerboard for procedural texture.\n"
        << "         --textureDim=<width>x<height>       Set dimensions of procedural texture (default 2048x2048).\n"
        << "         --bias | -b <bias>                  Mip level bias (default 0.0)\n"
        << "         --textureScale <s>                  Texture scale (how many times to wrap the texture around the sphere) (default 1.0f)\n"
        << "         --bucketSize <dim>                  The size of the screen-space tiles used for rendering (default 256).\n"
        << "         --numThreads <n>                    The number of threads to use for processing requests; 0 is automatic (default 0).\n"
#ifdef OPTIX_SAMPLE_USE_CORE_EXR        
        << "         --useCoreEXR <true|false>           Use the CoreEXR reader (default false).\n"
#endif
        << "\n";
    // clang-format on
    exit( 1 );
}


void initCameraState()
{
    float3 camEye = {-6.0f, 0.0f, 0.0f};
    g_camera.setEye( camEye );
    g_camera.setLookAt( make_float3( 0.0f, 0.0f, 0.0f ) );
    g_camera.setUp( make_float3( 0.0f, 0.0f, 1.0f ) );
    g_camera.setFovY( 30.0f );
    g_camera.setAspectRatio( static_cast<float>( g_width ) / static_cast<float>( g_height ) );
}


void createContext( PerDeviceSampleState& state )
{
    // Initialize CUDA on this device
    OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( cudaStreamCreate( &state.stream ) );

    OptixDeviceContext        context;
    CUcontext                 cuCtx   = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    otk::util::setLogger( options );
    OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}


void createContexts( std::vector<PerDeviceSampleState>& states )
{
    for( unsigned int deviceIndex : getDemandLoadDevices( true ) )
    {
        states.emplace_back();
        states.back().device_idx = static_cast<int32_t>( deviceIndex );
        createContext( states.back() );
    }
}


void buildAccel( PerDeviceSampleState& state )
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    OptixAabb   aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
    CUdeviceptr d_aabb_buffer;
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
    OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};

    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = 1;

    uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_gas;
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OTK_ERROR_CHECK( optixAccelBuild( state.context,
                                  0,  // CUDA stream
                                  &accel_options, &aabb_input,
                                  1,  // num build inputs
                                  d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_gas_and_compacted_size,
                                  gas_buffer_sizes.outputSizeInBytes, &state.gas_handle,
                                  &emitProperty,  // emitted property list
                                  1               // num emitted properties
                                  ) );

    OTK_ERROR_CHECK( cuMemFree( d_temp_buffer_gas ) );
    OTK_ERROR_CHECK( cuMemFree( d_aabb_buffer ) );

    size_t compacted_gas_size;
    OTK_ERROR_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OTK_ERROR_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer,
                                        compacted_gas_size, &state.gas_handle ) );

        OTK_ERROR_CHECK( cuMemFree( d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


void createModule( PerDeviceSampleState& state )
{
    OptixModuleCompileOptions module_compile_options{};
    otk::configModuleCompileOptions( module_compile_options );

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 3;
    state.pipeline_compile_options.numAttributeValues    = 6;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    OTK_ERROR_CHECK_LOG( optixModuleCreate( state.context, &module_compile_options, &state.pipeline_compile_options,
                                        textureKernelCudaText(), textureKernelCudaSize, log, &sizeof_log, &state.optixir_module ) );
}


void createProgramGroups( PerDeviceSampleState& state )
{
    OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.optixir_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.optixir_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log                                  = sizeof( log );
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.miss_prog_group ) );

    OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = state.optixir_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroup_prog_group_desc.hitgroup.moduleIS            = state.optixir_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
    sizeof_log                                            = sizeof( log );
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.hitgroup_prog_group ) );
}


void createPipeline( PerDeviceSampleState& state )
{
    const uint32_t    max_trace_depth  = 1;
    OptixProgramGroup program_groups[] = {state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace_depth;
    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OTK_ERROR_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ), log,
                                          &sizeof_log, &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
#if OPTIX_VERSION < 70700
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
#else
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, state.pipeline ) );
#endif
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OTK_ERROR_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                             0,  // maxCCDepth
                                             0,  // maxDCDEpth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OTK_ERROR_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}


void createSBT( PerDeviceSampleState& state, const DemandTexture& texture, float texture_scale, float texture_lod )
{
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &raygen_record ), raygen_record_size ) );
    RayGenSbtRecord rg_sbt = {};
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &miss_record ), miss_record_size ) );
    MissSbtRecord ms_sbt;
    ms_sbt.data = {0.05f, 0.05f, 0.3f};
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &ms_sbt ) );
    OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );

    // The demand-loaded texture id is passed to the closest hit program via the hitgroup record.
    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &hitgroup_record ), hitgroup_record_size ) );
    HitGroupSbtRecord hg_sbt;
    hg_sbt.data = {1.5f /*radius*/, texture.getId(), texture_scale, texture_lod};
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hg_sbt ) );
    OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( hitgroup_record ), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice ) );

    state.sbt.raygenRecord                = raygen_record;
    state.sbt.missRecordBase              = miss_record;
    state.sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    state.sbt.missRecordCount             = 1;
    state.sbt.hitgroupRecordBase          = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    state.sbt.hitgroupRecordCount         = 1;
}


void cleanupState( PerDeviceSampleState& state )
{
    OTK_ERROR_CHECK( optixPipelineDestroy( state.pipeline ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.hitgroup_prog_group ) );
    OTK_ERROR_CHECK( optixModuleDestroy( state.optixir_module ) );
    OTK_ERROR_CHECK( optixDeviceContextDestroy( state.context ) );

    OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
    OTK_ERROR_CHECK( cudaStreamDestroy( state.stream ) );

    OTK_ERROR_CHECK( cuMemFree( state.sbt.raygenRecord ) );
    OTK_ERROR_CHECK( cuMemFree( state.sbt.missRecordBase ) );
    OTK_ERROR_CHECK( cuMemFree( state.sbt.hitgroupRecordBase ) );
    OTK_ERROR_CHECK( cuMemFree( state.d_gas_output_buffer ) );
    OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( state.d_params ) ) );
}


TextureDescriptor makeTextureDescription()
{
    TextureDescriptor texDesc{};
    texDesc.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
    texDesc.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
    texDesc.filterMode       = CU_TR_FILTER_MODE_LINEAR;
    texDesc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
    texDesc.maxAnisotropy    = 16;

    return texDesc;
}


void initLaunchParams( PerDeviceSampleState& state, unsigned int numDevices )
{
    state.params.image_width    = g_width;
    state.params.image_height   = g_height;
    state.params.origin_x       = g_width / 2;
    state.params.origin_y       = g_height / 2;
    state.params.handle         = state.gas_handle;
    state.params.device_idx     = state.device_idx;
    state.params.num_devices    = numDevices;
    state.params.mipLevelBias   = g_mipLevelBias;

    state.params.eye = g_camera.eye();
    g_camera.UVWFrame( state.params.U, state.params.V, state.params.W );

    if( state.d_params == nullptr )
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &state.d_params ), sizeof( Params ) ) );
}


// Returns number of requests processed (over all streams and devices).
unsigned int performLaunches( otk::CUDAOutputBuffer<uchar4>& output_buffer, std::vector<PerDeviceSampleState>& states )
{
    auto startTime = std::chrono::steady_clock::now();

    const uint32_t bucketCountX = ( g_width  + g_bucketSize - 1 ) / g_bucketSize;
    const uint32_t bucketCountY = ( g_height + g_bucketSize - 1 ) / g_bucketSize;
    const uint32_t numBuckets   = bucketCountX * bucketCountY;

    uchar4* outputPtr = output_buffer.map();
    for( auto& state : states )
        state.params.result_buffer = outputPtr;

    uint32_t numRequestsProcessed = 0;
    uint32_t bucketIdx            = 0;

    while( bucketIdx < numBuckets )
    {
        for( auto& state : states )
        {
            uint32_t cur_index = bucketIdx++;
            if( cur_index >= numBuckets )
                continue;

            state.params.bucket_index  = cur_index;
            state.params.bucket_width  = g_bucketSize;
            state.params.bucket_height = g_bucketSize;

            OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
            state.demandLoader->launchPrepare( state.stream, state.params.demandTextureContext );

            initLaunchParams( state, static_cast<unsigned int>( states.size() ) );

            // Perform the rendering launches
            OTK_ERROR_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( state.d_params ),
                                       reinterpret_cast<CUdeviceptr>( &state.params ), sizeof( Params ), state.stream ) );
            OTK_ERROR_CHECK( optixLaunch( state.pipeline,
                                      state.stream,
                                      reinterpret_cast<CUdeviceptr>( state.d_params ),
                                      sizeof( Params ),
                                      &state.sbt,
                                      state.params.bucket_width,   // launch width
                                      state.params.bucket_height,  // launch height
                                      1                            // launch depth
                                      ) );

            // Initiate asynchronous request processing for the previous launch
            state.ticket = state.demandLoader->processRequests( state.stream, state.params.demandTextureContext );
        }

        // Wait for any outstanding requests
        for( auto& state : states )
        {
            state.ticket.wait();
            assert( state.ticket.numTasksTotal() >= 0 );
            numRequestsProcessed += state.ticket.numTasksTotal();
        }
    }
    output_buffer.unmap();

    ++g_totalLaunches;
    g_totalLaunchTime += std::chrono::duration<double>( std::chrono::steady_clock::now() - startTime ).count();

    return numRequestsProcessed;
}


int main( int argc, char* argv[] )
{
    std::string outfile;
    float textureScale = 4.0f;

    // Image credit: CC0Textures.com (https://cc0textures.com/view.php?tex=Bricks12)
    // Licensed under the Creative Commons CC0 License.
    std::string textureFile = "Bricks12_col.exr";  // use --texture "" for procedural texture

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        bool              lastArg = ( i == argc - 1 );

        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( ( arg == "--file" || arg == "-f" ) && !lastArg )
        {
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            otk::parseDimensions( arg.substr( 6 ).c_str(), g_width, g_height );
        }
        else if( ( arg == "--texture" || arg == "-t" ) && !lastArg )
        {
            textureFile = argv[++i];
        }
        else if( arg.substr( 0, 13 ) == "--textureDim=" )
        {
            otk::parseDimensions( arg.substr( 13 ).c_str(), g_textureWidth, g_textureHeight );
        }
        else if( ( arg == "--bias" || arg == "-b" ) && !lastArg )
        {
            g_mipLevelBias = static_cast<float>( atof( argv[++i] ) );
        }
        else if( arg == "--textureScale" && !lastArg )
        {
            textureScale = static_cast<float>( atof( argv[++i] ) );
        }
        else if( arg == "--bucketSize" && !lastArg )
        {
            g_bucketSize = atoi( argv[++i] );
            if( g_bucketSize <= 0 )
            {
                std::cerr << "Warning: Bucket size must be greater than 0. Setting bucket size to 256" << std::endl;
                g_bucketSize = 256;
            }
        }
        else if( arg == "--numThreads" && !lastArg )
        {
            g_numThreads = atoi( argv[++i] );
        }
#ifdef OPTIX_SAMPLE_USE_CORE_EXR
        else if( arg == "--useCoreEXR" && !lastArg )
        {
            g_useCoreExr = std::string( argv[++i] ) != "false";
        }
#endif
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCameraState();

        // Initialize OptiX
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
        OTK_ERROR_CHECK( optixInit() );

        std::vector<PerDeviceSampleState> states;
        createContexts( states );

        std::string                  directory( getSourceDir() + "/Textures/" );
        std::shared_ptr<ImageSource> imageSource = imageSources::createImageSource( textureFile, directory );

        // Set up OptiX per-device states and demand loaders
        // The texture id is passed to the closest hit shader via a hit group record in the SBT.
        // The texture sampler array (indexed by texture id) is passed as a launch parameter.
        demandLoading::Options options{};
        options.maxThreads = g_numThreads;  // maximum threads to use when processing page requests
        TextureDescriptor    texDesc = makeTextureDescription();

        // Set up OptiX per-device states
        for( PerDeviceSampleState& state : states )
        {
            OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
            state.demandLoader.reset( createDemandLoader( options ), destroyDemandLoader );
            const DemandTexture& texture = state.demandLoader->createTexture( imageSource, texDesc );

            buildAccel( state );
            createModule( state );
            createProgramGroups( state );
            createPipeline( state );
            createSBT( state, texture, textureScale, 0.f /*textureLod*/ );
        }

        // Create the output buffer to hold the rendered image
        otk::CUDAOutputBuffer<uchar4> outputBuffer( otk::CUDAOutputBufferType::ZERO_COPY, g_width, g_height );

        // Perform launches (launch until there are no more requests to fill), up to
        // the maximum number of launches.
        unsigned int numFilled   = 0;
        const int    maxLaunches = 1024;
        do
        {
            numFilled = performLaunches( outputBuffer, states );
            g_totalRequests += numFilled;
        } while( numFilled > 0 && g_totalLaunches < maxLaunches );

        std::cout << "Launches:              " << g_totalLaunches << "\n";
        std::cout << "Avg. launch time:      " << ( 1000.0 * g_totalLaunchTime / g_totalLaunches ) << " ms\n";
        std::cout << "Texture tile requests: " << g_totalRequests << "\n";

        // Display result image
        {
            otk::ImageBuffer buffer;
            buffer.data         = outputBuffer.getHostPointer();
            buffer.width        = g_width;
            buffer.height       = g_height;
            buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;
            if( outfile.empty() )
                otk::displayBufferWindow( argv[0], buffer );
            else
                otk::saveImage( outfile.c_str(), buffer, false );
        }

        // Clean up the states, deleting their resources
        for( PerDeviceSampleState& state : states )
            cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
