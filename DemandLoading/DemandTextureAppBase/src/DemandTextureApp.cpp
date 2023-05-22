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

#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/Gui.h>
#include <OptiXToolkit/Util/Logger.h>

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>

#include <OptiXToolkit/DemandTextureAppBase/LaunchParams.h>
#include <OptiXToolkit/DemandTextureAppBase/PerDeviceOptixState.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h>

#include <GLFW/glfw3.h>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

namespace demandTextureApp
{

DemandTextureApp::DemandTextureApp( const char* appName, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
    : m_windowWidth( width )
    , m_windowHeight( height )
    , m_outputFileName( outFileName )
{
    // Init cuda on all devices
    CUDA_CHECK( cudaSetDevice( 0 ) );
    CUDA_CHECK( cudaFree( 0 ) );
    
    int numDevices;
    CUDA_CHECK( cuDeviceGetCount( &numDevices ) );
    for( unsigned int deviceIdx = 1; deviceIdx < numDevices; ++deviceIdx )
    {
        CUDA_CHECK( cudaSetDevice( deviceIdx ) );
        CUDA_CHECK( cudaFree( 0 ) );
    }
    CUDA_CHECK( cudaSetDevice( 0 ) );

    // Create display window for interactive mode
    if( isInteractive() )
    {
        m_window = otk::initUI( appName, width, height );
        otk::initGL();
        m_glDisplay.reset( new otk::GLDisplay( otk::BufferImageFormat::UNSIGNED_BYTE4 ) );
        setGLFWCallbacks( this );
    }

    // Reset the output buffer
    glInterop = glInterop && isInteractive() && (numDevices==1);
    otk::CUDAOutputBufferType outputBufferType =
        glInterop ? otk::CUDAOutputBufferType::GL_INTEROP : otk::CUDAOutputBufferType::ZERO_COPY;
    m_outputBuffer.reset( new otk::CUDAOutputBuffer<uchar4>( outputBufferType, m_windowWidth, m_windowHeight ) );

    // Create the per device optix states
    OPTIX_CHECK( optixInit() );
    m_perDeviceOptixStates.resize( numDevices );
    for( unsigned int device_idx = 0; device_idx < numDevices; ++device_idx )
        m_perDeviceOptixStates[device_idx].device_idx = device_idx;

    initView();
}


DemandTextureApp::~DemandTextureApp()
{
    for( PerDeviceOptixState state : m_perDeviceOptixStates )
        cleanupState( state );
    otk::cleanupUI( m_window );
}


//------------------------------------------------------------------------------
// OptiX setup
//------------------------------------------------------------------------------

void DemandTextureApp::createContext( PerDeviceOptixState& state )
{
    // Initialize CUDA on this device
    CUDA_CHECK( cudaSetDevice( state.device_idx ) );
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext                 cuCtx   = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    otk::util::setLogger( options );
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &state.context ) );

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
}


void DemandTextureApp::buildAccel( PerDeviceOptixState& state )
{
    // The code below creates a single bounding box, suitable for a single object contained in 
    // [-1,-1,-1] to [1,1,1].  Apps with more complicated geometry should override buildAccel.

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    OptixAabb   aabb = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    CUdeviceptr d_aabb_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};

    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = 1;

    uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild( state.context,
                                  0,  // CUDA stream
                                  &accel_options, &aabb_input,
                                  1,  // num build inputs
                                  d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_gas_and_compacted_size,
                                  gas_buffer_sizes.outputSizeInBytes, &state.gas_handle,
                                  &emitProperty,  // emitted property list
                                  1               // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
    CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer,
                                        compacted_gas_size, &state.gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


void DemandTextureApp::createModule( PerDeviceOptixState& state, const char* moduleCode, size_t codeSize )
{
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 2;
    state.pipeline_compile_options.numAttributeValues    = 6;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreate( state.context, &module_compile_options, &state.pipeline_compile_options,
                                        moduleCode, codeSize, log, &sizeof_log, &state.ptx_module ) );
}


void DemandTextureApp::createProgramGroups( PerDeviceOptixState& state )
{
    // Make program groups for raygen, miss, and hitgroup
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.miss_prog_group ) );

    OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroup_prog_group_desc.hitgroup.moduleIS            = m_scene_is_triangles ? nullptr : state.ptx_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = m_scene_is_triangles ? nullptr : "__intersection__is";
    OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.hitgroup_prog_group ) );
}


void DemandTextureApp::createPipeline( PerDeviceOptixState& state )
{
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    const uint32_t    max_trace_depth  = 1;
    OptixProgramGroup program_groups[] = {state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace_depth;
    OPTIX_CHECK_LOG( optixPipelineCreate( state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                          program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ), log,
                                          &sizeof_log, &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
#if OPTIX_VERSION < 70700
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
#else
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, state.pipeline ) );
#endif
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                             0,  // maxCCDepth
                                             0,  // maxDCDEpth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}


void DemandTextureApp::createSBT( PerDeviceOptixState& state )
{
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
    RayGenSbtRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
    MissSbtRecord ms_sbt;
    ms_sbt.data.background_color = m_backgroundColor;
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &ms_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
    HitGroupSbtRecord hg_sbt;
    hg_sbt.data.texture_id = m_textureIds[0]; 
    OPTIX_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hg_sbt ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( hitgroup_record ), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice ) );

    state.sbt.raygenRecord                = raygen_record;
    state.sbt.missRecordBase              = miss_record;
    state.sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    state.sbt.missRecordCount             = 1;
    state.sbt.hitgroupRecordBase          = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    state.sbt.hitgroupRecordCount         = 1;
}


void DemandTextureApp::cleanupState( PerDeviceOptixState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hitgroup_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
}


void DemandTextureApp::initOptixPipelines( const char* moduleCode )
{
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        createContext( state );
        buildAccel( state );
        createModule( state, moduleCode, ::strlen( moduleCode ) );
        createProgramGroups( state );
        createPipeline( state );
        createSBT( state );
    }
}


//------------------------------------------------------------------------------
// Demand Loading
//------------------------------------------------------------------------------

demandLoading::TextureDescriptor DemandTextureApp::makeTextureDescriptor( CUaddress_mode addressMode, CUfilter_mode filterMode )
{
    demandLoading::TextureDescriptor texDesc{};
    texDesc.addressMode[0]   = addressMode;
    texDesc.addressMode[1]   = addressMode;
    texDesc.filterMode       = filterMode;
    texDesc.mipmapFilterMode = filterMode;
    texDesc.maxAnisotropy    = 16;

    return texDesc;
}


imageSource::ImageSource* DemandTextureApp::createExrImage( const char* filePath )
{
    try
    {
        if( filePath == nullptr || filePath[0] == '\0' )
            return nullptr;
        return new imageSource::EXRREADER( filePath, true );
    }
    catch( ... )
    {
    }
    return nullptr;
}


void DemandTextureApp::initDemandLoading()
{
    // Default to final frame rendering. Use all of the device memory available for textures,
    // and set maxRequests to high values to reduce the total number of launches.
    unsigned int maxTexMem     = 0;  // unlimited
    unsigned int maxRequests   = 8192;
    unsigned int maxStalePages = 4096;

    // In interactive mode, reduce the max texture memory to exercise eviction. Set maxRequests to
    // a low value to keep the system from being bogged down by pullRequests for any single launch.
    if( isInteractive() )
    {
        maxTexMem     = 2ULL * 1024 * 1024 * 1024;
        maxRequests   = 128;
        maxStalePages = 128;
    }

    demandLoading::Options options{};
    options.numPages            = 64 * 1024 * 1024;  // max possible texture tiles in virtual memory system
    options.numPageTableEntries = 1024 * 1024;       // 2 * max possible textures
    options.maxRequestedPages   = maxRequests;       // max requests to pull from device in pullRequests
    options.maxFilledPages      = 2 * maxRequests;   // number of slots to push mappings back to device
    options.maxStalePages       = maxStalePages;     // max stale pages to pull from the device in pullRequests
    options.maxEvictablePages   = 0;                 // evicatable pages currently not used
    options.maxInvalidatedPages = maxStalePages;     // max slots to push invalidated pages back to device
    options.maxStagedPages      = maxStalePages;     // max pages to stage for eviction
    options.maxRequestQueueSize = maxStalePages;     // max size of host-side request queue
    options.useLruTable         = true;              // Whether to use an LRU table for eviction
    options.maxTexMemPerDevice  = maxTexMem;         // max texture to use before starting eviction (0 is unlimited)
    options.maxPinnedMemory     = 64 * 1024 * 1024;  // max pinned memory to reserve for transfers.
    options.maxThreads          = 0;                 // request threads. (0 is std::thread::hardware_concurrency)
    options.evictionActive      = true;              // turn on or off eviction
    options.useSparseTextures   = true;              // use sparse or dense textures

    m_demandLoader =
        std::shared_ptr<demandLoading::DemandLoader>( createDemandLoader( options ), demandLoading::destroyDemandLoader );
}


void DemandTextureApp::printDemandLoadingStats()
{
    demandLoading::Statistics stats = m_demandLoader->getStatistics();
   
    std::cout << std::fixed << std::setprecision( 1 );
    std::cout << "\n============================================\n";
    std::cout << "Demand Loading Stats\n";
    std::cout << "============================================\n";
    std::cout << "Launch cycles:            " << m_launchCycles << "\n";
    std::cout << "Num textures:             " << stats.numTextures << "\n";
    std::cout << "Virtual texture Size:     " << stats.virtualTextureBytes / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "Tiles read from disk:     " << stats.numTilesRead << "\n";
    
    std::cout << "Max device memory used:   ";
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        const double deviceMemory = stats.perDevice[state.device_idx].memoryUsed / (1024.0 * 1024.0);
        std::cout << "[GPU-" << state.device_idx << ": " << deviceMemory << " MiB]  ";
    }
    std::cout << "\n";

    std::cout << "Texture data transferred: ";
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        const size_t tilesTransferred = stats.perDevice[state.device_idx].bytesTransferred / 65536;
        const double transferData = stats.perDevice[state.device_idx].bytesTransferred / (1024.0 * 1024.0);
        std::cout << "[GPU-" << state.device_idx << ": " << tilesTransferred << " tiles (" << transferData << " MiB)]  ";
    }
    std::cout << "\n";

    std::cout << "Evicted tiles:            ";
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        std::cout << "[GPU-" << state.device_idx << ": " << stats.perDevice[state.device_idx].numEvictions << "]  ";
    }
    std::cout << "\n" << std::endl;
}


//------------------------------------------------------------------------------
// OptiX launches
//------------------------------------------------------------------------------

void DemandTextureApp::initView()
{
    // Set the view so that the square (0,0,0) to (1,1,0) exactly covers the viewport.
    setView( float3{0.5f, 0.5f, 1.0f}, float3{0.5f, 0.5f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, 53.130102354f );
}

void DemandTextureApp::setView( float3 eye, float3 lookAt, float3 up, float fovY )
{
    float aspectRatio = static_cast<float>( m_windowWidth ) / static_cast<float>( m_windowHeight );
    m_camera = otk::Camera( eye, lookAt, up, fovY, aspectRatio );
    m_subframeId = 0;
}

void DemandTextureApp::panCamera( float3 pan )
{
    m_camera.setEye( m_camera.eye() + pan );
    m_camera.setLookat( m_camera.lookat() + pan );
    m_subframeId = 0;
}

void DemandTextureApp::zoomCamera( float zoom )
{
    float tanVal = zoom * tanf( m_camera.fovY() * (M_PI / 360.0f) );
    m_camera.setFovY( atanf( tanVal ) * 360.0f / M_PI );
    m_subframeId = 0;
}

void DemandTextureApp::rotateCamera( float rot )
{
    float3 U, V, W;
    m_camera.UVWFrame( U, V, W );
    W = float3{ W.x * cosf(rot) - W.y * sinf(rot), W.x * sinf(rot) + W.y * cosf(rot), W.z };
    m_camera.setLookat( m_camera.eye() + W );
    m_subframeId = 0;
}

void DemandTextureApp::initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices )
{
    state.params.image_dim.x        = m_outputBuffer->width();
    state.params.image_dim.y        = m_outputBuffer->height();
    state.params.traversable_handle = state.gas_handle;
    state.params.device_idx         = state.device_idx;
    state.params.num_devices        = numDevices;
    state.params.display_texture_id = m_textureIds[0];
    state.params.interactive_mode   = isInteractive();
    state.params.render_mode        = m_render_mode;
    state.params.projection         = m_projection;
    state.params.lens_width         = m_lens_width;
    state.params.camera.eye         = m_camera.eye();
    m_camera.UVWFrame( state.params.camera.U, state.params.camera.V, state.params.camera.W );

    // Make sure a device-side copy of the params has been allocated
    if( state.d_params == nullptr )
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );
}


unsigned int DemandTextureApp::performLaunches( )
{
    const unsigned int numDevices     = static_cast<unsigned int>( m_perDeviceOptixStates.size() );
    unsigned int numRequestsProcessed = 0;

    // Resize the output buffer if needed. 
    m_outputBuffer->resize( m_windowWidth, m_windowHeight );
   
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        // Wait on the ticket from the previous launch
        CUDA_CHECK( cudaSetDevice( state.device_idx ) );
        state.ticket.wait();
        numRequestsProcessed += static_cast<unsigned int>( state.ticket.numTasksTotal() );

        // Call launchPrepare to synchronize new texture samplers and texture info to device memory,
        // and allocate device memory for the demand texture context.
        m_demandLoader->launchPrepare( state.stream, state.params.demand_texture_context );

        // Finish initialization of the launch params.
        state.params.result_buffer = m_outputBuffer->map();
        initLaunchParams( state, numDevices );

        // Copy launch params to device.  Note: cudaMemcpy measured faster than cudaMemcpyAsync 
        // for this application, so we use it here.  Other applications may differ.
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.d_params ), &state.params, sizeof( Params ), cudaMemcpyHostToDevice ) );

        // Peform the OptiX launch, with each device doing a part of the work
        unsigned int launchHeight = ( state.params.image_dim.y + numDevices - 1 ) / numDevices;
        OPTIX_CHECK( optixLaunch( state.pipeline,  // OptiX pipeline
                                  state.stream,    // Stream for launch and demand loading
                                  reinterpret_cast<CUdeviceptr>( state.d_params ),  // Launch params
                                  sizeof( Params ),                                 // Param size in bytes
                                  &state.sbt,                                       // Shader binding table
                                  state.params.image_dim.x,                         // Launch width
                                  launchHeight,                                     // Launch height
                                  1                                                 // launch depth
                                  ) );

        // Begin to process demand load requests. This asynchronously pulls a batch of requests 
        // from the device and places them in a queue for processing.  The progress of the batch
        // can be polled using the returned ticket.
        state.ticket = m_demandLoader->processRequests( state.stream, state.params.demand_texture_context );
        
        // Unmap the output buffer. The device pointer from map should not be used after this call.
        m_outputBuffer->unmap();
    }

    return numRequestsProcessed;
}


void DemandTextureApp::startLaunchLoop()
{
    int numFilled = 0;
    if( isInteractive() )
    {
        while( !glfwWindowShouldClose( getWindow() ) )
        {
            glfwPollEvents();
            pollKeys();
            numFilled = performLaunches();
            m_numFilledRequests += numFilled;
            ++m_launchCycles;
            if( numFilled == 0 )
                m_subframeId++;
            displayFrame();
            drawGui();
            glfwSwapBuffers( getWindow() );
        }
    }
    else 
    {
        // Launch repeatedly until there are no more requests to fill.
        do
        {
            numFilled = performLaunches();
            m_numFilledRequests += numFilled;
            ++m_launchCycles;
            if( numFilled == 0 )
                m_subframeId++;
        } while( numFilled > 0 || m_launchCycles < m_minLaunches );

        saveImage();
    }
}


//------------------------------------------------------------------------------
// Display
//------------------------------------------------------------------------------

void DemandTextureApp::drawGui()
{
    sutil::beginFrameImGui();
    sutil::displayFPS( m_launchCycles );
    sutil::endFrameImGui();
}

void DemandTextureApp::displayFrame()
{
    m_glDisplay->display( m_outputBuffer->width(), m_outputBuffer->height(), m_windowWidth, m_windowHeight,
                          m_outputBuffer->getPBO() );
}

void DemandTextureApp::saveImage()
{
    otk::ImageBuffer buffer;
    buffer.data         = m_outputBuffer->getHostPointer();
    buffer.width        = m_outputBuffer->width();
    buffer.height       = m_outputBuffer->height();
    buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;
    otk::saveImage( m_outputFileName.c_str(), buffer, false );
}

 void DemandTextureApp::resetAccumulator()
 {
    size_t accumulatorSize = sizeof(float4) * m_windowWidth * m_windowHeight;
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        CUDA_CHECK( cudaSetDevice( state.device_idx ) );
        CUDA_CHECK( cudaFree( state.params.accum_buffer ) );
        CUDA_CHECK( cudaMalloc( &state.params.accum_buffer, accumulatorSize ) );
    }
 }

//------------------------------------------------------------------------------
// User Interaction via GLFW
//------------------------------------------------------------------------------

void DemandTextureApp::mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    glfwGetCursorPos( window, &m_mousePrevX, &m_mousePrevY );
    m_mouseButton = ( action == GLFW_PRESS ) ? button : NO_BUTTON;
}

void DemandTextureApp::cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    float dx = static_cast<float>( xpos - m_mousePrevX );
    float dy = static_cast<float>( ypos - m_mousePrevY );

    if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )  // pan camera
    {
        float moveScale = 2.0f * tanf( m_camera.fovY() * M_PIf / 360.0f ) / m_windowHeight;
        panCamera( float3{ -dx * moveScale, dy * moveScale, 0.0f } );
    }
    else if( m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )  // zoom camera
    {
        zoomCamera( powf( 1.003f, ( dy - dx ) ) );
    }

    m_mousePrevX = xpos;
    m_mousePrevY = ypos;
}

void DemandTextureApp::windowSizeCallback( GLFWwindow* window, int32_t width, int32_t height )
{
    m_windowWidth  = width;
    m_windowHeight = height;
    m_camera.setAspectRatio( static_cast<float>( m_windowWidth ) / static_cast<float>( m_windowHeight ) );
    if( m_perDeviceOptixStates[0].params.accum_buffer != nullptr )
        resetAccumulator();
    m_subframeId = 0;
}

void DemandTextureApp::pollKeys()
{
    const float pan  = 0.003f * ( m_camera.fovY() * M_PIf / 360.0f );
    const float zoom = 1.003f;

    if( glfwGetKey( getWindow(), GLFW_KEY_A ) )
        panCamera( float3{-pan, 0.0f, 0.0f} );
    if( glfwGetKey( getWindow(), GLFW_KEY_D ) )
        panCamera( float3{pan, 0.0f, 0.0f} );
    if( glfwGetKey( getWindow(), GLFW_KEY_S ) )
        panCamera( float3{0.0f, -pan, 0.0f} );
    if( glfwGetKey( getWindow(), GLFW_KEY_W ) )
        panCamera( float3{0.0f, pan, 0.0f} );
    if( glfwGetKey( getWindow(), GLFW_KEY_Q ) )
        zoomCamera( zoom );
    if( glfwGetKey( getWindow(), GLFW_KEY_E ) )
        zoomCamera( 1.0f / zoom );
}

void DemandTextureApp::keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    if( action != GLFW_PRESS )
        return;

    if( key == GLFW_KEY_ESCAPE )
        glfwSetWindowShouldClose( window, true );
    else if( key == GLFW_KEY_C )
        initView();
}

void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    DemandTextureApp* app = reinterpret_cast<DemandTextureApp*>( glfwGetWindowUserPointer( window ) );
    app->mouseButtonCallback( window, button, action, mods );
}
void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    DemandTextureApp* app = reinterpret_cast<DemandTextureApp*>( glfwGetWindowUserPointer( window ) );
    app->cursorPosCallback( window, xpos, ypos );
}
void windowSizeCallback( GLFWwindow* window, int32_t width, int32_t height )
{
    DemandTextureApp* app = reinterpret_cast<DemandTextureApp*>( glfwGetWindowUserPointer( window ) );
    app->windowSizeCallback( window, width, height );
}
void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    DemandTextureApp* app = reinterpret_cast<DemandTextureApp*>( glfwGetWindowUserPointer( window ) );
    app->keyCallback( window, key, scancode, action, mods );
}

void setGLFWCallbacks( DemandTextureApp* app )
{
    glfwSetWindowUserPointer( app->getWindow(), app );
    glfwSetMouseButtonCallback( app->getWindow(), mouseButtonCallback );
    glfwSetCursorPosCallback( app->getWindow(), cursorPosCallback );
    glfwSetKeyCallback( app->getWindow(), keyCallback );
    glfwSetWindowSizeCallback( app->getWindow(), windowSizeCallback );
}

} // namespace demandTextureApp
