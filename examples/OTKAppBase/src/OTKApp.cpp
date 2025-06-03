// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cmath>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/Gui.h>
#include <OptiXToolkit/Gui/glfw3.h>

#include <OptiXToolkit/OptiXMemory/CompileOptions.h>

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/Util/Logger.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppPerDeviceOptixState.h>
#include <OptiXToolkit/OTKAppBase/OTKApp.h>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

#ifndef M_PI
constexpr float M_PI = 3.14159265358979323846f;
#endif

using namespace otk;  // for vec_math operators
using namespace demandLoading;

namespace otkApp
{

OTKApp::OTKApp( const char* appName, unsigned int width, unsigned int height,
                const std::string& outFileName, bool glInterop, OTKAppUIMode uiMode )
    : m_windowWidth( width )
    , m_windowHeight( height )
    , m_outputFileName( outFileName )
    , m_uiMode( uiMode )
{
    // Initialize CUDA and OptiX, create per device optix states
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    for( unsigned int deviceIndex : getDemandLoadDevices( m_useSparseTextures ) )
    {
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
        m_perDeviceOptixStates.emplace_back();
        m_perDeviceOptixStates.back().device_idx = static_cast<int>( deviceIndex );
    }
    OTK_ERROR_CHECK( optixInit() );

    // Create display window for interactive mode
    if( isInteractive() )
    {
        m_window = otk::initUI( appName, width, height );
        otk::initGL();
        m_glDisplay.reset( new otk::GLDisplay( otk::BufferImageFormat::UNSIGNED_BYTE4 ) );
        setGLFWCallbacks( this );
    }

    // Reset the output buffer
    glInterop = glInterop && isInteractive() && ( m_perDeviceOptixStates.size() == 1U );
    otk::CUDAOutputBufferType outputBufferType =
        glInterop ? otk::CUDAOutputBufferType::GL_INTEROP : otk::CUDAOutputBufferType::ZERO_COPY;
    m_outputBuffer.reset( new otk::CUDAOutputBuffer<uchar4>( outputBufferType, m_windowWidth, m_windowHeight ) );

    initView();
}


//------------------------------------------------------------------------------
// OptiX setup
//------------------------------------------------------------------------------

void OTKApp::createContext( OTKAppPerDeviceOptixState& state )
{
    CUcontext                 cuCtx   = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    otk::util::setLogger( options );
    OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &options, &state.context ) );

    OTK_ERROR_CHECK( cudaStreamCreate( &state.stream ) );
}

void OTKApp::buildAccel( OTKAppPerDeviceOptixState& state )
{
    // Copy vertex data to device
    void* d_vertices = nullptr;
    const size_t vertices_size_bytes = m_vertices.size() * sizeof( float4 );
    OTK_ERROR_CHECK( cudaMalloc( &d_vertices, vertices_size_bytes ) );
    OTK_ERROR_CHECK( cudaMemcpy( d_vertices, m_vertices.data(), vertices_size_bytes, cudaMemcpyHostToDevice ) );
    state.d_vertices = reinterpret_cast<CUdeviceptr>( d_vertices );

    // Copy material indices to device
    void* d_material_indices = nullptr;
    const size_t material_indices_size_bytes = m_material_indices.size() * sizeof( uint32_t );
    OTK_ERROR_CHECK( cudaMalloc( &d_material_indices, material_indices_size_bytes ) );
    OTK_ERROR_CHECK( cudaMemcpy( d_material_indices, m_material_indices.data(), material_indices_size_bytes, cudaMemcpyHostToDevice ) );

    // Make triangle input flags (one per sbt record). 
    std::vector<uint32_t> triangle_input_flags( m_materials.size(), m_optixGeometryFlags );

    // Make GAS accel build inputs
    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = static_cast<uint32_t>( sizeof( float4 ) );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( m_vertices.size() );
    triangle_input.triangleArray.vertexBuffers               = &state.d_vertices;
    triangle_input.triangleArray.flags                       = triangle_input_flags.data();
    triangle_input.triangleArray.numSbtRecords               = static_cast<uint32_t>( triangle_input_flags.size() );
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = reinterpret_cast<CUdeviceptr>( d_material_indices );
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    // Make accel options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory usage for accel build
    OptixAccelBufferSizes gas_buffer_sizes;
    const unsigned int num_build_inputs = 1;
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &triangle_input, num_build_inputs, &gas_buffer_sizes ) );

    // Allocate temporary buffer needed for accel build
    void* d_temp_buffer = nullptr;
    OTK_ERROR_CHECK( cudaMalloc( &d_temp_buffer, gas_buffer_sizes.tempSizeInBytes ) );

    // Allocate output buffer for (non-compacted) accel build result, and also compactedSize property.
    void* d_buffer_temp_output_gas_and_compacted_size = nullptr;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    OTK_ERROR_CHECK( cudaMalloc( &d_buffer_temp_output_gas_and_compacted_size, compactedSizeOffset + 8 ) );

    // Set up the accel build to return the compacted size, so compaction can be run after the build
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    // Finally perform the accel build
    OTK_ERROR_CHECK( optixAccelBuild(
                state.context,
                CUstream{0},
                &accel_options,
                &triangle_input,
                num_build_inputs,
                reinterpret_cast<CUdeviceptr>( d_temp_buffer ),
                gas_buffer_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>( d_buffer_temp_output_gas_and_compacted_size ),
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                &emitProperty,
                1
                ) );

    // Delete temporary buffers used for the accel build
    OTK_ERROR_CHECK( cudaFree( d_temp_buffer ) );
    OTK_ERROR_CHECK( cudaFree( d_material_indices ) );

    // Copy the size of the compacted GAS accel back from the device
    size_t compacted_gas_size;
    OTK_ERROR_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    // If compaction reduces the size of the accel, copy to a new buffer and delete the old one
    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );
        // use handle as input and output
        OTK_ERROR_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );
        OTK_ERROR_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = reinterpret_cast<CUdeviceptr>( d_buffer_temp_output_gas_and_compacted_size );
    }
}


SurfaceTexture OTKApp::makeSurfaceTex( int kd, int kdtex, int ks, int kstex, int kt, int kttex, float roughness, float ior )
{
    SurfaceTexture tex;
    tex.emission     = ColorTex{ float3{ 0.0f, 0.0f, 0.0f }, -1 };
    tex.diffuse      = ColorTex{ float3{ ((kd>>16)&0xff)/255.0f, ((kd>>8)&0xff)/255.0f, ((kd>>0)&0xff)/255.0f }, kdtex };
    tex.specular     = ColorTex{ float3{ ((ks>>16)&0xff)/255.0f, ((ks>>8)&0xff)/255.0f, ((ks>>0)&0xff)/255.0f }, kstex };
    tex.transmission = ColorTex{ float3{ ((kt>>16)&0xff)/255.0f, ((kt>>8)&0xff)/255.0f, ((kt>>0)&0xff)/255.0f }, kttex };
    tex.roughness    = roughness;
    tex.ior          = ior;
    return tex;
}


void OTKApp::addShapeToScene( std::vector<Vert>& shape, unsigned int materialId )
{
    for( unsigned int i=0; i<shape.size(); ++i )
    {
        m_vertices.push_back( make_float4( shape[i].p ) );
        m_normals.push_back( shape[i].n );
        m_tex_coords.push_back( shape[i].t );
        if( i % 3 == 0 )
            m_material_indices.push_back( materialId );
    }
}


void OTKApp::copyGeometryToDevice()
{
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );

        // m_vertices copied in buildAccel
        // m_material_indices copied in createSBT
        // m_materials copied in createSBT

        // m_normals
        OTK_ERROR_CHECK( cudaMalloc( &state.d_normals, m_normals.size() * sizeof(float3) ) );
        OTK_ERROR_CHECK( cudaMemcpy( state.d_normals, m_normals.data(),  m_normals.size() * sizeof(float3), cudaMemcpyHostToDevice ) );

        // m_tex_coords
        OTK_ERROR_CHECK( cudaMalloc( &state.d_tex_coords, m_tex_coords.size() * sizeof(float2) ) );
        OTK_ERROR_CHECK( cudaMemcpy( state.d_tex_coords, m_tex_coords.data(),  m_tex_coords.size() * sizeof(float2), cudaMemcpyHostToDevice ) );
    }
}




void OTKApp::createModule( OTKAppPerDeviceOptixState& state, const char* moduleCode, size_t codeSize )
{
    OptixModuleCompileOptions module_compile_options{};
    otk::configModuleCompileOptions( module_compile_options );

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 2;
    state.pipeline_compile_options.numAttributeValues    = 6;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OTK_ERROR_CHECK_LOG( optixModuleCreate( state.context, &module_compile_options, &state.pipeline_compile_options,
                                        moduleCode, codeSize, log, &sizeof_log, &state.optixir_module ) );
}


void OTKApp::createProgramGroups( OTKAppPerDeviceOptixState& state )
{
    // Make program groups for raygen, miss, and hitgroup
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = (m_raygenName) ? state.optixir_module : nullptr;
    raygen_prog_group_desc.raygen.entryFunctionName = m_raygenName;
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = (m_missName) ? state.optixir_module : nullptr;
    miss_prog_group_desc.miss.entryFunctionName = m_missName;
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.miss_prog_group ) );

    OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = (m_closestHitName) ? state.optixir_module : nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = m_closestHitName;
    hitgroup_prog_group_desc.hitgroup.moduleAH            = (m_anyhitName) ? state.optixir_module : nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = m_anyhitName;
    hitgroup_prog_group_desc.hitgroup.moduleIS            = (m_intersectName) ? state.optixir_module : nullptr;;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = m_intersectName;
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.hitgroup_prog_group ) );
}


void OTKApp::createPipeline( OTKAppPerDeviceOptixState& state )
{
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    const uint32_t    max_trace_depth  = 1;
    OptixProgramGroup program_groups[] = {state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace_depth;
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
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OTK_ERROR_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void OTKApp::createSBT( OTKAppPerDeviceOptixState& state )
{
    // Raygen record
    void*  d_raygen_record = nullptr;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &d_raygen_record, raygen_record_size ) );
    RayGenSbtRecord raygen_record = {};
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &raygen_record ) );
    OTK_ERROR_CHECK( cudaMemcpy( d_raygen_record, &raygen_record, raygen_record_size, cudaMemcpyHostToDevice ) );

    // Miss record
    void* d_miss_record = nullptr;
    const size_t miss_record_size = sizeof( MissSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &d_miss_record, miss_record_size ) );
    MissSbtRecord miss_record;
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &miss_record ) );
    OTK_ERROR_CHECK( cudaMemcpy( d_miss_record, &miss_record, miss_record_size, cudaMemcpyHostToDevice ) );

    // Hitgroup records (one for each material)
    const unsigned int MAT_COUNT = static_cast<unsigned int>( m_materials.size() );
    void* d_hitgroup_records = nullptr;
    const size_t hitgroup_record_size = sizeof( TriangleHitGroupSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &d_hitgroup_records, hitgroup_record_size * MAT_COUNT ) );
    std::vector<TriangleHitGroupSbtRecord> hitgroup_records( MAT_COUNT );
    for( unsigned int mat_idx = 0; mat_idx < MAT_COUNT; ++mat_idx )
    {
        OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hitgroup_records[mat_idx] ) );
        OTKAppTriangleHitGroupData* hg_data = &hitgroup_records[mat_idx].data;
        // Copy material definition, and then fill in device-specific values for vertices, normals, tex_coords
        *hg_data = m_materials[mat_idx];
        hg_data->vertices = reinterpret_cast<float4*>( state.d_vertices );
        hg_data->normals = state.d_normals;
        hg_data->tex_coords = state.d_tex_coords;
    }
    OTK_ERROR_CHECK( cudaMemcpy( d_hitgroup_records, &hitgroup_records[0], hitgroup_record_size * MAT_COUNT, cudaMemcpyHostToDevice ) );

    // Set up SBT
    state.sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>( d_raygen_record );
    state.sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>( d_miss_record );
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = 1;
    state.sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>( d_hitgroup_records );
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = MAT_COUNT;
}

void OTKApp::cleanupState( OTKAppPerDeviceOptixState& state )
{
    OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
    OTK_ERROR_CHECK( optixPipelineDestroy( state.pipeline ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.hitgroup_prog_group ) );
    OTK_ERROR_CHECK( optixModuleDestroy( state.optixir_module ) );
    OTK_ERROR_CHECK( optixDeviceContextDestroy( state.context ) );

    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer ) ) );
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
}


 void OTKApp::initOptixPipelines( const char* moduleCode, const size_t moduleCodeSize )
{
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        createContext( state );
        buildAccel( state );
        createModule( state, moduleCode, moduleCodeSize );
        createProgramGroups( state );
        createPipeline( state );
        createSBT( state );
    }
}


//------------------------------------------------------------------------------
// Demand Loading
//------------------------------------------------------------------------------

demandLoading::TextureDescriptor OTKApp::makeTextureDescriptor( CUaddress_mode addressMode, FilterMode filterMode )
{
    demandLoading::TextureDescriptor texDesc{};
    texDesc.addressMode[0]   = addressMode;
    texDesc.addressMode[1]   = addressMode;
    texDesc.filterMode       = filterMode;
    texDesc.mipmapFilterMode = toCudaFilterMode( filterMode );
    texDesc.maxAnisotropy    = 16;

    return texDesc;
}


std::shared_ptr<imageSource::ImageSource> OTKApp::createExrImage( const std::string& filePath )
{
    try
    {
        return filePath.empty() ? std::shared_ptr<imageSource::ImageSource>() : imageSource::createImageSource( filePath );
    }
    catch( ... )
    {
    }
    return nullptr;
}


void OTKApp::initDemandLoading( demandLoading::Options options )
{
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        state.demandLoader.reset( createDemandLoader( options ), demandLoading::destroyDemandLoader );
    }
}


void OTKApp::printDemandLoadingStats()
{
    std::vector<demandLoading::Statistics> stats;
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
        stats.push_back( state.demandLoader->getStatistics() );
   
    std::cout << std::fixed << std::setprecision( 1 );
    std::cout << "\n============================================\n";
    std::cout << "Demand Loading Stats\n";
    std::cout << "============================================\n";
    std::cout << "Launch cycles:            " << m_launchCycles << "\n";
    std::cout << "Num textures:             " << stats[0].numTextures << "\n";
    std::cout << "Virtual texture Size:     " << stats[0].virtualTextureBytes / ( 1024.0 * 1024.0 ) << " MiB\n";
    std::cout << "Tiles read from disk:     " << stats[0].numTilesRead << "\n";
    
    std::cout << "Max device memory used:   ";
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        const double deviceMemory = stats[state.device_idx].deviceMemoryUsed / ( 1024.0 * 1024.0 );
        std::cout << "[GPU-" << state.device_idx << ": " << deviceMemory << " MiB]  ";
    }
    std::cout << "\n";

    std::cout << "Texture data transferred: ";
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        const size_t tilesTransferred = stats[state.device_idx].bytesTransferredToDevice / 65536;
        const double transferData = stats[state.device_idx].bytesTransferredToDevice / ( 1024.0 * 1024.0 );
        std::cout << "[GPU-" << state.device_idx << ": " << tilesTransferred << " tiles (" << transferData << " MiB)]  ";
    }
    std::cout << "\n";

    std::cout << "Evicted tiles:            ";
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        std::cout << "[GPU-" << state.device_idx << ": " << stats[state.device_idx].numEvictions << "]  ";
    }

    std::cout << "\n" << std::endl;
}


//------------------------------------------------------------------------------
// OptiX launches
//------------------------------------------------------------------------------

void OTKApp::initView()
{
    // Set the view so that the square (0,0,0) to (1,1,0) exactly covers the viewport.
    setView( float3{0.5f, 0.5f, 1.0f}, float3{0.5f, 0.5f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, 53.130102354f );
}

void OTKApp::setView( float3 eye, float3 lookAt, float3 up, float fovY )
{
    float aspectRatio = static_cast<float>( m_windowWidth ) / static_cast<float>( m_windowHeight );
    m_camera = otk::Camera( eye, lookAt, up, fovY, aspectRatio );
}

void OTKApp::panCamera( float3 pan )
{
    if( dot(pan, pan) == 0.0f )
        return;
    m_camera.setEye( m_camera.eye() + pan );
    m_camera.setLookAt( m_camera.lookAt() + pan );
    m_subframeId = 0;
}

void OTKApp::zoomCamera( float zoom )
{
    if( zoom == 1.0f )
        return;
    float tanVal = zoom * tanf( m_camera.fovY() * (M_PI / 360.0f) );
    m_camera.setFovY( atanf( tanVal ) * 360.0f / M_PI );
    m_subframeId = 0;
}

void OTKApp::rotateCamera( float rot )
{
    if( rot == 0.0f )
        return;
    float3 U, V, W;
    m_camera.UVWFrame( U, V, W );
    W = float3{ W.x * cosf(rot) - W.y * sinf(rot), W.x * sinf(rot) + W.y * cosf(rot), W.z };
    m_camera.setLookAt( m_camera.eye() + W );
    m_subframeId = 0;
}

void OTKApp::initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices )
{
    state.params.image_dim.x        = m_outputBuffer->width();
    state.params.image_dim.y        = m_outputBuffer->height();
    state.params.traversable_handle = state.gas_handle;
    state.params.device_idx         = state.device_idx;
    state.params.num_devices        = numDevices;
    state.params.display_texture_id = (m_textureIds.size() > 0) ? m_textureIds[0] : 0;
    state.params.interactive_mode   = isInteractive();
    state.params.render_mode        = m_render_mode;
    state.params.subframe           = m_subframeId;
    state.params.projection         = m_projection;
    state.params.lens_width         = m_lens_width;
    state.params.camera.eye         = m_camera.eye();
    state.params.background_color   = m_backgroundColor;
    m_camera.UVWFrame( state.params.camera.U, state.params.camera.V, state.params.camera.W );

    // Make sure a device-side copy of the params has been allocated
    if( state.d_params == nullptr )
        OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( OTKAppLaunchParams ) ) );
}


unsigned int OTKApp::performLaunches( )
{
    const unsigned int numDevices     = static_cast<unsigned int>( m_perDeviceOptixStates.size() );
    unsigned int numRequestsProcessed = 0;

    // Resize the output buffer if needed. 
    m_outputBuffer->resize( m_windowWidth, m_windowHeight );

    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        // Wait on the ticket from the previous launch
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        state.ticket.wait();
        numRequestsProcessed += static_cast<unsigned int>( state.ticket.numTasksTotal() );

        // Call launchPrepare to synchronize new texture samplers and texture info to device memory,
        // and allocate device memory for the demand texture context.
        if( state.demandLoader )
            state.demandLoader->launchPrepare( state.stream, state.params.demand_texture_context );

        // Finish initialization of the launch params.
        state.params.result_buffer = m_outputBuffer->map();
        initLaunchParams( state, numDevices );

        // Copy launch params to device.  Note: cudaMemcpy measured faster than cudaMemcpyAsync 
        // for this application, so we use it here.  Other applications may differ.
        OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.d_params ), &state.params, sizeof( state.params ), cudaMemcpyHostToDevice ) );

        // Peform the OptiX launch, with each device doing a part of the work
        unsigned int launchHeight = ( state.params.image_dim.y + numDevices - 1 ) / numDevices;
        OTK_ERROR_CHECK( optixLaunch( state.pipeline,  // OptiX pipeline
                                  state.stream,    // Stream for launch and demand loading
                                  reinterpret_cast<CUdeviceptr>( state.d_params ),  // Launch params
                                  sizeof( state.params ),                           // Param size in bytes
                                  &state.sbt,                                       // Shader binding table
                                  state.params.image_dim.x,                         // Launch width
                                  launchHeight,                                     // Launch height
                                  1                                                 // launch depth
                                  ) );

        // Begin to process demand load requests. This asynchronously pulls a batch of requests 
        // from the device and places them in a queue for processing.  The progress of the batch
        // can be polled using the returned ticket.
        if( state.demandLoader )
            state.ticket = state.demandLoader->processRequests( state.stream, state.params.demand_texture_context );

        // Unmap the output buffer. The device pointer from map should not be used after this call.
        m_outputBuffer->unmap();
    }

    return numRequestsProcessed;
}


void OTKApp::startLaunchLoop()
{
    int numFilled = 0;

    if( isInteractive() )
    {
        while( !glfwWindowShouldClose( getWindow() ) )
        {
            glfwPollEvents();
            pollKeys();
            if( numFilled > m_reset_subframe_threshold || m_launchCycles <= m_minLaunches )
                m_subframeId = 0;
            if( m_subframeId >= m_maxSubframes )
                continue;
            numFilled = performLaunches();

            m_numFilledRequests += numFilled;
            ++m_launchCycles;
            ++m_subframeId;
            displayFrame();
            drawGui();
            glfwSwapBuffers( getWindow() );
        }
    }
    else 
    {
        // Launch repeatedly until there are no more requests to fill.
        numFilled = performLaunches();
        do
        {
            numFilled = performLaunches();
            m_numFilledRequests += numFilled;
            ++m_launchCycles;
            ++m_subframeId;
            if ( numFilled > m_reset_subframe_threshold )
                m_subframeId = 0;
        } while( numFilled > 0 || m_launchCycles < m_minLaunches );

        saveImage();
    }

    cleanup();
}

void OTKApp::cleanup()
{
    for( OTKAppPerDeviceOptixState state : m_perDeviceOptixStates )
    {
        cleanupState( state );
    }
    if( isInteractive() )
    {
        // The output buffer is tied to the OpenGL context in interactive mode.
        OTK_ERROR_CHECK( cudaSetDevice( m_perDeviceOptixStates[0].device_idx ) );
        m_outputBuffer.reset();
        otk::cleanupUI( m_window );
    }
}


//------------------------------------------------------------------------------
// Display
//------------------------------------------------------------------------------

void OTKApp::drawGui()
{
    otk::beginFrameImGui();
    otk::displayFPS( m_launchCycles );
    otk::endFrameImGui();
}

void OTKApp::displayFrame()
{
    m_glDisplay->display( m_outputBuffer->width(), m_outputBuffer->height(), m_windowWidth, m_windowHeight,
                          m_outputBuffer->getPBO() );
}

void OTKApp::saveImage()
{
    otk::ImageBuffer buffer;
    buffer.data         = m_outputBuffer->getHostPointer();
    buffer.width        = m_outputBuffer->width();
    buffer.height       = m_outputBuffer->height();
    buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;

    std::string fileName = !m_outputFileName.empty() ? m_outputFileName : "out.ppm";
    otk::saveImage( fileName.c_str(), buffer, false );
}

 void OTKApp::resetAccumulator()
 {
    size_t accumulatorSize = sizeof(float4) * m_windowWidth * m_windowHeight;
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        OTK_ERROR_CHECK( cudaFree( state.params.accum_buffer ) );
        OTK_ERROR_CHECK( cudaMalloc( &state.params.accum_buffer, accumulatorSize ) );
    }
 }

//------------------------------------------------------------------------------
// User Interaction
//------------------------------------------------------------------------------

void OTKApp::mouseButtonCallback( GLFWwindow* window, int button, int action, int /*mods*/ )
{
    glfwGetCursorPos( window, &m_mousePrevX, &m_mousePrevY );
    m_mouseButton = ( action == GLFW_PRESS ) ? button : NO_BUTTON;
}

void OTKApp::cursorPosCallback( GLFWwindow* /*window*/, double xpos, double ypos )
{
    float dx = static_cast<float>( xpos - m_mousePrevX );
    float dy = static_cast<float>( ypos - m_mousePrevY );

    // Image View
    if( m_uiMode == UI_IMAGEVIEW )
    {
        const float panScale = 2.0f * tanf( m_camera.fovY() * M_PIf / 360.0f ) / m_windowHeight;
        const float zoomScale = 1.003f;

        if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )
            panCamera( float3{ -dx * panScale, dy * panScale, 0.0f } );
        else if( m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )
            zoomCamera( powf( zoomScale, ( dy - dx ) ) );
    }

    // First Person
    if( m_uiMode == UI_FIRSTPERSON )
    {
        const float panScale = 0.03f;
        const float rotScale = 0.002f;
        float3 U, V, W;
        m_camera.UVWFrame( U, V, W );

        if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )
            panCamera( ( panScale * dx * normalize(U) ) + ( -panScale * dy * float3{0.0f, 1.0f, 0.0f} ) );
        else if( m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )
            rotateCamera( -rotScale * dx );
    }

    if( m_mouseButton != NO_BUTTON )
        m_subframeId = 0;
    m_mousePrevX = xpos;
    m_mousePrevY = ypos;
}

void OTKApp::windowSizeCallback( GLFWwindow* /*window*/, int32_t width, int32_t height )
{
    m_windowWidth  = width;
    m_windowHeight = height;
    m_camera.setAspectRatio( static_cast<float>( m_windowWidth ) / static_cast<float>( m_windowHeight ) );
    if( m_perDeviceOptixStates[0].params.accum_buffer != nullptr )
        resetAccumulator();
    m_subframeId = 0;
}

void OTKApp::pollKeys()
{
    GLFWwindow* wnd = m_window;

    // Image View
    if( m_uiMode == UI_IMAGEVIEW )
    {
        const float panScale  = 0.003f * ( m_camera.fovY() * M_PIf / 360.0f );
        const float zoomScale = 1.003f;

        float panx = panScale * ( glfwGetKey( wnd, GLFW_KEY_D ) - glfwGetKey( wnd, GLFW_KEY_A ) );
        float pany = panScale * ( glfwGetKey( wnd, GLFW_KEY_W ) - glfwGetKey( wnd, GLFW_KEY_S ) );
        panCamera( float3{panx, pany, 0.0f} );

        float zoom = glfwGetKey( wnd, GLFW_KEY_Q ) ? zoomScale : 1.0f;
        zoom /= glfwGetKey( wnd, GLFW_KEY_E ) ? zoomScale : 1.0f;
        zoomCamera( zoom );
    }

    // First Person
    if( m_uiMode == UI_FIRSTPERSON )
    {
        float3 U, V, W;
        m_camera.UVWFrame( U, V, W );

        // Assuming Y Up
        float uPan = 0.04f * ( glfwGetKey( wnd, GLFW_KEY_D ) - glfwGetKey( wnd, GLFW_KEY_A ) );
        float wPan = 0.04f * ( glfwGetKey( wnd, GLFW_KEY_W ) - glfwGetKey( wnd, GLFW_KEY_S ) );
        float upPan = 0.01f * ( glfwGetKey( wnd, GLFW_KEY_E ) - glfwGetKey( wnd, GLFW_KEY_Q ) );
        panCamera( normalize( U ) * uPan + normalize( float3{W.x, W.y, 0.0f} ) * wPan + normalize( m_camera.up() ) * upPan );

        float rot = 0.003f * ( glfwGetKey( wnd, GLFW_KEY_J ) - glfwGetKey( wnd, GLFW_KEY_L ) );
        rotateCamera( rot );
    }
}

void OTKApp::keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action != GLFW_PRESS )
        return;

    if( key == GLFW_KEY_F1 )
        saveImage();
    else if( key == GLFW_KEY_ESCAPE )
        glfwSetWindowShouldClose( window, true );
    else if( key == GLFW_KEY_C )
        initView();
    else if( key >= GLFW_KEY_0 && key <= GLFW_KEY_9 )
        m_render_mode = key - GLFW_KEY_0;
}

//------------------------------------------------------------------------------
// GLFW callbacks
//------------------------------------------------------------------------------

void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    OTKApp* app = reinterpret_cast<OTKApp*>( glfwGetWindowUserPointer( window ) );
    app->mouseButtonCallback( window, button, action, mods );
}
void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    OTKApp* app = reinterpret_cast<OTKApp*>( glfwGetWindowUserPointer( window ) );
    app->cursorPosCallback( window, xpos, ypos );
}
void windowSizeCallback( GLFWwindow* window, int32_t width, int32_t height )
{
    OTKApp* app = reinterpret_cast<OTKApp*>( glfwGetWindowUserPointer( window ) );
    app->windowSizeCallback( window, width, height );
}
void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    OTKApp* app = reinterpret_cast<OTKApp*>( glfwGetWindowUserPointer( window ) );
    app->keyCallback( window, key, scancode, action, mods );
}

void setGLFWCallbacks( OTKApp* app )
{
    glfwSetWindowUserPointer( app->getWindow(), app );
    glfwSetMouseButtonCallback( app->getWindow(), mouseButtonCallback );
    glfwSetCursorPosCallback( app->getWindow(), cursorPosCallback );
    glfwSetKeyCallback( app->getWindow(), keyCallback );
    glfwSetWindowSizeCallback( app->getWindow(), windowSizeCallback );
}

} // namespace OTKApp
