// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "SourceDir.h"  // generated from SourceDir.h.in

#include "PerDeviceOptixState.h"
#include "CuOmmBakingApp.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/glfw3.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>
#include <OptiXToolkit/Util/Logger.h>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif
    
namespace ommBakingApp
{

//------------------------------------------------------------------------------
// Texture Loading
//------------------------------------------------------------------------------

CudaTexture::CudaTexture( std::string textureName )
{
    if( textureName.empty() && textureName[0] == '\0' )
        throw std::runtime_error( "Invalid texture name" );

    std::string  textureFilename( getSourceDir() + "/../Textures/" + textureName );
    otk::EXRInputFile imageFile;
    imageFile.open( textureFilename );
    create( &imageFile );
}

void CudaTexture::create( otk::EXRInputFile* imageFile )
{
    destroy();

    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    size_t bytesPerPixel = ( desc.x + desc.y + desc.z + desc.w ) / 8;
    std::vector<char> data( bytesPerPixel * imageFile->getWidth() * imageFile->getHeight() );
    imageFile->read( data.data(), data.size() );

    CuPitchedBuffer<char> d_data;
    OTK_ERROR_CHECK( d_data.allocAndUpload( bytesPerPixel * imageFile->getWidth(), imageFile->getHeight(), data.data() ) );

    struct cudaResourceDesc resDesc;
    memset( &resDesc, 0, sizeof( resDesc ) );
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc = desc;
    resDesc.res.pitch2D.devPtr = ( void* )d_data.get();
    resDesc.res.pitch2D.width = imageFile->getWidth();
    resDesc.res.pitch2D.height = imageFile->getHeight();
    resDesc.res.pitch2D.pitchInBytes = d_data.pitch();

    struct cudaTextureDesc texDesc;
    memset( &texDesc, 0, sizeof( texDesc ) );
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = 0;
    texDesc.filterMode = cudaFilterModeLinear;

    cudaTextureObject_t texObj;
    OTK_ERROR_CHECK( cudaCreateTextureObject( &texObj, &resDesc, &texDesc, NULL ) );

    std::swap( m_texObj, texObj );
    std::swap( m_data, d_data );
}

//------------------------------------------------------------------------------
// Boilerplate App 
//------------------------------------------------------------------------------

OmmBakingApp::OmmBakingApp( const char* appName, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
    : m_windowWidth( width )
    , m_windowHeight( height )
    , m_outputFileName( outFileName )
    , m_glInterop( glInterop )
{
    // Create display window for interactive mode
    if( isInteractive() )
    {
        OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
        OTK_ERROR_CHECK( cudaFree( 0 ) );
        m_window = otk::initGLFW( appName, width, height );
        otk::initGL();
        m_glDisplay.reset( new otk::GLDisplay( otk::BufferImageFormat::UNSIGNED_BYTE4 ) );
        setGLFWCallbacks( this );
    }

    initView();
}


OmmBakingApp::~OmmBakingApp()
{
    for( PerDeviceOptixState state : m_perDeviceOptixStates )
        cleanupState( state );
}


//------------------------------------------------------------------------------
// OptiX setup
//------------------------------------------------------------------------------

void OmmBakingApp::createContext( PerDeviceOptixState& state )
{
    // Initialize CUDA on this device
    OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
    OTK_ERROR_CHECK( cudaFree( 0 ) );

    CUcontext                 cuCtx   = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    otk::util::setLogger( options );
    OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &options, &state.context ) );

    OTK_ERROR_CHECK( cudaStreamCreate( &state.stream ) );
}



void OmmBakingApp::createModule( PerDeviceOptixState& state, const char* moduleCode, size_t codeSize )
{
    OptixModuleCompileOptions module_compile_options{};
    otk::configModuleCompileOptions( module_compile_options );

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 3;
    state.pipeline_compile_options.numAttributeValues    = 6;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.allowOpacityMicromaps = true;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OTK_ERROR_CHECK_LOG( optixModuleCreate( state.context, &module_compile_options, &state.pipeline_compile_options,
                                        moduleCode, codeSize, log, &sizeof_log, &state.optixir_module ) );
}


void OmmBakingApp::createProgramGroups( PerDeviceOptixState& state )
{
    // Make program groups for raygen, miss, and hitgroup
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.optixir_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &raygen_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.raygen_prog_group ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.optixir_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.miss_prog_group ) );

    OptixProgramGroupDesc hitgroup_prog_group_desc        = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = state.optixir_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = state.optixir_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    hitgroup_prog_group_desc.hitgroup.moduleIS            = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( state.context, &hitgroup_prog_group_desc,
                                              1,  // num program groups
                                              &program_group_options, log, &sizeof_log, &state.hitgroup_prog_group ) );
}


void OmmBakingApp::createPipeline( PerDeviceOptixState& state )
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
                                             0,  // maxDCDEpth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OTK_ERROR_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void OmmBakingApp::cleanupState( PerDeviceOptixState& state )
{
    OTK_ERROR_CHECK( optixPipelineDestroy( state.pipeline ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
    OTK_ERROR_CHECK( optixProgramGroupDestroy( state.hitgroup_prog_group ) );
    OTK_ERROR_CHECK( optixModuleDestroy( state.optixir_module ) );
    OTK_ERROR_CHECK( optixDeviceContextDestroy( state.context ) );
}


void OmmBakingApp::initOptixPipelines( const char* moduleCode, const size_t moduleCodeSize, int numDevices )
{
    bool glInterop = m_glInterop && isInteractive() && ( m_perDeviceOptixStates.size() == 1 );
    otk::CUDAOutputBufferType outputBufferType =
        glInterop ? otk::CUDAOutputBufferType::GL_INTEROP : otk::CUDAOutputBufferType::ZERO_COPY;
    m_outputBuffer.reset( new otk::CUDAOutputBuffer<uchar4>( outputBufferType, m_windowWidth, m_windowHeight ) );

    m_perDeviceOptixStates.resize( numDevices );

    for( unsigned int i = 0; i < m_perDeviceOptixStates.size(); ++i )
    {
        PerDeviceOptixState& state = m_perDeviceOptixStates[i];
        state.device_idx           = i;
        createContext( state );
        buildAccel( state );
        createModule( state, moduleCode, moduleCodeSize );
        createProgramGroups( state );
        createPipeline( state );
        createSBT( state );
    }
}


//------------------------------------------------------------------------------
// OptiX launches
//------------------------------------------------------------------------------

void OmmBakingApp::initView()
{
    float aspectRatio = static_cast<float>( m_windowWidth ) / static_cast<float>( m_windowHeight );
    m_eye             = INITIAL_LOOK_FROM;
    m_viewDims        = float2{INITIAL_VIEW_DIM * aspectRatio, INITIAL_VIEW_DIM};
}

void OmmBakingApp::performLaunches( )
{
    // Resize the output buffer if needed. 
    m_outputBuffer->resize( m_windowWidth, m_windowHeight );
   
    m_viewDims.x = m_viewDims.y * getWidth() / getHeight();

    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );

        // Map the output buffer to get a valid device pointer for the launch output.
        uchar4* result_buffer = m_outputBuffer->map();

        performLaunch( state, result_buffer );

        // Unmap the output buffer. The device pointer from map should not be used after this call.
        m_outputBuffer->unmap();
    }
    
    return;
}


void OmmBakingApp::startLaunchLoop()
{
    if( isInteractive() )
    {
        while( !glfwWindowShouldClose( getWindow() ) )
        {
            glfwPollEvents();
            pollKeys();
            performLaunches();
            displayFrame();
            glfwSwapBuffers( getWindow() );
        }
    }
    else 
    {
        performLaunches();
        saveImage();
    }
}


//------------------------------------------------------------------------------
// Display
//------------------------------------------------------------------------------

void OmmBakingApp::displayFrame()
{
    m_glDisplay->display( m_outputBuffer->width(), m_outputBuffer->height(), m_windowWidth, m_windowHeight,
                          m_outputBuffer->getPBO() );
}

void OmmBakingApp::saveImage()
{
    otk::ImageBuffer buffer;
    buffer.data         = m_outputBuffer->getHostPointer();
    buffer.width        = m_outputBuffer->width();
    buffer.height       = m_outputBuffer->height();
    buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;
    otk::saveImage( m_outputFileName.c_str(), buffer, false );
}


//------------------------------------------------------------------------------
// User Interaction via GLFW
//------------------------------------------------------------------------------

void OmmBakingApp::mouseButtonCallback( GLFWwindow* window, int button, int action, int /*mods*/ )
{
    glfwGetCursorPos( window, &m_mousePrevX, &m_mousePrevY );
    m_mouseButton = ( action == GLFW_PRESS ) ? button : NO_BUTTON;
}

void OmmBakingApp::cursorPosCallback( GLFWwindow* /*window*/, double xpos, double ypos )
{
    double dx = xpos - m_mousePrevX;
    double dy = ypos - m_mousePrevY;

    if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )  // pan camera
    {
        m_eye.x -= static_cast<float>( dx * m_viewDims.x / m_windowWidth );
        m_eye.y += static_cast<float>( dy * m_viewDims.y / m_windowHeight );
    }
    else if( m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )  // zoom camera
    {
        float zoom = powf( 1.003f, static_cast<float>( dy - dx ) );
        m_viewDims.y *= zoom;  // x is reset based on y later
    }

    m_mousePrevX = xpos;
    m_mousePrevY = ypos;
}

void OmmBakingApp::windowSizeCallback( GLFWwindow* /*window*/, int32_t width, int32_t height )
{
    m_windowWidth  = width;
    m_windowHeight = height;
}

void OmmBakingApp::pollKeys()
{
    const float pan  = 0.003f * m_viewDims.y;
    const float zoom = 1.003f;

    if( glfwGetKey( getWindow(), GLFW_KEY_A ) )
        m_eye.x -= pan;
    if( glfwGetKey( getWindow(), GLFW_KEY_D ) )
        m_eye.x += pan;
    if( glfwGetKey( getWindow(), GLFW_KEY_S ) )
        m_eye.y -= pan;
    if( glfwGetKey( getWindow(), GLFW_KEY_W ) )
        m_eye.y += pan;
    if( glfwGetKey( getWindow(), GLFW_KEY_Q ) )
        m_viewDims.y *= zoom;  // x is reset based on y later
    if( glfwGetKey( getWindow(), GLFW_KEY_E ) )
        m_viewDims.y /= zoom;  // x is reset based on y later
}

void OmmBakingApp::keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
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
    OmmBakingApp* app = reinterpret_cast<OmmBakingApp*>( glfwGetWindowUserPointer( window ) );
    app->mouseButtonCallback( window, button, action, mods );
}
void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    OmmBakingApp* app = reinterpret_cast<OmmBakingApp*>( glfwGetWindowUserPointer( window ) );
    app->cursorPosCallback( window, xpos, ypos );
}
void windowSizeCallback( GLFWwindow* window, int32_t width, int32_t height )
{
    OmmBakingApp* app = reinterpret_cast<OmmBakingApp*>( glfwGetWindowUserPointer( window ) );
    app->windowSizeCallback( window, width, height );
}
void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    OmmBakingApp* app = reinterpret_cast<OmmBakingApp*>( glfwGetWindowUserPointer( window ) );
    app->keyCallback( window, key, scancode, action, mods );
}

void setGLFWCallbacks( OmmBakingApp* app )
{
    glfwSetWindowUserPointer( app->getWindow(), app );
    glfwSetMouseButtonCallback( app->getWindow(), mouseButtonCallback );
    glfwSetCursorPosCallback( app->getWindow(), cursorPosCallback );
    glfwSetKeyCallback( app->getWindow(), keyCallback );
    glfwSetWindowSizeCallback( app->getWindow(), windowSizeCallback );
}

} // namespace ommBakingApp
