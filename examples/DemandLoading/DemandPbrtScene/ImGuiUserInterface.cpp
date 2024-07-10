//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/Gui/glad.h>  // Glad insists on being included first.

#include "ImGuiUserInterface.h"
#include "FrameRate.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <OptiXToolkit/Gui/BufferMapper.h>
#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/GLDisplay.h>

#include <GLFW/glfw3.h>

#include <iomanip>
#include <utility>

#include "Renderer.h"
#include "Scene.h"

namespace demandPbrtScene {

using OutputBuffer = otk::CUDAOutputBuffer<uchar4>;

ImGuiUserInterface::ImGuiUserInterface( Options& options, RendererPtr renderer, ScenePtr scene )
    : m_options( options )
    , m_renderer( std::move( renderer ) )
    , m_scene( std::move( scene ) )
{
}

void ImGuiUserInterface::initialize( const LookAtParams& lookAt, const PerspectiveCamera& camera )
{
    m_debug = m_renderer->getDebugLocation();
    createWindow();
    createUI();
    initCamera( lookAt, camera);
    m_frameRate.reset();
}

void ImGuiUserInterface::initCamera( const LookAtParams& lookAt, const PerspectiveCamera& camera )
{
    m_trackballCamera.setCameraEye( lookAt.eye );
    m_trackballCamera.setCameraLookAt( lookAt.lookAt );
    m_trackballCamera.setCameraUp( lookAt.up );
    m_trackballCamera.setCameraFovY( camera.fovY );
    m_trackballCamera.setCameraAspectRatio( camera.aspectRatio );
}

void ImGuiUserInterface::createWindow()
{
    m_window = otk::initGLFW( "Demand PBRT Scene", m_options.width, m_options.height );
    otk::initGL();
    m_trackballCamera.associateWindow( m_window );
    glfwSetWindowUserPointer( m_window, this );
    glfwSetMouseButtonCallback( m_window, mouseButtonCallback );
    glfwSetCursorPosCallback( m_window, cursorPosCallback );
    glfwSetWindowSizeCallback( m_window, windowSizeCallback );
    glfwSetScrollCallback( m_window, scrollCallback );
    glfwSetKeyCallback( m_window, keyCallback );
    m_display = std::make_shared<otk::GLDisplay>();
}

void ImGuiUserInterface::createUI()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL( m_window, true );
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.Fonts->AddFontDefault();

    ImGui::GetStyle().WindowBorderSize = 0.0f;
}

void ImGuiUserInterface::setNextWindowBackground() const
{
    ImGui::SetNextWindowBgAlpha( 0.5f );
}

void ImGuiUserInterface::renderDebug()
{
    bool debugEnabled = m_debug.enabled;
    if( ImGui::Checkbox( "Debug mode", &debugEnabled ) )
    {
        m_debug.enabled        = debugEnabled;
        m_debugLocationChanged = true;
    }
    renderToggleOption( m_options.oneShotDebug, "One shot debug" );
}

void ImGuiUserInterface::renderToggleOption( bool& option, const char* label )
{
    bool currentOption = option;
    if( ImGui::Checkbox( label, &currentOption ) )
    {
        option           = currentOption;
        m_optionsChanged = true;
    }
}

void ImGuiUserInterface::renderFPS() const
{
    std::stringstream text;
    text << "fps: " << std::fixed << std::setw(4) << std::setprecision(0) << m_frameRate.getFPS();
    setNextWindowBackground();
    const float height = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().WindowPadding.y * 2;
    ImGui::SetNextWindowPos( ImVec2( 0.0f, ImGui::GetIO().DisplaySize.y - height ) );
    ImGui::Begin( "Text", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove
                  | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing );
    ImGui::Text( "%s", text.str().c_str() );
    ImGui::End();
}

void ImGuiUserInterface::renderGeometryCacheStatistics() const
{
    if( ImGui::TreeNode( "Geometry Cache" ) )
    {
        const auto& stats{ m_stats.geometryCache };
        ImGui::Text( "Traversables: %u", stats.numTraversables );
        ImGui::Text( "Triangles: %u", stats.numTriangles );
        ImGui::Text( "Spheres: %u", stats.numSpheres );
        ImGui::Text( "Normals: %u", stats.numNormals );
        ImGui::Text( "UVs: %u", stats.numUVs );
        ImGui::TreePop();
        ImGui::Spacing();
    }
}

void ImGuiUserInterface::renderImageCacheStatistics( const char* label, const imageSource::CacheStatistics& stats ) const
{
    if( ImGui::TreeNode( label ) )
    {
        ImGui::Text( "Num images: %u", stats.numImageSources );
        ImGui::Text( "Num tiles read: %llu", stats.totalTilesRead );
        ImGui::Text( "Num bytes read: %llu", stats.totalBytesRead );
        ImGui::Text( "Read time: %g secs", stats.totalReadTime );
        ImGui::TreePop();
        ImGui::Spacing();
    }
}

void ImGuiUserInterface::renderImageSourceFactoryStatistics() const
{
    if( ImGui::TreeNode( "Image Sources" ) )
    {
        renderImageCacheStatistics( "Files", m_stats.imageSourceFactory.fileSources );
        renderImageCacheStatistics( "Alpha images", m_stats.imageSourceFactory.alphaSources );
        renderImageCacheStatistics( "Diffuse images", m_stats.imageSourceFactory.diffuseSources );
        renderImageCacheStatistics( "Skybox images", m_stats.imageSourceFactory.skyboxSources );
        ImGui::TreePop();
        ImGui::Spacing();
    }
}

void ImGuiUserInterface::renderProxyFactoryStatistics() const
{
    if( ImGui::TreeNode( "Proxy Geometry" ) )
    {
        ImGui::Text( "Proxies created: %u", m_stats.proxyFactory.numGeometryProxiesCreated );
        ImGui::TreePop();
        ImGui::Spacing();
    }
}

void ImGuiUserInterface::renderSceneStatistics() const
{
    if( ImGui::TreeNode( "Scene" ) )
    {
        const auto& stats{ m_stats.scene };
        ImGui::Text( "Proxy geometries resolved: %u", stats.numProxyGeometriesResolved );
        ImGui::Text( "Geometries realized; %u", stats.numGeometriesRealized );
        ImGui::Text( "Proxy materials created: %u", stats.numProxyMaterialsCreated );
        ImGui::Text( "Partial materials realized: %u", stats.numPartialMaterialsRealized );
        ImGui::Text( "Materials realized: %u", stats.numMaterialsRealized );
        ImGui::TreePop();
        ImGui::Spacing();
    }
}

void ImGuiUserInterface::renderStatistics() const
{
    ImGui::SetNextWindowPos( ImVec2( 0.0f, m_optionsHeight ) );
    ImGui::SetNextWindowCollapsed( true, ImGuiCond_Once );
    if( ImGui::Begin( "Statistics", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings
        | ImGuiWindowFlags_NoFocusOnAppearing ) )
    {
        ImGui::Text( "Frames rendered: %u" , m_stats.numFramesRendered );
        renderGeometryCacheStatistics();
        renderImageSourceFactoryStatistics();
        renderProxyFactoryStatistics();
        renderSceneStatistics();
    }
    ImGui::End();
    renderFPS();
}

void ImGuiUserInterface::renderOptions()
{
    ImGui::SetNextWindowCollapsed( true, ImGuiCond_Once );
    if( ImGui::Begin( "Options", nullptr,
                      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings
                          | ImGuiWindowFlags_NoFocusOnAppearing ) )
    {
        if( ImGui::TreeNode( "Demand Loading Options" ) )
        {
            renderToggleOption( m_options.oneShotGeometry, "One shot geometry" );
            renderToggleOption( m_options.oneShotMaterial, "One shot material" );
            renderToggleOption( m_options.sortProxies, "Sort proxies" );
            ImGui::TreePop();
            ImGui::Spacing();
        }
        if( ImGui::TreeNode( "Rendering Options" ) )
        {
            renderToggleOption( m_options.faceForward, "Face forward" );
#ifndef NDEBUG
            renderToggleOption( m_options.usePinholeCamera, "Use pinhole camera, not pbrt camera" );
#endif
            ImGui::TreePop();
            ImGui::Spacing();
        }
        if( ImGui::TreeNode( "Render Mode" ) )
        {
            int currentRenderMode = m_options.renderMode;
            ImGui::RadioButton( "Phong Shading", &m_options.renderMode, PHONG_SHADING );
            ImGui::RadioButton( "Short AO", &m_options.renderMode, SHORT_AO );
            ImGui::RadioButton( "Long AO", &m_options.renderMode, LONG_AO );
            ImGui::RadioButton( "Path Tracing", &m_options.renderMode, PATH_TRACING );
            ImGui::TreePop();
            ImGui::Spacing();
            if( currentRenderMode != m_options.renderMode )
                m_renderer->setClearAccumulator();
        }
        if( ImGui::TreeNode( "Debug Options" ) )
        {
            renderDebug();
            renderToggleOption( m_options.verboseProxyGeometryResolution, "Verbose proxy geometry resolution" );
            renderToggleOption( m_options.verboseProxyMaterialResolution, "Verbose proxy material resolution" );
            renderToggleOption( m_options.verboseSceneDecomposition, "Verbose scene decomposition" );
            renderToggleOption( m_options.verboseTextureCreation, "Verbose texture creation" );
            ImGui::TreePop();
            ImGui::Spacing();
        }
    }
    m_optionsHeight = ImGui::GetWindowHeight();
    ImGui::End();
}

void ImGuiUserInterface::renderUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    setNextWindowBackground();
    ImGui::SetNextWindowPos( ImVec2( 0, 0 ) );
    renderOptions();
    renderStatistics();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}

void ImGuiUserInterface::destroyUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow( m_window );
    glfwTerminate();
}

bool ImGuiUserInterface::handleCameraUpdate()
{
    if( !m_trackballCamera.handleCameraUpdate() )
    {
        return false;
    }

    LookAtParams lookAt;
    lookAt.lookAt = m_trackballCamera.getCameraLookAt();
    lookAt.eye    = m_trackballCamera.getCameraEye();
    lookAt.up     = m_trackballCamera.getCameraUp();
    m_renderer->setLookAt( lookAt );
    return true;
}

bool ImGuiUserInterface::handleResize( OutputBuffer& output )
{
    if( !m_trackballCamera.handleResize() )
        return false;

    int width  = m_trackballCamera.getWidth();
    int height = m_trackballCamera.getHeight();
    output.resize( width, height );
    m_options.width  = width;
    m_options.height = height;
    return true;
}

bool ImGuiUserInterface::handleParamsUpdate()
{
    const bool changed     = m_debugLocationChanged;
    m_debugLocationChanged = false;
    if( changed )
    {
        m_renderer->setDebugLocation( m_debug );
    }
    return changed;
}

bool ImGuiUserInterface::handleOptionsUpdate()
{
    const bool optionsChanged = m_optionsChanged;
    m_optionsChanged          = false;
    return optionsChanged;
}

bool ImGuiUserInterface::updateState( OutputBuffer& output )
{
    const bool cameraUpdated  = handleCameraUpdate();
    const bool sizeUpdated    = handleResize( output );
    const bool paramsChanged  = handleParamsUpdate();
    const bool optionsChanged = handleOptionsUpdate();
    return cameraUpdated || sizeUpdated || paramsChanged || optionsChanged;
}

void ImGuiUserInterface::displayOutput( OutputBuffer& output )
{
    int frameBuffResX;
    int frameBuffResY;
    glfwGetFramebufferSize( m_window, &frameBuffResX, &frameBuffResY );
    m_display->display( output.width(), output.height(), frameBuffResX, frameBuffResY, output.getPBO() );
}

void ImGuiUserInterface::printLookAtKeywordValues()
{
    const float3 eye = m_trackballCamera.getCameraEye();
    const float3 lookAt = m_trackballCamera.getCameraLookAt();
    const float3 up = m_trackballCamera.getCameraUp();
    std::cout << "LookAt\n"                                                              //
              << "    " << eye.x << "    " << eye.y << "    " << eye.z << '\n'           //
              << "    " << lookAt.x << "    " << lookAt.y << "    " << lookAt.z << '\n'  //
              << "    " << up.x << "    " << up.y << "    " << up.z << '\n';             //
}

void ImGuiUserInterface::togglePause()
{
    m_paused = !m_paused;
    m_options.oneShotGeometry = m_paused;
    m_options.oneShotMaterial = m_paused;
}

bool ImGuiUserInterface::beforeLaunch( OutputBuffer& output )
{
    glfwPollEvents();
    return updateState( output );
}

void ImGuiUserInterface::afterLaunch( OutputBuffer& output )
{
    m_frameRate.update();
    displayOutput( output );
    renderUI();
    glfwSwapBuffers( m_window );
}

bool ImGuiUserInterface::shouldClose() const
{
    return glfwWindowShouldClose( m_window ) != 0;
}

static ImGuiUserInterface* self( GLFWwindow* window )
{
    return static_cast<ImGuiUserInterface*>( glfwGetWindowUserPointer( window ) );
}

void ImGuiUserInterface::mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    if( !ImGui::GetIO().WantCaptureMouse )
        self( window )->handleMouseButton( window, button, action, mods );
}

void ImGuiUserInterface::cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    if( !ImGui::GetIO().WantCaptureMouse )
        self( window )->handleCursorPos( window, xpos, ypos );
}

void ImGuiUserInterface::windowSizeCallback( GLFWwindow* window, int width, int height )
{
    if( !ImGui::GetIO().WantCaptureMouse )
        self( window )->handleSize( window, width, height );
}

void ImGuiUserInterface::scrollCallback( GLFWwindow* window, double xoffset, double yoffset )
{
    if( !ImGui::GetIO().WantCaptureMouse )
        self( window )->handleScroll( window, xoffset, yoffset );
}

void ImGuiUserInterface::keyCallback( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods )
{
    if( !ImGui::GetIO().WantCaptureKeyboard )
        self( window )->handleKey( window, key, scanCode, action, mods );
}

inline uint_t toPixelCoord( double val )
{
    return static_cast<uint_t>( std::floor( val ) );
}

inline uint3 cursorPosition( GLFWwindow* window )
{
    double x;
    double y;
    glfwGetCursorPos( window, &x, &y );
    int w;
    int h;
    glfwGetWindowSize( window, &w, &h );
    return make_uint3( toPixelCoord( x ), h - toPixelCoord( y ), 0 );
}

void ImGuiUserInterface::handleMouseButton( GLFWwindow* window, int button, int action, int mods )
{
    if( action == GLFW_PRESS && mods & GLFW_MOD_CONTROL )
    {
        if( m_debug.enabled )
        {
            m_debug.debugIndex    = cursorPosition( window );
            m_debug.debugIndexSet = true;
            printf( "Debug location set to: [%d, %d, %d]\n", m_debug.debugIndex.x, m_debug.debugIndex.y,
                    m_debug.debugIndex.z );
            m_debugLocationChanged = true;
        }
    }
    else
    {
        m_trackballCamera.mouseButton( window, button, action, mods );
    }
}

void ImGuiUserInterface::handleCursorPos( GLFWwindow* window, double xpos, double ypos )
{
    m_trackballCamera.cursorPos( window, xpos, ypos );
}

void ImGuiUserInterface::handleSize( GLFWwindow* window, int width, int height )
{
    m_trackballCamera.windowSize( window, width, height );
}

void ImGuiUserInterface::handleScroll( GLFWwindow* window, double xoffset, double yoffset )
{
    m_trackballCamera.scroll( window, xoffset, yoffset );
}

void ImGuiUserInterface::handleKey( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods )
{
    bool handled = false;
    if( window == m_window && action == GLFW_PRESS )
    {
        switch( key )
        {
            case GLFW_KEY_Q:  // quit
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose( window, 1 );
                handled = true;
                break;

            case GLFW_KEY_D:
                if( m_debug.enabled && m_options.oneShotDebug )
                {
                    // fire off one debug info dump
                    m_renderer->fireOneDebugDump();
                }
                else
                {
                    // toggle debug mode
                    m_debug.enabled        = !m_debug.enabled;
                    m_debugLocationChanged = true;
                    handled                = true;
                }
                break;

            case GLFW_KEY_G:  // resolve one geometry
                m_scene->resolveOneGeometry();
                break;

            case GLFW_KEY_L:  // dump look at keyword for current camera
                printLookAtKeywordValues();
                break;

            case GLFW_KEY_M:  // resolve one material
                m_scene->resolveOneMaterial();
                break;

            case GLFW_KEY_SPACE:  // toggle pause
                togglePause();
                break;
                
            default:
                break;
        }
    }
    if( !handled )
        m_trackballCamera.key( window, key, scanCode, action, mods );
}

void ImGuiUserInterface::cleanup()
{
    destroyUI();
}

std::shared_ptr<UserInterface> createUserInterface( Options& options, RendererPtr renderer, ScenePtr scene )
{
    return std::make_shared<ImGuiUserInterface>( options, std::move( renderer ), std::move( scene ) );
}

}  // namespace demandPbrtScene
