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

#pragma once

#include <imgui.h>
#include <OptiXToolkit/Gui/glad.h>  // Glad insists on being included first.

#include "UserInterface.h"

#include "Dependencies.h"
#include "FrameRate.h"
#include "Options.h"
#include "Params.h"
#include "Sample.h"
#include "UserInterfaceStatistics.h"

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/TrackballCamera.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/ShaderUtil/DebugLocation.h>

#include <iostream>
#include <memory>
#include <string>

struct GLFWwindow;

namespace otk {

class GLDisplay;

namespace pbrt {

class Logger;
class SceneLoader;

}  // namespace pbrt
}  // namespace otk

namespace demandPbrtScene {

class ApiImpl;
class MeshInfoReader;
class Renderer;
class Scene;

using GLDisplayPtr = std::shared_ptr<otk::GLDisplay>;

class ImGuiUserInterface : public UserInterface
{
  public:
    ImGuiUserInterface( Options& options, RendererPtr renderer, ScenePtr scene );
    ~ImGuiUserInterface() override = default;

    void initialize( const LookAtParams& lookAt, const PerspectiveCamera& camera ) override;
    void cleanup() override;

    bool beforeLaunch( OutputBuffer& output ) override;
    void afterLaunch( otk::CUDAOutputBuffer<uchar4>& output ) override;
    bool shouldClose() const override;

    void setStatistics( const UserInterfaceStatistics& stats ) override { m_stats = stats; }

  private:
    void initCamera( const LookAtParams& lookAt, const PerspectiveCamera& camera );
    void createWindow();
    void createUI();
    void setNextWindowBackground() const;
    void renderGeometryCacheStatistics() const;
    void renderImageCacheStatistics( const char* label, const imageSource::CacheStatistics& stats ) const;
    void renderImageSourceFactoryStatistics() const;
    void renderProxyFactoryStatistics() const;
    void renderSceneStatistics() const;
    void renderStatistics() const;
    void renderOptions();
    void renderDebug();
    void renderToggleOption( bool& option, const char* label );
    void renderFPS() const;
    void renderUI();
    void destroyUI();
    bool handleCameraUpdate();
    bool handleResize( otk::CUDAOutputBuffer<uchar4>& output );
    bool handleParamsUpdate();
    bool handleOptionsUpdate();
    bool updateState( otk::CUDAOutputBuffer<uchar4>& output );
    void displayOutput( otk::CUDAOutputBuffer<uchar4>& output );
    void printLookAtKeywordValues();
    void togglePause();

    static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods );
    static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos );
    static void windowSizeCallback( GLFWwindow* window, int width, int height );
    static void scrollCallback( GLFWwindow* window, double xoffset, double yoffset );
    static void keyCallback( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods );
    void        handleMouseButton( GLFWwindow* window, int button, int action, int mods );
    void        handleCursorPos( GLFWwindow* window, double xpos, double ypos );
    void        handleSize( GLFWwindow* window, int width, int height );
    void        handleScroll( GLFWwindow* window, double xoffset, double yoffset );
    void        handleKey( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods );

    UserInterfaceStatistics m_stats{};
    Options&                m_options;
    RendererPtr             m_renderer;
    ScenePtr                m_scene;
    GLDisplayPtr            m_display;
    otk::TrackballCamera    m_trackballCamera;
    GLFWwindow*             m_window{};
    otk::DebugLocation      m_debug{};
    bool                    m_debugLocationChanged{};
    bool                    m_optionsChanged{};
    FrameRate               m_frameRate;
    float                   m_optionsHeight{};
    bool                    m_paused{};
};

}  // namespace demandPbrtScene
