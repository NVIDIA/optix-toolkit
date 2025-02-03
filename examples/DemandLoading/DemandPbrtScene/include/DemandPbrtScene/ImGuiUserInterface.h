// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Gui/glad.h>  // Glad insists on being included first.

#include "DemandPbrtScene/UserInterface.h"

#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/FrameRate.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Sample.h"
#include "DemandPbrtScene/UserInterfaceStatistics.h"

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/TrackballCamera.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/ShaderUtil/DebugLocation.h>

#include <imgui.h>

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
    void dumpStats() const;

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
