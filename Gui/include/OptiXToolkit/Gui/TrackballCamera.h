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
#pragma once

#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/Trackball.h>

#include <GLFW/glfw3.h>

#include <functional>
#include <utility>

namespace otk {

/// Handle all the user interaction with the main window via a camera driven by a trackball.
///
class TrackballCamera
{
  public:
    using KeyHandler =
        std::function<void( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods, void* context )>;

    TrackballCamera();

    /// Associate this trackball camera with a window.
    /// @param window The window to track.
    void trackWindow( GLFWwindow* window );
    void setKeyHandler( KeyHandler handler, void* context )
    {
        m_keyHandler = std::move( handler );
        m_keyContext = context;
    }

    bool handleCameraUpdate();
    bool handleResize();

    const float3& getCameraEye() const { return m_camera.eye(); }
    void          cameraUVWFrame( float3& u, float3& v, float3& w ) const { m_camera.UVWFrame( u, v, w ); }
    int           getWidth() const { return m_width; }
    int           getHeight() const { return m_height; }

  private:
    enum
    {
        RESET_Y_POS       = -99999,
        MOUSE_BUTTON_NONE = -1
    };
    static TrackballCamera* self( GLFWwindow* window )
    {
        return static_cast<TrackballCamera*>( glfwGetWindowUserPointer( window ) );
    }
    static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
    {
        self( window )->mouseButton( window, button, action, mods );
    }
    static void cursorPosCallback( GLFWwindow* window, double xPos, double yPos )
    {
        self( window )->cursorPos( window, xPos, yPos );
    }
    static void windowSizeCallback( GLFWwindow* window, int32_t xRes, int32_t yRes )
    {
        self( window )->windowSize( window, xRes, yRes );
    }
    static void scrollCallback( GLFWwindow* window, double xScroll, double yScroll )
    {
        self( window )->scroll( window, xScroll, yScroll );
    }
    static void keyCallback( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods )
    {
        self( window )->key( window, key, scanCode, action, mods );
    }

    void initTrackball();
    void initCamera();

    void mouseButton( GLFWwindow* window, int button, int action, int );
    void cursorPos( GLFWwindow* window, double xPos, double yPos );
    void windowSize( GLFWwindow* window, int32_t xRes, int32_t yRes );
    void scroll( GLFWwindow*, double, double yScroll );
    void key( GLFWwindow* window, int32_t key, int32_t, int32_t action, int32_t );

    GLFWwindow* m_window{};
    int         m_width{};
    int         m_height{};
    Camera      m_camera;
    Trackball   m_trackball;
    bool        m_cameraChanged{ true };
    bool        m_sizeChanged{ false };
    int         m_mouseButton{ MOUSE_BUTTON_NONE };
    int         m_prevYPos{ RESET_Y_POS };
    KeyHandler  m_keyHandler{};
    void*       m_keyContext{};
};

}  // namespace demandGeometryViewer
