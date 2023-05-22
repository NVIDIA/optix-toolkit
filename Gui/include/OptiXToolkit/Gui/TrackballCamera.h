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

/// Handles user interaction with a window via a Camera driven by a Trackball.
///
/// The trackball camera is either associated with a window, or it tracks a window.
/// The difference is in how GLFW events are handled.  The simpler case is a tracked
/// window, where the trackball camera registers GLFW event handlers and an optional
/// key event handler can be used to obtain any unhandled key event.
///
/// More sophisticated error handling, such as using ImGUI in conjunction with GLFW,
/// is obtained by using an associated window. An associated window has key,
/// mouse and resize events delegated to it by the owner of the trackball camera.
/// Associated windows do not use the key event handler of the trackball camera.
/// A tracked window has GLFW event handlers installed by the trackball camera and
/// key events are sent to any register key event handler.
///
class TrackballCamera
{
  public:
    using KeyHandler =
        std::function<void( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods, void* context )>;

    /// Constructs a trackball camera not yet connected to any GLFW window.
    TrackballCamera();

    /// Returns true if the camera data has been updated by events.
    bool handleCameraUpdate();

    /// Returns true if the window has been resized by events.
    bool handleResize();

    /// Returns the camera eye position.
    const float3& getCameraEye() const { return m_camera.eye(); }
    /// Sets the camera eye position.
    /// @param val  The camera eye position.
    void setCameraEye( const float3& val ) { m_camera.setEye( val ); }

    /// Returns the camera look at direction.
    const float3& getCameraLookAt() const { return m_camera.lookat(); }
    /// Sets the camera look at direction.
    /// @param val  The camera look at direction.
    void setCameraLookAt( const float3& val ) { m_camera.setLookat( val ); }

    /// Returns the camera up direction.
    const float3& getCameraUp() const { return m_camera.up(); }
    /// Sets the camera up direction.
    /// @param val  The camera up direction.
    void setCameraUp( const float3& val ) { m_camera.setUp( val ); }

    // Returns the camera field of view Y angle.
    float getCameraFovY() const { return m_camera.fovY(); }
    /// Sets the camera field of view Y angle.
    /// @param val  The camera field of view Y angle.
    void setCameraFovY( float val ) { m_camera.setFovY( val ); }

    ///  Returns the camera aspect ratio (width divided by height).
    float getCameraAspectRatio() const { return m_camera.aspectRatio(); }
    /// Sets the camera aspect ratio.
    /// @param val  The camera aspect ratio.
    void setCameraAspectRatio( float val ) { m_camera.setAspectRatio( val ); }

    /// Returns the orthonormal frame vectors for the camera orientation.
    void cameraUVWFrame( float3& u, float3& v, float3& w ) const { m_camera.UVWFrame( u, v, w ); }

    /// Returns the width of the camera view plane in pixels.
    int getWidth() const { return m_width; }
    /// Returns the height of the camera view plane in pixels.
    int getHeight() const { return m_height; }

    /// Track all mouse, key and resize events associated with a window.
    /// @param window The window to track.
    void trackWindow( GLFWwindow* window );

    /// Sets a keyboard handler for this trackball camera.  When a key event is
    /// received by a tracked window and not handled directly by the trackball camera,
    /// it is passed to the keyboard handler.
    /// @param handler  The keyboard handler callback to invoke for key events.
    /// @param context  An arbitrary client context pointer passed to the handler.
    void setKeyHandler( KeyHandler handler, void* context )
    {
        m_keyHandler = std::move( handler );
        m_keyContext = context;
    }

    /// Associates a window with the trackball camera, but does not register event handlers for the window.
    /// A trackball camera associated with a window has the necessary mouse, resize and key events sent to it
    /// via the corresponding functions.
    /// @param window   The window to associate with the trackball camera.
    void associateWindow( GLFWwindow* window );

    /// Send a mouse button event to the trackball camera.
    /// @param window   The window from which the event originates.
    /// @param button   The button parameter from the GLFW event.
    /// @param action   The action parameter from the GLFW event.
    /// @param mods     The mods parameter from the GLFW event.
    void mouseButton( GLFWwindow* window, int button, int action, int mods );

    /// Send a cursor position event to the trackball camera.
    /// @param window   The window from which the event originates.
    /// @param xPos     The xPos parameter from the GLFW event.
    /// @param yPos     The yPos parameter from the GLFW event.
    void cursorPos( GLFWwindow* window, double xPos, double yPos );

    /// Send a window size event to the trackball camera.
    /// @param window   The window from which the event originates.
    /// @param xRes     The xRes parameter from the GLFW event.
    /// @param yRes     The yRes parameter from the GLFW event.
    void windowSize( GLFWwindow* window, int32_t xRes, int32_t yRes );

    /// Send a scroll event to the trackball camera.
    /// @param window   The window from which the event originates.
    /// @param xScroll  The xScroll parameter from the GLFW event.
    /// @param yScroll  The yScroll parameter from the GLFW event.
    void scroll( GLFWwindow* window, double xScroll, double yScroll );

    /// Send a key event to the trackball camera.
    /// @param window   The window from which the event originates.
    /// @param key      The key parameter from the GLFW event.
    /// @param scanCode The scanCode parameter from the GLFW event.
    /// @param action   The action parameter from the GLFW event.
    /// @param mods     The mods parameter from the GLFW event.
    void key( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods );

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
