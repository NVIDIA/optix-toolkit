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

#include <OptiXToolkit/Gui/TrackballCamera.h>

namespace otk {

TrackballCamera::TrackballCamera()
{
    // The trackball depends on the camera, so the camera must be initialized first.
    initCamera();
    initTrackball();
}

void TrackballCamera::initCamera()
{
    // Two variables can be modified to see how precision affects the results:
    // dollyCam can be increased to move the camera away from the model and tighten its FOV.
    // worldTranslate can be used to move the camera and world to a different location.
    const float  dollyCamera    = 1.0f;
    const float3 worldTranslate = make_float3( 0.0f, 0.0f, 0.0f );

    // The traditional view given in the standard procedural databases code.
    const float3 eye    = make_float3( dollyCamera * 2.1f, dollyCamera * 1.7f, dollyCamera * -1.3f ) + worldTranslate;
    const float3 lookAt = make_float3( 0.0f, 0.0f, 0.0f ) + worldTranslate;
    const float3 up     = make_float3( 0.0f, 1.0f, 0.0f );
    const float  fovY   = 45.0f / dollyCamera;

    m_camera.setEye( eye );
    m_camera.setLookAt( lookAt );
    m_camera.setUp( up );
    m_camera.setFovY( fovY );
}

void TrackballCamera::initTrackball()
{
    m_trackball.setCamera( &m_camera );
    m_trackball.setMoveSpeed( 10.0f );
    const float3 xAxis = make_float3( 1.0f, 0.0f, 0.0f );
    const float3 yAxis = make_float3( 0.0f, 1.0f, 0.0f );
    const float3 zAxis = make_float3( 0.0f, 0.0f, 1.0f );
    m_trackball.setReferenceFrame( xAxis, zAxis, yAxis );
    m_trackball.setGimbalLock( true );
}

void TrackballCamera::associateWindow( GLFWwindow* window )
{
    m_window = window;
    glfwGetWindowSize( window, &m_width, &m_height );
    glfwSetWindowAspectRatio( window, m_width, m_height );
}

void TrackballCamera::trackWindow( GLFWwindow* window )
{
    associateWindow( window );
    glfwSetWindowUserPointer( window, this );
    glfwSetMouseButtonCallback( window, mouseButtonCallback );
    glfwSetCursorPosCallback( window, cursorPosCallback );
    glfwSetWindowSizeCallback( window, windowSizeCallback );
    glfwSetScrollCallback( window, scrollCallback );
    glfwSetKeyCallback( window, keyCallback );
}

bool TrackballCamera::handleCameraUpdate()
{
    if( !m_cameraChanged )
        return false;

    m_cameraChanged = false;
    m_camera.setAspectRatio( static_cast<float>( m_width ) / static_cast<float>( m_height ) );
    return true;
}

bool TrackballCamera::handleResize()
{
    if( !m_sizeChanged )
        return false;

    m_sizeChanged = false;
    return true;
}

void TrackballCamera::mouseButton( GLFWwindow* window, int button, int action, int /*mods*/ )
{
    if( window != m_window )
        return;

    if( action == GLFW_PRESS )
    {
        double xPos;
        double yPos;
        glfwGetCursorPos( window, &xPos, &yPos );
        m_mouseButton = button;
        m_trackball.startTracking( static_cast<int>( xPos ), static_cast<int>( yPos ) );
    }
    else
    {
        m_mouseButton = MOUSE_BUTTON_NONE;
        m_prevYPos    = RESET_Y_POS;
    }
}

void TrackballCamera::cursorPos( GLFWwindow* window, double xPos, double yPos )
{
    if( window != m_window || m_mouseButton == MOUSE_BUTTON_NONE )
        return;

    if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )
    {
        m_trackball.setViewMode( otk::Trackball::LookAtFixed );
        m_trackball.updateTracking( static_cast<int>( xPos ), static_cast<int>( yPos ), m_width, m_height );
        m_cameraChanged = true;
    }
    else if( m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )
    {
        m_trackball.setViewMode( otk::Trackball::EyeFixed );
        m_trackball.updateTracking( static_cast<int>( xPos ), static_cast<int>( yPos ), m_width, m_height );
        m_cameraChanged = true;
    }
    else if( m_mouseButton == GLFW_MOUSE_BUTTON_MIDDLE )
    {
        if( m_prevYPos != RESET_Y_POS && m_prevYPos != static_cast<int>( yPos ) )
        {
            m_trackball.zoom( m_prevYPos - static_cast<int>( yPos ) );
            m_cameraChanged = true;
        }
        m_prevYPos = static_cast<int>( yPos );
    }
}

void TrackballCamera::windowSize( GLFWwindow* window, int32_t xRes, int32_t yRes )
{
    if( window != m_window )
        return;

    m_width         = xRes;
    m_height        = yRes;
    m_cameraChanged = true;
    m_sizeChanged   = true;
}

void TrackballCamera::scroll( GLFWwindow* window, double /*xScroll*/, double yScroll )
{
    if( window != m_window )
        return;

    m_cameraChanged = m_trackball.wheelEvent( static_cast<int>( yScroll ) );
}

void TrackballCamera::key( GLFWwindow* window, int32_t key, int32_t scanCode, int32_t action, int32_t mods )
{
    if( window != m_window )
        return;

    if( m_keyHandler )
        m_keyHandler( window, key, scanCode, action, mods, m_keyContext );
}

}  // namespace demandGeometryViewer
