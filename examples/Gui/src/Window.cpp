//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Gui/GLCheck.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/Window.h>
#include <OptiXToolkit/Gui/glfw3.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <stb_image_write.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

namespace otk {

static void errorCallback( int error, const char* description )
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
}


void initGL()
{
    if( !gladLoadGL() )
        throw std::runtime_error( "Failed to initialize GL" );

    GL_CHECK( glClearColor( 0.212f, 0.271f, 0.31f, 1.0f ) );
    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT ) );
}

void initGLFW()
{
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw std::runtime_error( "Failed to initialize GLFW" );

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE );

    GLFWwindow* window = glfwCreateWindow( 64, 64, "", nullptr, nullptr );
    if( !window )
        throw std::runtime_error( "Failed to create GLFW window" );

    glfwMakeContextCurrent( window );
    glfwSwapInterval( 0 );  // No vsync
}

GLFWwindow* initGLFW( const char* window_title, int width, int height )
{
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw std::runtime_error( "Failed to initialize GLFW" );

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    window = glfwCreateWindow( width, height, window_title, nullptr, nullptr );
    if( !window )
        throw std::runtime_error( "Failed to create GLFW window" );

    glfwMakeContextCurrent( window );
    glfwSwapInterval( 0 );  // No vsync

    return window;
}


void displayBufferWindow( const char* title, const ImageBuffer& buffer )
{
    //
    // Initialize GLFW state
    //
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw std::runtime_error( "Failed to initialize GLFW" );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    window = glfwCreateWindow( buffer.width, buffer.height, title, nullptr, nullptr );
    if( !window )
        throw std::runtime_error( "Failed to create GLFW window" );
    glfwMakeContextCurrent( window );
    glfwSetKeyCallback( window, keyCallback );


    //
    // Initialize GL state
    //
    initGL();
    // Scope GLDisplay to ensure owned gl resources are cleaned up before glfwTermninate.
    {
        GLDisplay display( buffer.pixel_format );

        GLuint pbo = 0u;
        GL_CHECK( glGenBuffers( 1, &pbo ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, pbo ) );
        GL_CHECK( glBufferData( GL_ARRAY_BUFFER, pixelFormatSize( buffer.pixel_format ) * buffer.width * buffer.height,
                                buffer.data, GL_STREAM_DRAW ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );

        //
        // Display loop
        //
        int framebuf_res_x = 0, framebuf_res_y = 0;
        do
        {
            glfwWaitEvents();
            glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
            display.display( buffer.width, buffer.height, framebuf_res_x, framebuf_res_y, pbo );
            glfwSwapBuffers( window );
        } while( !glfwWindowShouldClose( window ) );
    }

    glfwDestroyWindow( window );
    glfwTerminate();
}


void parseDimensions( const char* arg, int& width, int& height )
{
    // look for an 'x': <width>x<height>
    size_t width_end    = strchr( arg, 'x' ) - arg;
    size_t height_begin = width_end + 1;

    if( height_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        const char* height_arg = &arg[height_begin];

        // copy width to null-terminated string
        char width_arg[32];
        strncpy( width_arg, arg, width_end );
        width_arg[width_end] = '\0';

        // terminate the width string
        width_arg[width_end] = '\0';

        width  = atoi( width_arg );
        height = atoi( height_arg );
        return;
    }
    const std::string err = "Failed to parse width, height from string '" + std::string( arg ) + "'";
    throw std::invalid_argument( err.c_str() );
}

double currentTime()
{
    return std::chrono::duration_cast< std::chrono::duration< double > >
        ( std::chrono::high_resolution_clock::now().time_since_epoch() ).count();
}

void ensureMinimumSize( int& w, int& h )
{
    if( w <= 0 )
        w = 1;
    if( h <= 0 )
        h = 1;
}

void ensureMinimumSize( unsigned& w, unsigned& h )
{
    if( w == 0 )
        w = 1;
    if( h == 0 )
        h = 1;
}

} // namespace otk
