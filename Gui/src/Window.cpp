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

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/Gui/GLCheck.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/Window.h>
#include <OptiXToolkit/Util/Exception.h>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

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


static void savePPM( const unsigned char* Pix, const char* fname, int wid, int hgt, int chan )
{
    if( Pix == NULL || wid < 1 || hgt < 1 )
        throw Exception( "savePPM: Image is ill-formed. Not saving" );

    if( chan != 1 && chan != 3 && chan != 4 )
        throw Exception( "savePPM: Attempting to save image with channel count != 1, 3, or 4." );

    std::ofstream OutFile( fname, std::ios::out | std::ios::binary );
    if( !OutFile.is_open() )
        throw Exception( "savePPM: Could not open file for" );

    bool is_float = false;
    OutFile << 'P';
    OutFile << ( ( chan == 1 ? ( is_float ? 'Z' : '5' ) : ( chan == 3 ? ( is_float ? '7' : '6' ) : '8' ) ) )
            << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write( reinterpret_cast<char*>( const_cast<unsigned char*>( Pix ) ), wid*hgt*chan*( is_float ? 4 : 1 ) );
    OutFile.close();
}

size_t pixelFormatSize( BufferImageFormat format )
{
    switch( format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            return sizeof( char ) * 4;
        case BufferImageFormat::FLOAT3:
            return sizeof( float ) * 3;
        case BufferImageFormat::FLOAT4:
            return sizeof( float ) * 4;
        default:
            throw Exception( "otk::pixelFormatSize: Unrecognized buffer format" );
    }
}

void initGL()
{
    if( !gladLoadGL() )
        throw Exception( "Failed to initialize GL" );

    GL_CHECK( glClearColor( 0.212f, 0.271f, 0.31f, 1.0f ) );
    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT ) );
}

void initGLFW()
{
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw Exception( "Failed to initialize GLFW" );

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE );

    GLFWwindow* window = glfwCreateWindow( 64, 64, "", nullptr, nullptr );
    if( !window )
        throw Exception( "Failed to create GLFW window" );

    glfwMakeContextCurrent( window );
    glfwSwapInterval( 0 );  // No vsync
}

GLFWwindow* initGLFW( const char* window_title, int width, int height )
{
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw Exception( "Failed to initialize GLFW" );

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    window = glfwCreateWindow( width, height, window_title, nullptr, nullptr );
    if( !window )
        throw Exception( "Failed to create GLFW window" );

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
        throw Exception( "Failed to initialize GLFW" );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    window = glfwCreateWindow( buffer.width, buffer.height, title, nullptr, nullptr );
    if( !window )
        throw Exception( "Failed to create GLFW window" );
    glfwMakeContextCurrent( window );
    glfwSetKeyCallback( window, keyCallback );


    //
    // Initialize GL state
    //
    initGL();
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

    glfwDestroyWindow( window );
    glfwTerminate();
}


static float toSRGB( float c )
{
    float invGamma = 1.0f / 2.4f;
    float powed    = std::pow( c, invGamma );
    return c < 0.0031308f ? 12.92f * c : 1.055f * powed - 0.055f;
}

void saveImage( const char* fname, const ImageBuffer& image, bool disable_srgb_conversion )
{
    const std::string filename( fname );
    if( filename.length() < 5 )
        throw Exception( "otk::saveImage(): Failed to determine filename extension" );

    const std::string ext = filename.substr( filename.length()-3 );
    if( ext == "PPM" || ext == "ppm" )
    {
        //
        // Note -- we are flipping image vertically as we write it into output buffer
        //
        const int32_t width  = image.width;
        const int32_t height = image.height;
        std::vector<unsigned char> pix( width*height*3 );

        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        pix[ dst_idx+0] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+0 ];
                        pix[ dst_idx+1] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+1 ];
                        pix[ dst_idx+2] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+2 ];
                    }
                }
            } break;

            case BufferImageFormat::FLOAT3:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 3*width*j            + 3*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            case BufferImageFormat::FLOAT4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            default:
            {
                throw Exception( "otk::saveImage(): Unrecognized image buffer pixel format.\n" );
            }
        }

        savePPM( pix.data(), filename.c_str(), width, height, 3 );
    }
    else
    {
        throw Exception( ( "otk::saveImage(): Failed unsupported filetype '" + ext + "'" ).c_str() );
    }
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
