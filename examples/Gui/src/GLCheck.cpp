// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Gui/glad.h>  // Glad insists on being included first.

#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <iostream>
#include <sstream>

namespace otk {
    
const char* getGLErrorString( GLenum error )
{
    switch( error )
    {
        case GL_NO_ERROR:
            return "No error";
        case GL_INVALID_ENUM:
            return "Invalid enum";
        case GL_INVALID_VALUE:
            return "Invalid value";
        case GL_INVALID_OPERATION:
            return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:
            return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default:
            return "Unknown GL error";
    }
}

void glCheck( const char* call, const char* file, unsigned int line )
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString( err ) << " at " << file << "(" << line << "): " << call << '\n';
        std::cerr << ss.str() << std::endl;
        throw std::runtime_error( ss.str().c_str() );
    }
}

void glCheckErrors( const char* file, unsigned int line )
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString( err ) << " at " << file << "(" << line << ")";
        std::cerr << ss.str() << std::endl;
        throw std::runtime_error( ss.str().c_str() );
    }
}

void checkGLError()
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::ostringstream oss;
        do
        {
            oss << "GL error: " << getGLErrorString( err ) << '\n';
            err = glGetError();
        } while( err != GL_NO_ERROR );

        throw std::runtime_error( oss.str().c_str() );
    }
}

} // namespace otk
