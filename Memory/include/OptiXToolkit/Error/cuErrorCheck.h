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
#ifndef OTK_ERROR_CU_ERROR_CHECK_H
#define OTK_ERROR_CU_ERROR_CHECK_H

#include <OptiXToolkit/Error/ErrorCheck.h>

#include <cuda.h>

#include <sstream>

namespace otk {
namespace error {

/// Specializations for CUDA driver API error names.
template <>
inline std::string getErrorName( CUresult value )
{
    const char* name{};
    ::cuGetErrorName( value, &name );
    if( name )
        return name;
    return {};
}

/// Specializations for CUDA driver API error messages.
template <>
inline std::string getErrorMessage( CUresult value )
{
    const char* message{};
    ::cuGetErrorString( value, &message );
    if( message )
        return message;
    return {};
}

/// Get the context of a stream
inline CUcontext getCudaContext( CUstream stream ) 
{
    CUcontext context;
    OTK_ERROR_CHECK( cuStreamGetCtx( stream, &context ) );
    return context;
}

/// Verify that the current CUDA context matches the given context.
inline void cudaContextCheck( CUcontext expected, const char* file, unsigned int line )
{
    CUcontext current;
    OTK_ERROR_CHECK( cuCtxGetCurrent( &current ) );
    if( expected != current )
    {
        std::stringstream ss;
        ss << "Cuda context check failed (" << file << ":" << line << ")\n";
        throw std::runtime_error( ss.str() );
    }
}

#define OTK_ASSERT_CONTEXT_IS( context ) otk::error::cudaContextCheck( context, __FILE__, __LINE__ )
#define OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream ) if( stream ) otk::error::cudaContextCheck( otk::error::getCudaContext( stream ), __FILE__, __LINE__ )

}  // namespace error
}  // namespace otk

#endif
