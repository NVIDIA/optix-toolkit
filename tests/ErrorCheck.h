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

#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include <optix.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>

template <typename T>
const char* getErrorName( T result );
template <typename T>
const char* getErrorMessage( T result );

template <>
inline const char* getErrorName( CUresult result )
{
    const char* errorName;
    if( cuGetErrorName( result, &errorName ) )
        errorName = nullptr;
    return errorName;
}

template <>
inline const char* getErrorName( cudaError_t result )
{
    return cudaGetErrorName( result );
}

template <>
inline const char* getErrorName( OptixResult result )
{
    return optixGetErrorName( result );
}

template <>
inline const char* getErrorMessage( CUresult result )
{
    const char* errorMessage;
    if( cuGetErrorString( result, &errorMessage ) )
        errorMessage = nullptr;
    return errorMessage;
}

template <>
inline const char* getErrorMessage( cudaError_t result )
{
    return cudaGetErrorString( result );
}

template <>
inline const char* getErrorMessage( OptixResult result )
{
    return optixGetErrorString( result );
}

[[noreturn]]
inline void reportError( int result, const char* errorName, const char* errorMessage, const char* expr, const char* file, unsigned int line, const char* message = nullptr )
{
    std::ostringstream ss;
    ss << file << '(' << line << "): " << expr << " failed with error " << result;
    if( errorName != nullptr )
        ss << " (" << errorName << ')';
    if( errorMessage != nullptr )
        ss << ' ' << errorMessage;
    if( message != nullptr )
        ss << message;
    throw std::runtime_error( ss.str() );
}

template <typename T>
void errorCheck( T result, const char* expr, const char* file, unsigned int line )
{
    if( result )
    {
        reportError( static_cast<int>( result ), getErrorName( result ), getErrorMessage( result ), expr, file, line );
    }
}

inline void errorCheckLog( OptixResult res, const char* log, size_t logCapacity, size_t logActualSize, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::stringstream ss;
        ss << "Log:\n" << log << ( logActualSize > logCapacity ? "<TRUNCATED>" : "" ) << '\n';
        reportError( res, getErrorName( res ), getErrorMessage( res ), call, file, line, ss.str().c_str() );
    }
}

#define ERROR_CHECK( expr_ ) errorCheck( expr_, #expr_, __FILE__, __LINE__ )

#define OPTIX_CHECK_LOG2( call )                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        char   LOG[512];                                                                                               \
        size_t LOG_SIZE = sizeof( LOG );                                                                               \
        errorCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call, __FILE__, __LINE__ );                                \
    } while( false )

#endif
