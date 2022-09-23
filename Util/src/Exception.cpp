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

#include <OptiXToolkit/Util/Exception.h>

#include <optix_stubs.h>

#include <iostream>
#include <sstream>

namespace otk {

std::string Exception::createMessage( OptixResult res, const char* msg )
{
    std::ostringstream out;
    out << ::optixGetErrorName( res ) << ": " << msg;
    return out.str();
}

void optixCheck( OptixResult res, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        throw Exception( res, ss.str().c_str() );
    }
}

void optixCheckLog( OptixResult res, const char* log, size_t sizeof_log, size_t sizeof_log_returned, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
           << log << ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" ) << '\n';
        throw Exception( res, ss.str().c_str() );
    }
}

void optixCheckNoThrow( OptixResult res, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        std::terminate();
    }
}

void cudaCheck( cudaError_t error, const char* call, const char* file, unsigned int line )
{
    if( error != cudaSuccess )
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '" << cudaGetErrorString( error ) << "' (" << file << ":"
           << line << ")\n";
        throw Exception( ss.str().c_str() );
    }
}

void cudaCheck( CUresult result, const char* expr, const char* file, unsigned int line )
{
    if( result != CUDA_SUCCESS )
    {
        const char* errorStr;
        cuGetErrorString( result, &errorStr );
        std::stringstream ss;
        ss << "CUDA call (" << expr << " ) failed with error: '" << errorStr << "' (" << file << ":" << line << ")\n";
        throw Exception( ss.str().c_str() );
    }
}

void cudaSyncCheck( const char* file, unsigned int line )
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess )
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '" << cudaGetErrorString( error ) << "' (" << file << ":" << line << ")\n";
        throw Exception( ss.str().c_str() );
    }
}

void cudaCheckNoThrow( cudaError_t error, const char* call, const char* file, unsigned int line ) noexcept
{
    if( error != cudaSuccess )
    {
        std::cerr << "CUDA call (" << call << " ) failed with error: '" << cudaGetErrorString( error ) << "' (" << file
                  << ":" << line << ")\n";
        std::terminate();
    }
}

void cudaCheckNoThrow( CUresult result, const char* expr, const char* file, unsigned int line ) noexcept
{
    if( result != CUDA_SUCCESS )
    {
        const char* errorStr;
        cuGetErrorString( result, &errorStr );
        std::cerr << "CUDA call (" << expr << " ) failed with error: '" << errorStr << "' (" << file << ":" << line << ")\n";
        std::terminate();
    }
}

void assertCond( bool result, const char* cond, const char* file, unsigned int line )
{
    if( !result )
    {
        std::stringstream ss;
        ss << file << " (" << line << "): " << cond;
        throw Exception( ss.str().c_str() );
    }
}

void assertCondMsg( bool result, const char* cond, const std::string& msg, const char* file, unsigned int line )
{
    if( !result )
    {
        std::stringstream ss;
        ss << msg << ": " << file << " (" << line << "): " << cond;
        throw Exception( ss.str().c_str() );
    }
}

[[noreturn]] void assertFailMsg( const std::string& msg, const char* file, unsigned int line )
{
    std::stringstream ss;
    ss << msg << ": " << file << " (" << line << ')';
    throw Exception( ss.str().c_str() );
}

}  // namespace otk
