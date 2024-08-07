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

#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <OptiXToolkit/Error/ErrorCheck.h>

#include <sstream>

namespace otk {
namespace error {

/// Specialization for OptiX error names.
template <>
inline std::string getErrorName( OptixResult value )
{
    return ::optixGetErrorName( value );
}

/// Specialization for OptiX error messages.
template <>
inline std::string getErrorMessage( OptixResult value )
{
    return ::optixGetErrorString( value );
}

inline void optixCheckLog( OptixResult res, const char* log, size_t sizeof_log, size_t sizeof_log_returned, const char* call, const char* file, unsigned int line )
{
    if( isFailure( res ) )
    {
        std::stringstream ss;
        ss << "; Log:\n" << log << ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" ) << '\n';
        throw std::runtime_error( makeErrorString( res, call, file, line, ss.str().c_str() ) );
    }
}

}  // namespace error
}  // namespace otk

#define OTK_ERROR_CHECK_LOG( call )                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        char LOG[400];                                                                                                 \
        LOG[0]          = 0;                                                                                           \
        size_t LOG_SIZE = sizeof( LOG );                                                                               \
        ::otk::error::optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call, __FILE__, __LINE__ );                  \
    } while( false )
