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
#ifndef OTK_ERROR_ERROR_CHECK_H
#define OTK_ERROR_ERROR_CHECK_H

#include <stdexcept>
#include <string>

namespace otk {
namespace error {

// Specialize these two functions for your error types to
// return enum names for the error code and a human readable
// error message for the error code.

// ReSharper disable once CppFunctionIsNotImplemented
template <typename T>
std::string getErrorName( T value );
// ReSharper disable once CppFunctionIsNotImplemented
template <typename T>
std::string getErrorMessage( T value );

// Assume that any non-zero value represents failure;
// if your error type represents success with some non-zero value,
// then specialize this function for your distinct error type.
template <typename T>
bool isFailure( T value )
{
    return static_cast<bool>( value );
}

template <typename T>
[[noreturn]]
void reportError( T error, const char* expr, const char* file, unsigned int line, const char* extra )
{
    std::string message{ file };
    message += '(' + std::to_string( line ) + "): " + expr + " failed with error " + std::to_string( error );
    const std::string errorName{ getErrorName( error ) };
    if( !errorName.empty() )
        message += " (" + errorName + ')';
    const std::string errorMessage{ getErrorMessage( error ) };
    if( !errorMessage.empty() )
        message += ' ' + errorMessage;
    if( extra != nullptr )
        message += extra;
    throw std::runtime_error( message );
}

template <typename T>
void checkError( T result, const char* expr, const char* file, unsigned int line, const char* extra = nullptr )
{
    if( isFailure( result ) )
    {
        reportError( result, expr, file, line, extra );
    }
}

}  // namespace errorCheck
}  // namespace otk

#define OTK_ERROR_CHECK( expr_ ) ::otk::error::checkError( expr_, #expr_, __FILE__, __LINE__ )

#endif
