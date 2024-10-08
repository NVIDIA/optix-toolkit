// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>

namespace otk {
namespace error {

/// A generalized error checking mechanism for CUDA device API errors,
/// CUDA runtime API errors and OptiX errors.

// Specialize these two functions for your error types to
// return enum names for the error code and a human readable
// error message for the error code.

/// Return a string giving the name of the error code.
///
/// @param value    The error code to be named.
/// @returns        A string naming the error code.
///
// ReSharper disable once CppFunctionIsNotImplemented
template <typename T>
std::string getErrorName( T value );

/// Specialization for assertions.
template <>
inline std::string getErrorName( bool value )
{
    return value ? "true" : "false";
}

/// Specialization for int status.
template <>
inline std::string getErrorName( int value )
{
    return std::to_string( value );
}

/// Return a string giving the error message for an error code.
///
/// @param value    The error code to be described.
/// @returns        A string describing the error.
///
// ReSharper disable once CppFunctionIsNotImplemented
template <typename T>
std::string getErrorMessage( T value );

/// Specialization for assertions.
template <>
inline std::string getErrorMessage( bool value )
{
    return value ? "true" : "false";
}

/// Specialization for integer status.
template <>
inline std::string getErrorMessage( int value )
{
    return std::to_string( value );
}

/// Identify an error code as a failure.
///
/// Assume that any non-zero value represents failure;
/// if your error type represents success with some non-zero value,
/// then specialize this function for your distinct error type.
///
/// @param value    The error code value to be tested.
/// @returns        true if the code represents failure.
///
template <typename T>
bool isFailure( T value )
{
    return static_cast<bool>( value );
}

/// Build a complete error string.
///
/// @param error    The failed error code to be reported.
/// @param expr     The originating source code expression that generated the error.
/// @param file     The file name containing the expression.
/// @param line     The source line number containing the expression.
/// @param extra    Optional additional error text.
template <typename T>
std::string makeErrorString( T error, const char* expr, const char* file, unsigned int line, const char* extra )
{
    std::string message{ file };
    message += '(' + std::to_string( line ) + "): " + expr + " failed with error " + std::to_string( static_cast<int>( error ) );
    const std::string errorName{ getErrorName( error ) };
    if( !errorName.empty() )
        message += " (" + errorName + ')';
    const std::string errorMessage{ getErrorMessage( error ) };
    if( !errorMessage.empty() )
        message += ' ' + errorMessage;
    if( extra != nullptr )
        message += std::string( ": " ) + extra;
    return message;    
}

/// Checks an error code and reports detected failures by throwing std::runtime_error.
///
/// @param result   The error code to be tested for failure.
/// @param expr     The originating source code expression that generated the error.
/// @param file     The file name containing the expression.
/// @param line     The source line number containing the expression.
/// @param extra    Optional additional error text.
template <typename T>
void checkError( T result, const char* expr, const char* file, unsigned int line, const char* extra )
{
    if( isFailure( result ) )
    {
        throw std::runtime_error( makeErrorString( result, expr, file, line, extra ) );
    }
}

/// Checks an error code and reports detected failures via std::cerr.
///
/// @param result   The error code to be tested for failure.
/// @param expr     The originating source code expression that generated the error.
/// @param file     The file name containing the expression.
/// @param line     The source line number containing the expression.
/// @param extra    Optional additional error text.
template <typename T>
void checkErrorNoThrow( T result, const char* expr, const char* file, unsigned int line, const char* extra )
{
    if( isFailure( result ) )
    {
        try
        {
            std::cerr << makeErrorString( result, expr, file, line, extra ) << std::endl;
        }
        catch( ... )
        {
        }
    }
}

}  // namespace errorCheck
}  // namespace otk

/// Check an expression for error
/// @param  expr   The source expression to check.
///
#define OTK_ERROR_CHECK( expr ) ::otk::error::checkError( expr, #expr, __FILE__, __LINE__, /*extra=*/nullptr )

#define OTK_ERROR_CHECK_MSG( expr, msg ) ::otk::error::checkError( expr, #expr, __FILE__, __LINE__, msg )

#define OTK_ERROR_CHECK_NOTHROW( expr ) ::otk::error::checkErrorNoThrow( expr, #expr, __FILE__, __LINE__, /*extra=*/nullptr )


#ifndef NDEBUG
// Note that a non-zero value represents failure
#define OTK_ASSERT( expr ) OTK_ERROR_CHECK( !static_cast<bool>( expr ) )
#define OTK_ASSERT_MSG( expr, msg ) OTK_ERROR_CHECK_MSG( !static_cast<bool>( expr ), msg )
#else
#define OTK_ASSERT( expr ) {}
#define OTK_ASSERT_MSG( expr, msg ) {}
#endif
