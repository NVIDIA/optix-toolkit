// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

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
