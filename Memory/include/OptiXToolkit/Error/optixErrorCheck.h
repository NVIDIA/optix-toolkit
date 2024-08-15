// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
