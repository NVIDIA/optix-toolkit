// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>

#include <OptiXToolkit/Error/ErrorCheck.h>

namespace otk {
namespace error {

/// Specializations for CUDA runtime error names.
template <>
inline std::string getErrorName( cudaError_t value )
{
    return ::cudaGetErrorName( value );
}

/// Specializations for CUDA runtime error messages.
template <>
inline std::string getErrorMessage( cudaError_t value )
{
    return ::cudaGetErrorString( value );
}

/// Synchronize with a CUDA device and then check for errors.
///
/// @param file     The source file that invoked this function.
/// @param line     The source file line that invoked this function.
inline void syncCheck( const char* file, unsigned int line )
{
    cudaDeviceSynchronize();
    checkError( cudaGetLastError(), "otk::error::syncCheck()", file, line, /*extra=*/nullptr );
}

}  // namespace error
}  // namespace otk

/// Synchronize with a CUDA device and then check for errors at the current file and line location.
#define OTK_CUDA_SYNC_CHECK() ::otk::error::syncCheck( __FILE__, __LINE__ )
