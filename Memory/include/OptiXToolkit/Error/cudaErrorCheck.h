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
