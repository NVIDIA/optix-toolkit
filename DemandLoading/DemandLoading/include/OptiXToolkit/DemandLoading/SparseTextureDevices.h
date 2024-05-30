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

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

#include <numeric>
#include <vector>

namespace demandLoading {

const unsigned int MAX_DEVICES = 32;

/// Get the number of CUDA devices supported on the system
inline unsigned int getCudaDeviceCount()
{
    int numDevices;
    OTK_ERROR_CHECK( cuDeviceGetCount( &numDevices ) );
    return static_cast<unsigned int>( numDevices );
}

/// Get vector of all CUDA device indices
inline std::vector<unsigned int> getCudaDevices()
{
    std::vector<unsigned int> devices;
    devices.resize( getCudaDeviceCount() );
    std::iota( devices.begin(), devices.end(), 0 );
    return devices;
}

/// Deteremine if a CUDA device supports sparse textures
inline bool deviceSupportsSparseTextures( unsigned int deviceIndex )
{
    int sparseSupport = 0;
    OTK_ERROR_CHECK( cuDeviceGetAttribute( &sparseSupport, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, deviceIndex ) );

    // Skip devices in TCC mode.  This guards against an "operation not supported" error when
    // querying the recommended allocation granularity via cuMemGetAllocationGranularity.
    int inTccMode = 0;
    OTK_ERROR_CHECK( cuDeviceGetAttribute( &inTccMode, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, deviceIndex ) );

    return sparseSupport && !inTccMode;
}

/// Returns the index of the first CUDA device that supports sparse textures
/// or MAX_DEVICES if no devices support sparse textures.
inline unsigned int getFirstSparseTextureDevice()
{
    unsigned int numDevices = getCudaDeviceCount();
    for( unsigned int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
    {
        if( deviceSupportsSparseTextures( deviceIndex ) )
            return deviceIndex;
    }
    return MAX_DEVICES;
}

/// Get bitmap of CUDA devices that support sparse textures
inline std::vector<unsigned int> getSparseTextureDevices()
{
    unsigned int              numDevices = getCudaDeviceCount();
    std::vector<unsigned int> devices;
    for( unsigned int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
    {
        if( deviceSupportsSparseTextures( deviceIndex ) )
            devices.push_back( deviceIndex );
    }
    return devices;
}

}  // namespace demandLoading
