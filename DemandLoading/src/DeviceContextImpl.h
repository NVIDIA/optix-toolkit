//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "Memory/BulkMemory.h"
#include "Util/Exception.h"
#include "Util/Math.h"

#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/Options.h>

namespace demandLoading {

/// DeviceContextImpl encapsulates per-stream device memory allocation logic.
/// Note that the TextureSampler array and the CUtexObjectArray members of
/// DeviceContext are pointers to device memory allocated by ExtensibleArray.
class DeviceContextImpl : public DeviceContext
{
  public:
    /// The constructor delegates to the base struct type.
    DeviceContextImpl()
        : DeviceContext()
    {
    }

    /// Not copyable
    DeviceContextImpl( const DeviceContextImpl& other ) = delete;

    /// Not assignable
    DeviceContextImpl& operator=( const DeviceContextImpl& other ) = delete;

    /// The destructor does nothing because the data is owned by the BulkDeviceMemory.
    ~DeviceContextImpl() {}

    /// Reserve memory for per-device data in the given BulkDeviceMemory.
    void reservePerDeviceData( BulkDeviceMemory* memory, const Options& options );

    /// Allocate memory for per-device data in the given BulkDeviceMemory.
    void allocatePerDeviceData( BulkDeviceMemory* memory, const Options& options );

    /// Set per-device data pointers to those of the given context.
    void setPerDeviceData( const DeviceContext& other );

    /// Reserve memory for per-stream data in the given BulkDeviceMemory.
    static void reservePerStreamData( BulkDeviceMemory* memory, const Options& options );

    /// Allocate memory for this context in the given BulkDeviceMemory.  Must be preceded by a matching call to reserve().
    void allocatePerStreamData( BulkDeviceMemory* memory, const Options& options );

  private:
    static const unsigned int NUM_ARRAYS           = 3;
    static const unsigned int BIT_VECTOR_ALIGNMENT = 128;

    static unsigned int sizeofReferenceBits( const Options& options ) { return idivCeil( options.numPages, 8 ); }

    static size_t sizeofResidenceBits( const Options& options ) { return idivCeil( options.numPages, 8 ); }

    static bool isAligned( void* ptr, size_t alignment ) { return reinterpret_cast<uintptr_t>( ptr ) % alignment == 0; }
};

}  // namespace demandLoading
