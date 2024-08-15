// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/HeapSuballocator.h>
#include <OptiXToolkit/Memory/MemoryPool.h>
#include "Util/Math.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

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

    /// The destructor does nothing because the data is owned by the the MemoryPool.
    ~DeviceContextImpl() {}

    /// Allocate memory for per-device data in the given BulkDeviceMemory.
    void allocatePerDeviceData( otk::MemoryPool<otk::DeviceAllocator, otk::HeapSuballocator>* memPool, const Options& options );

    /// Set per-device data pointers to those of the given context.
    void setPerDeviceData( const DeviceContext& other );

    /// Allocate memory for this context in the given BulkDeviceMemory.  Must be preceded by a matching call to reserve().
    void allocatePerStreamData( otk::MemoryPool<otk::DeviceAllocator, otk::HeapSuballocator>* memPool, const Options& options );

  private:
    static const unsigned int BIT_VECTOR_ALIGNMENT = 128;

    static bool isAligned( void* ptr, size_t alignment ) { return reinterpret_cast<uintptr_t>( ptr ) % alignment == 0; }
};

}  // namespace demandLoading
