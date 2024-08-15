// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include "DeviceContextImpl.h"
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/FixedSuballocator.h>
#include <OptiXToolkit/Memory/HeapSuballocator.h>
#include <OptiXToolkit/Memory/MemoryPool.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

using namespace demandLoading;
using namespace otk;

class TestDeviceContextImpl : public testing::Test
{
  public:
    const unsigned int m_deviceIndex = 0;
    Options            m_options{};
    TestDeviceContextImpl()
    {
        m_options.numPages          = 1025;
        m_options.maxRequestedPages = 65;
        m_options.maxFilledPages    = 63;
        m_options.maxStalePages     = 33;
        m_options.maxEvictablePages = 31;
        m_options.maxEvictablePages = 17;
        m_options.useLruTable       = true;
    }

    void SetUp() { cudaFree( nullptr ); }
};

TEST_F( TestDeviceContextImpl, TestConstructor )
{
    // Alignment is checked by assertions in the constructor.
    MemoryPool<DeviceAllocator, HeapSuballocator> memPool( new DeviceAllocator(), nullptr );
    DeviceContextImpl context{};

    context.allocatePerDeviceData( &memPool, m_options );
    context.allocatePerStreamData( &memPool, m_options );
}
