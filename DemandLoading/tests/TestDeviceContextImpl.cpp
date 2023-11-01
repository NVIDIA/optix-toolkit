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
