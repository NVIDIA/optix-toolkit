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

#include "PagingSystemTestKernels.h"

#include "Memory/DeviceContextPool.h"
#include "Memory/DeviceMemoryManager.h"
#include "Memory/PinnedMemoryManager.h"
#include "PageTableManager.h"
#include "PagingSystem.h"
#include "RequestProcessor.h"
#include "Util/Exception.h"

#include <gtest/gtest.h>

#include <cuda.h>

using namespace demandLoading;

class DevicePaging
{
  public:
    const unsigned int  m_deviceIndex;
    DeviceMemoryManager m_deviceMemoryManager;
    PinnedMemoryManager m_pinnedMemoryManager;
    PagingSystem        m_paging;
    DeviceContextPool   m_contextPool;
    CUstream            m_stream{};

    DevicePaging( unsigned int deviceIndex, const Options& options, RequestProcessor* requestProcessor )
        : m_deviceIndex( deviceIndex )
        , m_deviceMemoryManager( m_deviceIndex, options )
        , m_pinnedMemoryManager( options )
        , m_paging( deviceIndex, options, &m_deviceMemoryManager, &m_pinnedMemoryManager, requestProcessor )
        , m_contextPool( deviceIndex, options )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuStreamCreate( &m_stream, 0U ) );
    }

    ~DevicePaging()
    {
        DEMAND_CUDA_CHECK_NOTHROW( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK_NOTHROW( cuStreamDestroy( m_stream ) );
    }

    std::vector<unsigned long long> requestPages( const std::vector<unsigned int> pageIds )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

        // Copy given pageIds to device.
        size_t        numPages = pageIds.size();
        unsigned int* devPageIds;
        DEMAND_CUDA_CHECK( cudaMalloc( &devPageIds, numPages * sizeof( unsigned int ) ) );
        DEMAND_CUDA_CHECK( cudaMemcpy( devPageIds, pageIds.data(), numPages * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );

        // Allocate buffer of output pages.
        unsigned long long* devPages;
        size_t              sizeofPages = numPages * sizeof( unsigned long long );
        DEMAND_CUDA_CHECK( cudaMalloc( &devPages, sizeofPages ) );
        DEMAND_CUDA_CHECK( cudaMemset( devPages, 0xDEADBEEF, sizeofPages ) );

        // Launch kernel
        DeviceContext* context = m_contextPool.allocate();
        launchPageRequester( m_stream, *context, static_cast<unsigned int>( numPages ), devPageIds, devPages );

        // Copy output pages to host.
        std::vector<unsigned long long> pages( numPages );
        DEMAND_CUDA_CHECK( cudaMemcpy( pages.data(), devPages, sizeofPages, cudaMemcpyDeviceToHost ) );

        // Clean up.
        m_contextPool.free( context );
        DEMAND_CUDA_CHECK( cudaFree( devPageIds ) );
        DEMAND_CUDA_CHECK( cudaFree( devPages ) );

        return pages;
    }

    unsigned int pushMappings()
    {
        DeviceContext* context     = m_contextPool.allocate();
        unsigned int   numMappings = m_paging.pushMappings( *context, m_stream );
        m_contextPool.free( context );
        return numMappings;
    }
};

class TestPagingSystem : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize paging system options.
        m_options.numPages            = 1025;
        m_options.numPageTableEntries = 128;
        m_options.maxRequestedPages   = 63;
        m_options.maxFilledPages      = 65;
        m_options.maxStalePages       = 33;
        m_options.maxEvictablePages   = 31;
        m_options.maxEvictablePages   = 17;
        m_options.useLruTable         = true;
        m_options.maxActiveStreams    = 4;

        m_pageTableManager.reset( new PageTableManager( m_options.numPages ) );
        m_requestProcessor.reset( new RequestProcessor( m_pageTableManager.get(), m_options.maxRequestQueueSize ) );

        // Create per-device PagingSystem, etc.
        int numDevices;
        cudaGetDeviceCount( &numDevices );
        m_devices.reserve( numDevices );
        for( int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
        {
            // Initialize CUDA
            DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
            DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

            // Create PagingSystem, etc.
            m_devices.emplace_back( new DevicePaging( deviceIndex, m_options, m_requestProcessor.get() ) );
        }
        m_firstDevice = m_devices.at( 0 ).get();
    }

    void TearDown() override { m_devices.clear(); }

  protected:
    Options                                    m_options;
    std::unique_ptr<PageTableManager>          m_pageTableManager;
    std::unique_ptr<RequestProcessor>          m_requestProcessor;
    std::vector<std::unique_ptr<DevicePaging>> m_devices;
    DevicePaging*                              m_firstDevice;
};

TEST_F( TestPagingSystem, TestNonResidentPage )
{
    for( auto& device : m_devices )
    {
        // Request a page that is not resident.
        std::vector<unsigned int>       pageIds{0};
        std::vector<unsigned long long> pages = device->requestPages( pageIds );

        // Non-resident pages should map to zero.
        EXPECT_EQ( 0ULL, pages[0] );
    }
}

TEST_F( TestPagingSystem, TestResidentPage )
{
    for( auto& device : m_devices )
    {
        // Map page 0 to an arbitrary value.
        device->m_paging.addMapping( 0, 0, 42ULL );
        EXPECT_EQ( 1U, device->pushMappings() );

        // Fetch the page table entry from a kernel.
        std::vector<unsigned int>       pageIds{0};
        std::vector<unsigned long long> pages = device->requestPages( pageIds );

        // Check the page table entry that the kernel observed.
        EXPECT_EQ( 42ULL, pages[0] );
    }
}

TEST_F( TestPagingSystem, TestUnbackedPageTableEntry )
{
    for( auto& device : m_devices )
    {
        // Page table entries are allocated only for texture samplers, not tiles,
        // so page ids >= Options::numPageTableEntries are mapped to zero.
        const unsigned int pageId = m_options.numPageTableEntries + 1;

        // Map a page id that has no corresponding page table entry.
        device->m_paging.addMapping( pageId, 0 /*lruValue*/, 42ULL );
        EXPECT_EQ( 1U, device->pushMappings() );

        // Look up the mapping from a kernel.
        std::vector<unsigned int>       pageIds{0};
        std::vector<unsigned long long> pages = device->requestPages( pageIds );

        // The mapping should be zero for page ids without corresponding page table entries.
        EXPECT_EQ( 0ULL, pages[0] );
    }
}
