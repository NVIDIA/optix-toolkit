//
//  Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
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
