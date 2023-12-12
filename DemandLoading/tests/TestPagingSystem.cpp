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

#include "Memory/DeviceMemoryManager.h"
#include "PageTableManager.h"
#include "PagingSystem.h"
#include "ThreadPoolRequestProcessor.h"
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <gtest/gtest.h>

#include <cuda.h>

#include <memory>

const unsigned long long PINNED_ALLOC = 2u << 20;
const unsigned long long MAX_PINNED_MEM = 32u << 20;

using namespace demandLoading;
using namespace otk;

class DevicePaging
{
  public:
    const unsigned int                            m_deviceIndex;
    DeviceMemoryManager                           m_deviceMemoryManager;
    MemoryPool<PinnedAllocator, RingSuballocator> m_pinnedMemoryPool;
    PagingSystem                                  m_paging;
    CUstream                                      m_stream{};
    std::unique_ptr<bool[]>                       m_pagesResident;

    DevicePaging( unsigned int deviceIndex, std::shared_ptr<Options> options, RequestProcessor* requestProcessor )
        : m_deviceIndex( deviceIndex )
        , m_deviceMemoryManager( options )
        , m_pinnedMemoryPool( new PinnedAllocator(), new RingSuballocator( PINNED_ALLOC ), PINNED_ALLOC, MAX_PINNED_MEM )
        , m_paging( options, &m_deviceMemoryManager, &m_pinnedMemoryPool, requestProcessor )
    {
        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0U ) );
    }

    ~DevicePaging()
    {
        OTK_ERROR_CHECK_NOTHROW( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK_NOTHROW( cuStreamDestroy( m_stream ) );
    }

    std::vector<unsigned long long> requestPages( const std::vector<unsigned int> pageIds )
    {
        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );

        // Copy given pageIds to device.
        size_t        numPages = pageIds.size();
        unsigned int* devPageIds;
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devPageIds ), numPages * sizeof( unsigned int ) ) );
        OTK_ERROR_CHECK( cudaMemcpy( devPageIds, pageIds.data(), numPages * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );

        // Allocate buffer of output page table entries and resident flags.
        unsigned long long* devPages;
        const size_t        sizeofPages = numPages * sizeof( unsigned long long );
        bool*               devPagesResident;
        const size_t        sizeofPagesResident = numPages * sizeof( bool );
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devPages ), sizeofPages ) );
        OTK_ERROR_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( devPages ), 0xFF, sizeofPages ) );
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devPagesResident ), sizeofPagesResident ) );
        OTK_ERROR_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( devPagesResident ), 0xFF, sizeofPagesResident ) );

        // Launch kernel
        DeviceContext* context = m_deviceMemoryManager.allocateDeviceContext();
        launchPageRequester( m_stream, *context, static_cast<unsigned int>( numPages ), devPageIds, devPages, devPagesResident );

        // Copy output values to host.
        std::vector<unsigned long long> pages( numPages );
        OTK_ERROR_CHECK( cudaMemcpy( pages.data(), devPages, sizeofPages, cudaMemcpyDeviceToHost ) );
        m_pagesResident.reset( new bool[numPages] );
        OTK_ERROR_CHECK( cudaMemcpy( m_pagesResident.get(), devPagesResident, sizeofPagesResident, cudaMemcpyDeviceToHost ) );

        // Clean up.
        m_deviceMemoryManager.freeDeviceContext( context );
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devPageIds ) ) );
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devPages ) ) );

        return pages;
    }

    unsigned int pushMappings()
    {
        DeviceContext* context     = m_deviceMemoryManager.allocateDeviceContext();
        unsigned int   numMappings = m_paging.pushMappings( *context, m_stream );
        m_deviceMemoryManager.freeDeviceContext( context );
        return numMappings;
    }
};

class TestPagingSystem : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize paging system options.
        m_options.reset( new Options );
        m_options->numPages            = 1025;
        m_options->numPageTableEntries = 128;
        m_options->maxRequestedPages   = 63;
        m_options->maxFilledPages      = 65;
        m_options->maxStalePages       = 33;
        m_options->maxEvictablePages   = 31;
        m_options->maxEvictablePages   = 17;
        m_options->useLruTable         = true;
        m_options->maxActiveStreams    = 4;

        m_pageTableManager = std::make_shared<PageTableManager>( m_options->numPages, m_options->numPageTableEntries );
        m_requestProcessor.reset( new ThreadPoolRequestProcessor( m_pageTableManager, *m_options ) );

        // Create per-device PagingSystem, etc.
        int numDevices;
        cudaGetDeviceCount( &numDevices );
        m_devices.reserve( numDevices );
        for( int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
        {
            // Initialize CUDA
            OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
            OTK_ERROR_CHECK( cudaFree( nullptr ) );

            // Create PagingSystem, etc.
            m_devices.emplace_back( new DevicePaging( deviceIndex, m_options, m_requestProcessor.get() ) );
        }
        m_firstDevice = m_devices.at( 0 ).get();
    }

  protected:
    std::shared_ptr<Options>                   m_options;
    std::shared_ptr<PageTableManager>          m_pageTableManager;
    std::unique_ptr<RequestProcessor>          m_requestProcessor;
    std::vector<std::unique_ptr<DevicePaging>> m_devices;
    DevicePaging*                              m_firstDevice{};
};

TEST_F( TestPagingSystem, TestNonResidentPage )
{
    for( auto& device : m_devices )
    {
        OTK_ERROR_CHECK( cudaSetDevice( device->m_deviceIndex ) );

        // Request a page that is not resident.
        const unsigned int pageId = 0;
        std::vector<unsigned int>       pageIds{pageId};
        std::vector<unsigned long long> pages = device->requestPages( pageIds );

        // Non-resident pages should map to zero.
        EXPECT_EQ( 0ULL, pages[0] ) << "page " << pageId;
        EXPECT_FALSE( device->m_pagesResident[0] ) << "page " << pageId << " was resident.";
    }
}

TEST_F( TestPagingSystem, TestResidentPage )
{
    for( auto& device : m_devices )
    {
        OTK_ERROR_CHECK( cudaSetDevice( device->m_deviceIndex ) );

        // Map page 0 to an arbitrary value.
        const unsigned int pageId = 0;
        device->m_paging.addMapping( pageId, 0, 42ULL );
        EXPECT_EQ( 1U, device->pushMappings() );

        // Fetch the page table entry from a kernel.
        std::vector<unsigned int>       pageIds{pageId};
        std::vector<unsigned long long> pages = device->requestPages( pageIds );

        // Check the page table entry that the kernel observed.
        EXPECT_EQ( 42ULL, pages[0] ) << "page " << pageId;
        EXPECT_TRUE( device->m_pagesResident[0] ) << "page " << pageId << " was not resident.";
    }
}

TEST_F( TestPagingSystem, TestUnbackedPageTableEntry )
{
    for( auto& device : m_devices )
    {
        OTK_ERROR_CHECK( cudaSetDevice( device->m_deviceIndex ) );

        // Page table entries are allocated only for texture samplers, not tiles,
        // so page ids >= Options::numPageTableEntries are mapped to zero.
        const unsigned int pageId = m_options->numPageTableEntries + 1;
        // Map a page id that has no corresponding page table entry.
        device->m_paging.addMapping( pageId, 0 /*lruValue*/, 42ULL );
        EXPECT_EQ( 1U, device->pushMappings() );

        // Look up the mapping from a kernel.
        std::vector<unsigned int>       pageIds{pageId};
        std::vector<unsigned long long> pages = device->requestPages( pageIds );

        // The mapping should be zero for page ids without corresponding page table entries.
        EXPECT_EQ( 0ULL, pages[0] ) << "page " << pageId;
        EXPECT_TRUE( device->m_pagesResident[0] ) << "page " << pageId << " was not resident.";
    }
}

