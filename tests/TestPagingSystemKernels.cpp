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

#include "Memory/DeviceContextPool.h"
#include "PagingSystemKernels.h"

#include <gtest/gtest.h>

#include <cuda.h>

#include <algorithm> 

using namespace demandLoading;

class TestPagingSystemKernels : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

        // Initialize paging system options.
        m_options.numPages          = 1025;
        m_options.maxRequestedPages = 65;
        m_options.maxFilledPages    = 63;
        m_options.maxStalePages     = 33;
        m_options.maxEvictablePages = 31;
        m_options.maxStagedPages    = 31;
        m_options.useLruTable       = true;

        // Allocate and initialize device context.
        m_contextPool.reset( new DeviceContextPool( m_options ) );
        m_context = m_contextPool->allocate();
    }

    const DeviceContext& getContext() const { return *m_context; }

  private:
    const unsigned int                 m_deviceIndex = 0;
    Options                            m_options{};
    BulkDeviceMemory                   m_contextMemory;
    std::unique_ptr<DeviceContextPool> m_contextPool;
    DeviceContext*                     m_context;
};

TEST_F( TestPagingSystemKernels, TestEmptyPullRequests )
{
    CUstream stream{};
    launchPullRequests( stream, getContext(), 0 /*launchNum*/, 0 /*lruThreshold*/, 0 /*startPage*/,
                        getContext().pageTable.capacity /*endPage*/ );
}

TEST_F( TestPagingSystemKernels, TestInvalidatePages )
{
    // Set first word to all 1's
    DEMAND_CUDA_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( getContext().residenceBits ), 0xFF, 4 ) );

    // Invalidate the following this list of pages
    std::vector<unsigned int> invalidatedPages = { 1, 2, 4, 8 };
    unsigned int numInvalidatedPages = static_cast<unsigned int>( invalidatedPages.size() );
    unsigned int invalidMask = 0xFFFFFFFF;
    for( unsigned int p : invalidatedPages )
        invalidMask = invalidMask ^ (1u << p);

    // Copy list of invalidated pages to device and call invalidate kernel
    CUstream stream{};
    DEMAND_CUDA_CHECK( cudaMemcpy( getContext().invalidatedPages.data, invalidatedPages.data(),
                                   numInvalidatedPages * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );
    launchInvalidatePages( stream, getContext(), numInvalidatedPages );
    cudaDeviceSynchronize();

    // Pull the first word of the residence bits back to the host
    unsigned int residenceBits;
    DEMAND_CUDA_CHECK( cudaMemcpy( &residenceBits, getContext().residenceBits, sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
    cudaDeviceSynchronize();

    EXPECT_EQ( invalidMask, residenceBits );
}

TEST_F( TestPagingSystemKernels, TestGetStalePages )
{
    // Set first word to all 0's (not resident)
    DEMAND_CUDA_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( getContext().residenceBits ), 0xFF, 4 ) );

    // Set LRU values for the first 16 pages to be the same as the page id.
    // Note that 0xF is the non-evictable value, so it should not show up as a stale page.
    std::vector<unsigned int> lruVals = {0x76543210, 0xFEDCBA98, 0x00000000, 0x00000000};
    DEMAND_CUDA_CHECK( cudaMemcpy( getContext().lruTable, lruVals.data(), lruVals.size() * sizeof( unsigned int ), cudaMemcpyHostToDevice ) );

    // Launch pullRequests
    CUstream stream{};
    unsigned int lruThreshold = 4;
    unsigned int launchNum = 0;
    unsigned int startPage = 0;
    unsigned int endPage = 32;
    launchPullRequests( stream, getContext(), launchNum, lruThreshold, startPage, endPage );
    cudaDeviceSynchronize();

    // Copy list of stale pages back to host
    std::vector<StalePage> stalePages( getContext().stalePages.capacity, StalePage{0,0,0} );
    unsigned int arrayLengths[3];
    DEMAND_CUDA_CHECK( cudaMemcpy( arrayLengths, getContext().arrayLengths.data, 3*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
    stalePages.resize( arrayLengths[STALE_PAGES_LENGTH] );
    DEMAND_CUDA_CHECK( cudaMemcpy( stalePages.data(), getContext().stalePages.data, stalePages.size() * sizeof( StalePage ), cudaMemcpyDeviceToHost ) );

    EXPECT_EQ( 11u, (unsigned int)stalePages.size() );
    for( unsigned int i=0; i<stalePages.size(); ++i )
    {
        EXPECT_TRUE( stalePages[i].pageId >= 4 && stalePages[i].pageId < 16 ); // Check page range
        EXPECT_EQ( stalePages[i].pageId, stalePages[i].lruVal ); // Check lru values.
    }
}

TEST_F( TestPagingSystemKernels, TestPullRequests )
{
    // Set some bits as resident, but request all bits in first word
    DEMAND_CUDA_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( getContext().residenceBits ), 0x0F, 4 ) );
    DEMAND_CUDA_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( getContext().referenceBits ), 0xFF, 4 ) );
    std::vector<unsigned int> expectedPages = { 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31 };

    // Launch pullRequests
    CUstream stream{};
    unsigned int lruThreshold = 4;
    unsigned int launchNum = 0;
    unsigned int startPage = 0;
    unsigned int endPage = 32;
    launchPullRequests( stream, getContext(), launchNum, lruThreshold, startPage, endPage );
    cudaDeviceSynchronize();

    // Copy list of stale pages back to host
    std::vector<unsigned int> requestedPages( getContext().requestedPages.capacity, 0u );
    unsigned int arrayLengths[3];
    DEMAND_CUDA_CHECK( cudaMemcpy( arrayLengths, getContext().arrayLengths.data, 3*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
    requestedPages.resize( arrayLengths[PAGE_REQUESTS_LENGTH] );
    DEMAND_CUDA_CHECK( cudaMemcpy( requestedPages.data(), getContext().requestedPages.data, requestedPages.size() * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );

    EXPECT_EQ( 16u, (unsigned int)requestedPages.size() );
    for( unsigned int i=0; i<requestedPages.size(); ++i )
    {
        EXPECT_TRUE( std::find( expectedPages.begin(), expectedPages.end(), requestedPages[i] ) != expectedPages.end() ); 
    }
}

TEST_F( TestPagingSystemKernels, TestPushMappings )
{
    std::vector<PageMapping> filledPages;
    filledPages.push_back( PageMapping{1, 0, 0ULL} );
    unsigned int numFilledPages = static_cast<unsigned int>( filledPages.size() );

    CUstream stream{};
    DEMAND_CUDA_CHECK( cudaMemcpy( getContext().filledPages.data, filledPages.data(),
                                   numFilledPages * sizeof( PageMapping ), cudaMemcpyHostToDevice ) );
    launchPushMappings( stream, getContext(), numFilledPages );
    cudaDeviceSynchronize();

    // Pull the first word of the residence bits back to the host
    unsigned int residenceBits;
    DEMAND_CUDA_CHECK( cudaMemcpy( &residenceBits, getContext().residenceBits, sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
    cudaDeviceSynchronize();

    EXPECT_EQ( 2U, residenceBits );
}
