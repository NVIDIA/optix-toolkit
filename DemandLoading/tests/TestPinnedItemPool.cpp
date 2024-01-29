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

#include <OptiXToolkit/Memory/PinnedItemPool.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace demandLoading;

class TestPinnedItemPool : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA.
        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );

        m_pool.reset( new PinnedItemPool<int>( m_maxItems ) );

        // Create streams.
        OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0U ) );
        OTK_ERROR_CHECK( cuStreamCreate( &m_stream2, 0U ) );
    }

    void TearDown() override
    {
        m_pool.reset( nullptr );
        OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
        OTK_ERROR_CHECK( cuStreamDestroy( m_stream2 ) );
    }

  protected:
    const size_t m_maxItems = 1;

    std::unique_ptr<PinnedItemPool<int>> m_pool;

    CUstream     m_stream;
    CUstream     m_stream2;
    unsigned int m_deviceIndex = 0;
};

TEST_F( TestPinnedItemPool, TestUnused )
{
    EXPECT_EQ( 0U, m_pool->size() );
    EXPECT_EQ( m_maxItems, m_pool->capacity() );

    m_pool->shutDown();
}

TEST_F( TestPinnedItemPool, TestAllocateAndFree )
{
    int* item = m_pool->allocate();
    EXPECT_EQ( 1U, m_pool->size() );

    m_pool->free( item, m_stream );
    EXPECT_EQ( 0U, m_pool->size() );

    m_pool->shutDown();
}

TEST_F( TestPinnedItemPool, TestWaitSingleThreaded )
{
    int* item = m_pool->allocate();
    EXPECT_EQ( 1U, m_pool->size() );

    m_pool->free( item, m_stream );
    EXPECT_EQ( 0U, m_pool->size() );

    // There are no opertions on the stream, so allocation should not block.
    item = m_pool->allocate();
    EXPECT_EQ( 1U, m_pool->size() );

    m_pool->shutDown();
}

TEST_F( TestPinnedItemPool, TestWaitMultiThreaded )
{
    int* item = m_pool->allocate();
    EXPECT_EQ( 1U, m_pool->size() );

    // Subsequent allocation in a separate thread will block until the item is freed.
    std::atomic<bool> finished( false );
    std::thread       waiter( [this, &finished] {
        // Initialize CUDA.
        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );

        int* item = m_pool->allocate();
        m_pool->free( item, m_stream2 );
        finished = true;
    } );

    // Sleep to allow the waiter to run.
    using msec = std::chrono::duration<int, std::milli>;
    std::this_thread::sleep_for( msec( 100 ) );

    // Free the item and busy-wait until the waiter is finished.
    m_pool->free( item, m_stream );
    while( !finished.load() )
    {
        std::this_thread::sleep_for( msec( 10 ) );
    }
    EXPECT_EQ( 0U, m_pool->size() );

    waiter.join();
    m_pool->shutDown();
}

// Test shutting down the m_pool while a thread is waiting.
TEST_F( TestPinnedItemPool, TestShutDown )
{
    // Allocate an item, but don't free it.
    m_pool->allocate();
    EXPECT_EQ( 1U, m_pool->size() );

    // Subsequent allocation in a separate thread will block until m_pool is shut down.
    std::atomic<bool> finished( false );
    std::thread       waiter( [this, &finished] {
        void* item = m_pool->allocate();
        EXPECT_EQ( nullptr, item );
        finished = true;
    } );

    // Sleep to allow the waiter to run.
    using msec = std::chrono::duration<int, std::milli>;
    std::this_thread::sleep_for( msec( 100 ) );

    // Shut down the m_pool and busy-wait until the waiter is finished.
    m_pool->shutDown();
    while( !finished.load() )
    {
        std::this_thread::sleep_for( msec( 10 ) );
    }
    waiter.join();
}
