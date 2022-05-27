//
//  Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
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

#include "Memory/PinnedItemPool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace demandLoading;

class TestPinnedItemPool : public testing::Test
{
  public:
    void SetUp() override
    {
        m_pool.reset( new PinnedItemPool<int>( m_maxItems ) );

        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

        // Create streams.
        DEMAND_CUDA_CHECK( cuStreamCreate( &m_stream, 0U ) );
        DEMAND_CUDA_CHECK( cuStreamCreate( &m_stream2, 0U ) );
    }

    void TearDown() override
    {
        m_pool.reset( nullptr );
        DEMAND_CUDA_CHECK( cuStreamDestroy( m_stream ) );
        DEMAND_CUDA_CHECK( cuStreamDestroy( m_stream2 ) );
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

    m_pool->free( item, m_deviceIndex, m_stream );
    EXPECT_EQ( 0U, m_pool->size() );

    m_pool->shutDown();
}

TEST_F( TestPinnedItemPool, TestWaitSingleThreaded )
{
    int* item = m_pool->allocate();
    EXPECT_EQ( 1U, m_pool->size() );

    m_pool->free( item, m_deviceIndex, m_stream );
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
        int* item = m_pool->allocate();
        m_pool->free( item, m_deviceIndex, m_stream2 );
        finished = true;
    } );

    // Sleep to allow the waiter to run.
    using msec = std::chrono::duration<int, std::milli>;
    std::this_thread::sleep_for( msec( 100 ) );

    // Free the item and busy-wait until the waiter is finished.
    m_pool->free( item, m_deviceIndex, m_stream );
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
