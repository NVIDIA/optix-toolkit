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

#include "Memory/BulkPinnedItemPool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace demandLoading;

struct Item
{
    int*         array;
    unsigned int length;

    static void reserve( BulkPinnedMemory* memory, unsigned int length ) { memory->reserve<int>( length ); }

    void allocate( BulkPinnedMemory* memory, unsigned int length )
    {
        array  = memory->allocate<int>( length );
        length = length;
    }
};

using ItemPool = BulkPinnedItemPool<Item, unsigned int>;

class TestBulkPinnedItemPool : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

        // Create streams.
        DEMAND_CUDA_CHECK( cuStreamCreate( &m_stream, 0U ) );
        DEMAND_CUDA_CHECK( cuStreamCreate( &m_stream2, 0U ) );
    }

    void TearDown() override
    {
        DEMAND_CUDA_CHECK( cuStreamDestroy( m_stream ) );
        DEMAND_CUDA_CHECK( cuStreamDestroy( m_stream2 ) );
    }

  protected:
    unsigned int m_arrayLength = 32;
    CUstream     m_stream;
    CUstream     m_stream2;
    unsigned int m_deviceIndex = 0;
};

TEST_F( TestBulkPinnedItemPool, TestUnused )
{
    ItemPool pool( 1, m_arrayLength );

    EXPECT_EQ( 0U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );

    pool.shutDown();
}

TEST_F( TestBulkPinnedItemPool, TestAllocateAndFree )
{
    ItemPool pool( 1, m_arrayLength );
    Item*    item = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );

    pool.free( item, m_deviceIndex, m_stream );
    EXPECT_EQ( 0U, pool.size() );

    pool.shutDown();
}

TEST_F( TestBulkPinnedItemPool, TestWaitSingleThreaded )
{
    ItemPool pool( 1, m_arrayLength );
    Item*    item = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );

    pool.free( item, m_deviceIndex, m_stream );
    EXPECT_EQ( 0U, pool.size() );

    // There are no opertions on the stream, so allocation should not block.
    item = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );

    pool.shutDown();
}

TEST_F( TestBulkPinnedItemPool, TestWaitMultiThreaded )
{
    ItemPool pool( 1, m_arrayLength );
    Item*    item = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );

    // Subsequent allocation in a separate thread will block until the item is freed.
    std::atomic<bool> finished( false );
    std::thread       waiter( [this, &pool, &finished] {
        Item* item = pool.allocate();
        pool.free( item, m_deviceIndex, m_stream2 );
        finished = true;
    } );

    // Sleep to allow the waiter to run.
    using msec = std::chrono::duration<int, std::milli>;
    std::this_thread::sleep_for( msec( 100 ) );

    // Free the item and busy-wait until the waiter is finished.
    pool.free( item, m_deviceIndex, m_stream );
    while( !finished.load() )
    {
        std::this_thread::sleep_for( msec( 10 ) );
    }
    EXPECT_EQ( 0U, pool.size() );

    waiter.join();
    pool.shutDown();
}

// Test shutting down the pool while a thread is waiting.
TEST_F( TestBulkPinnedItemPool, TestShutDown )
{
    ItemPool pool( 1, m_arrayLength );

    // Allocate an item, but don't free it.
    pool.allocate();
    EXPECT_EQ( 1U, pool.size() );

    // Subsequent allocation in a separate thread will block until pool is shut down.
    std::atomic<bool> finished( false );
    std::thread       waiter( [this, &pool, &finished] {
        Item* item = pool.allocate();
        EXPECT_EQ( nullptr, item );
        finished = true;
    } );

    // Sleep to allow the waiter to run.
    using msec = std::chrono::duration<int, std::milli>;
    std::this_thread::sleep_for( msec( 100 ) );

    // Shut down the pool and busy-wait until the waiter is finished.
    pool.shutDown();
    while( !finished.load() )
    {
        std::this_thread::sleep_for( msec( 10 ) );
    }
    waiter.join();
}
