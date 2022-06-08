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
