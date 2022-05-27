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

#include "Util/MutexArray.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using msec = std::chrono::duration<int, std::milli>;

using namespace demandLoading;

class TestMutexArray : public testing::Test
{
};

TEST_F( TestMutexArray, ConstructDestroy )
{
    MutexArray mutex( 2 );
}

TEST_F( TestMutexArray, LockUnlock )
{
    MutexArray mutex( 1 );
    mutex.lock( 0 );
    mutex.unlock( 0 );

    mutex.lock( 0 );
    mutex.unlock( 0 );
}

TEST_F( TestMutexArray, LockUnlockMulti )
{
    MutexArray mutex( 2 );
    mutex.lock( 0 );
    mutex.lock( 1 );
    mutex.unlock( 0 );
    mutex.unlock( 1 );

    mutex.lock( 0 );
    mutex.lock( 1 );
    mutex.unlock( 0 );
    mutex.unlock( 1 );
}

TEST_F( TestMutexArray, MutexArrayLock )
{
    MutexArray mutex( 1 );
    {
        MutexArrayLock( &mutex, 0 );
    }
    MutexArrayLock( &mutex, 0 );
}

TEST_F( TestMutexArray, MutexArrayLockMulti )
{
    MutexArray mutex( 2 );
    {
        MutexArrayLock( &mutex, 0 );
        MutexArrayLock( &mutex, 1 );
    }
    MutexArrayLock( &mutex, 0 );
    MutexArrayLock( &mutex, 1 );
}

TEST_F( TestMutexArray, ExclusionSingle )
{
    MutexArray mutex( 1 );

    // Acquire mutex.
    mutex.lock( 0 );

    // Start a separate thread that blocks until the mutex is released.
    std::atomic<bool> started( false );
    std::atomic<bool> finished( false );
    std::thread       waiter( [&mutex, &started, &finished] {
        started  = true;
        mutex.lock( 0 );
        finished = true;
        mutex.unlock( 0 );
    } );

    // Wait for the waiter to start
    while( !started.load() )
    {
        std::this_thread::sleep_for( msec( 10 ) );
    }
    
    // Verify that the waiter is blocked.
    EXPECT_FALSE( finished.load() );

    // Release the mutex and busy-wait until the waiter is finished.
    mutex.unlock( 0 );
    while( !finished.load() )
    {
        std::this_thread::sleep_for( msec( 10 ) );
    }
    waiter.join();
}


TEST_F( TestMutexArray, ExclusionMulti )
{
    // The work is divided into buckets, which will be controlled by a MutexArray.
    const unsigned int numBuckets = 4;
    const int initialBucketValue = 10;
    std::vector<int>   buckets( numBuckets, initialBucketValue );
    MutexArray         mutex( numBuckets );

    // We assign several worker threads per bucket.
    unsigned int             numThreads = numBuckets * 8;
    std::vector<std::thread> threads;
    threads.reserve( numThreads );
    unsigned int whichBucket = 0;
    for( unsigned int i = 0; i < numThreads; ++i )
    {
        int* bucket = &buckets.at(whichBucket);
        whichBucket = ( whichBucket + 1 ) % numBuckets;
        threads.emplace_back( [bucket, whichBucket, initialBucketValue, &mutex] {
            while( true )
            {
                // Acquire mutex.
                mutex.lock( whichBucket );

                // Sanity check bucket value.
                EXPECT_GE( *bucket, 0 );
                EXPECT_LE( *bucket, initialBucketValue );

                // Exit thread if bucket value has dropped to zero.
                if( *bucket == 0 )
                {
                    mutex.unlock( whichBucket );
                    return;
                }

                // Sleep for a bit, and check that the bucket value doesn't change.
                int value = *bucket;
                std::this_thread::sleep_for( msec( 10 ) );
                EXPECT_EQ( value, *bucket );

                // Decrement the bucket value
                --*bucket;

                // Release the mutex and sleep for a bit to allow another thread to access the bucket.
                mutex.unlock( whichBucket );
                std::this_thread::sleep_for( msec( 10 ) );
            }
        } );
    }

    // Wait for the threads to complete.
    for( std::thread& thread : threads )
        thread.join();

    // Verify that the buckets are all zero.
    for( int bucket : buckets )
    {
        EXPECT_EQ( 0, bucket );
    }
}
