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

#include "Memory/Allocators.h"
#include "Memory/ItemPool.h"

#include <gtest/gtest.h>

#include <vector>

using namespace demandLoading;

class IntPool : public ItemPool<int, PinnedAllocator>
{
  public:
    IntPool()
        : ItemPool<int, PinnedAllocator>( PinnedAllocator() )
    {
    }
};


class TestItemPool : public testing::Test
{
};

TEST_F( TestItemPool, Unused )
{
    IntPool pool;
    EXPECT_EQ( 0U, pool.size() );
}

TEST_F( TestItemPool, AllocateAndFree )
{
    IntPool pool;
    int*    item = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );

    pool.free( item );
    EXPECT_EQ( 0U, pool.size() );
}

TEST_F( TestItemPool, ReuseFreedItem )
{
    // Verify that freed items are reused.  (This test is implementation specific.)
    IntPool pool;
    int*    item1 = pool.allocate();
    pool.free( item1 );
    int* item2 = pool.allocate();
    EXPECT_EQ( item1, item2 );
    pool.free( item2 );
}
