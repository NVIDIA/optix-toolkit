// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Memory/RingSuballocator.h>

#include <vector>

#include <gtest/gtest.h>

using namespace otk;

class TestRingSuballocator : public testing::Test
{
  public:
    RingSuballocator ringSuballocator;

    TestRingSuballocator()
        : ringSuballocator( 1 << 20 )
    {
    }
};

TEST_F( TestRingSuballocator, track )
{
    ringSuballocator.track( 0, 1024 );
    ringSuballocator.track( 2048, 1024 );
    ringSuballocator.track( 4096, 1024 );
    EXPECT_EQ( static_cast<uint64_t>( 3 * 1024 ), ringSuballocator.freeSpace() );
}

TEST_F( TestRingSuballocator, decrementAllocCount )
{
    ringSuballocator.track( 0, 1024 );
    ringSuballocator.track( 2048, 1024 );
    std::vector<MemoryBlockDesc> usedBlocks;

    const uint32_t numAllocs = 16;
    const uint32_t alignment = 32;
    for( uint32_t i = 0; i < numAllocs; ++i )
    {
        MemoryBlockDesc block = ringSuballocator.alloc( 100, alignment );
        uint64_t        ptr   = block.ptr;
        usedBlocks.push_back( block );
        EXPECT_TRUE( ptr % alignment == 0 );
    }
    EXPECT_EQ( static_cast<uint64_t>( 28 ), ringSuballocator.freeSpace() );

    for( uint32_t i = 0; i < usedBlocks.size(); ++i )
    {
        ringSuballocator.free( usedBlocks[i] );
    }
    EXPECT_EQ( static_cast<uint64_t>( 2 * 1024 ), ringSuballocator.freeSpace() );
}

TEST_F( TestRingSuballocator, freeAll )
{
    for( int i = 0; i < 100; ++i )
    {
        ringSuballocator.track( i * ( 1 << 20 ), 1 << 20 );
    }

    const uint32_t alignment = 1;
    uint32_t       numAllocs = 1 << 20;

    for( uint32_t i = 0; i < numAllocs; ++i )
    {
        MemoryBlockDesc block = ringSuballocator.alloc( 100, alignment );
        if( block.isBad() )
        {
            ringSuballocator.freeAll();
            block = ringSuballocator.alloc( 100, alignment );
        }
        EXPECT_TRUE( block.isGood() );
    }
}
