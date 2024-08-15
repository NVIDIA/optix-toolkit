// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Memory/FixedSuballocator.h>

#include <algorithm>
#include <map>
#include <vector>

#include <gtest/gtest.h>

using namespace otk;

class TestFixedSuballocator : public testing::Test
{
};

TEST_F( TestFixedSuballocator, checkAlignment )
{
    FixedSuballocator suballocator1( 128, 512 );
    EXPECT_EQ( static_cast<uint64_t>( 512 ), suballocator1.alignment() );
    EXPECT_EQ( static_cast<uint64_t>( 512 ), suballocator1.itemSize() );

    FixedSuballocator suballocator2( 512, 100 );
    EXPECT_EQ( static_cast<uint64_t>( 100 ), suballocator2.alignment() );
    EXPECT_EQ( static_cast<uint64_t>( 600 ), suballocator2.itemSize() );
}

TEST_F( TestFixedSuballocator, allocMany )
{
    const uint64_t numAllocations = 1 << 20;
    const uint64_t allocSize      = 128;
    const uint64_t alignment      = 128;

    FixedSuballocator suballocator( allocSize, alignment );
    suballocator.track( 0, numAllocations * allocSize );
    EXPECT_EQ( numAllocations * allocSize, suballocator.trackedSize() );
    EXPECT_EQ( numAllocations * allocSize, suballocator.freeSpace() );

    for( uint64_t i = 0; i < numAllocations; ++i )
    {
        MemoryBlockDesc memBlock = suballocator.alloc();
        EXPECT_TRUE( !memBlock.isBad() );
    }
    EXPECT_TRUE( suballocator.freeSpace() == 0 );
}

TEST_F( TestFixedSuballocator, allocFree )
{
    const uint64_t numAllocations = 1 << 20;
    const uint64_t allocSize      = 1;
    const uint64_t alignment      = 1;

    FixedSuballocator            suballocator( allocSize, alignment );
    std::vector<MemoryBlockDesc> allocs;
    suballocator.track( 0, numAllocations );

    // Allocate a bunch of blocks
    for( uint64_t i = 0; i < numAllocations; ++i )
    {
        MemoryBlockDesc memBlock = suballocator.alloc();
        allocs.push_back( memBlock );
        EXPECT_TRUE( memBlock.isGood() );
    }
    EXPECT_TRUE( suballocator.freeSpace() == 0 );

    for( MemoryBlockDesc& memBlock : allocs )
    {
        suballocator.free( memBlock );
    }
    EXPECT_TRUE( suballocator.freeSpace() == suballocator.trackedSize() );
}
