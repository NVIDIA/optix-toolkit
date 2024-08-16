// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Memory/HeapSuballocator.h>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace otk;

class TestHeapSuballocator : public testing::Test
{
  public:
    void dumpMap( std::map<uint64_t, uint64_t>& m, const char* header )
    {
        std::cout << header;
        for( const auto& e : m )
            std::cout << "(" << e.first << "," << e.second << "), ";
        printf( "\n" );
    }

  protected:
    HeapSuballocator heapSuballocator;
};

TEST_F( TestHeapSuballocator, track )
{
    heapSuballocator.track( 0, 1024 );
    heapSuballocator.track( 2048, 65536 );

    const std::map<uint64_t, uint64_t>& beginMap = heapSuballocator.getBeginMap();
    EXPECT_EQ( static_cast<uint64_t>( 2 ), beginMap.size() );
    EXPECT_EQ( static_cast<uint64_t>( 1024 + 65536 ), heapSuballocator.trackedSize() );
    EXPECT_EQ( static_cast<uint64_t>( 1024 + 65536 ), heapSuballocator.freeSpace() );

    uint64_t val = beginMap.at( 0 );
    EXPECT_EQ( static_cast<uint64_t>( 1024 ), val );
    val = beginMap.at( 2048 );
    EXPECT_EQ( static_cast<uint64_t>( 65536 ), val );
}

TEST_F( TestHeapSuballocator, allocMany )
{
    const uint64_t numAllocations = 1 << 20;
    heapSuballocator.track( 0, numAllocations );

    for( uint64_t i = 0; i < numAllocations; ++i )
    {
        MemoryBlockDesc memBlock = heapSuballocator.alloc( 1, 1 );
        EXPECT_TRUE( memBlock.isGood() );
    }
    EXPECT_TRUE( heapSuballocator.freeSpace() == 0 );
}

TEST_F( TestHeapSuballocator, allocFree )
{
    const uint64_t               numAllocations = 1 << 18;
    std::vector<MemoryBlockDesc> allocs;
    heapSuballocator.track( 0, numAllocations );

    // Allocate a bunch of blocks
    for( uint64_t i = 0; i < numAllocations; ++i )
    {
        MemoryBlockDesc memBlock = heapSuballocator.alloc( 1, 1 );
        allocs.push_back( memBlock );
        EXPECT_TRUE( memBlock.isGood() );
    }
    EXPECT_TRUE( heapSuballocator.freeSpace() == 0 );

    std::random_device rng;
    std::mt19937       urng( rng() );
    std::shuffle( allocs.begin(), allocs.end(), urng );  // worst case
    //std::reverse( allocs.begin(), allocs.end() ); // best case

    const std::map<uint64_t, uint64_t>& beginMap = heapSuballocator.getBeginMap();
    uint64_t maxMapSize = 0;
    for( MemoryBlockDesc& memBlock : allocs )
    {
        heapSuballocator.free( memBlock );
        maxMapSize = std::max( beginMap.size(), maxMapSize );
    }
    EXPECT_TRUE( heapSuballocator.freeSpace() == heapSuballocator.trackedSize() );
}

TEST_F( TestHeapSuballocator, untrack )
{
    heapSuballocator.track( 0, 1024 );
    heapSuballocator.track( 2048, 1024 );

    heapSuballocator.untrack( 0, 1024 );
    EXPECT_EQ( heapSuballocator.trackedSize(), 1024ULL );
}
