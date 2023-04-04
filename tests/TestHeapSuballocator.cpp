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

#include "Memory/HeapSuballocator.h"
#include <algorithm>
#include <map>
#include <vector>

#include <gtest/gtest.h>

using namespace demandLoading;

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
    const uint64_t               numAllocations = 1 << 20;
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

    std::random_shuffle( allocs.begin(), allocs.end() );  // worst case
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
