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

#include <OptiXToolkit/Memory/BinnedSuballocator.h>

#include <algorithm>
#include <map>
#include <vector>

#include <gtest/gtest.h>

using namespace otk;

class BinnedSuballocatorT : public BinnedSuballocator
{
  public:
    BinnedSuballocatorT( const std::vector<uint64_t>& itemSizes, const std::vector<uint64_t>& itemsPerChunk )
        : BinnedSuballocator( itemSizes, itemsPerChunk )
    {
    }
    std::vector<uint64_t>&          getItemSizes() { return m_itemSizes; }
    std::vector<FixedSuballocator>& getFixedSuballocators() { return m_fixedSuballocators; }
    HeapSuballocator&               getHeapSuballocator() { return m_heapSuballocator; };
};

class TestBinnedSuballocator : public testing::Test
{
};

TEST_F( TestBinnedSuballocator, track )
{
    BinnedSuballocatorT suballocator( {2, 4, 8, 16, 32}, {128, 64, 32, 32, 32} );
    suballocator.track( 0, 1024 );
    EXPECT_EQ( suballocator.freeSpace(), static_cast<uint64_t>( 1024 ) );
    EXPECT_EQ( suballocator.trackedSize(), static_cast<uint64_t>( 1024 ) );
}


TEST_F( TestBinnedSuballocator, allocMany )
{
    const uint64_t        numAllocations = 1 << 20;
    std::vector<uint64_t> sizes( {2, 4, 8, 16, 32} );
    BinnedSuballocatorT   suballocator( sizes, {8, 8, 8, 8, 8} );
    suballocator.track( 0, numAllocations * 128 );

    sizes.push_back( 256 );

    for( uint64_t i = 0; i < numAllocations; ++i )
    {
        uint64_t        allocSize = sizes[i % sizes.size()];
        MemoryBlockDesc memBlock  = suballocator.alloc( allocSize, 1 );
        EXPECT_TRUE( memBlock.ptr != BAD_ADDR );
        EXPECT_TRUE( memBlock.size == allocSize );
    }
}


TEST_F( TestBinnedSuballocator, allocFree )
{
    const uint64_t        numAllocations = 1 << 20;
    std::vector<uint64_t> sizes( {2, 4, 8, 16, 32} );
    std::vector<uint64_t> chunkItems( {128, 128, 128, 128, 128} );
    BinnedSuballocatorT   suballocator( sizes, chunkItems );
    suballocator.track( 0, numAllocations * 128 );

    sizes.push_back( 256 );

    std::vector<MemoryBlockDesc> blocks;

    uint64_t idx = 0;
    for( uint64_t i = 0; i < numAllocations; ++i )
    {
        uint64_t        allocSize = sizes[idx];
        MemoryBlockDesc memBlock  = suballocator.alloc( allocSize, 1 );
        blocks.push_back( memBlock );
        EXPECT_TRUE( memBlock.isGood() );
        EXPECT_TRUE( memBlock.size == allocSize );
        idx = ( idx >= sizes.size() - 1 ) ? 0 : idx + 1;
    }

    for( uint64_t i = 0; i < numAllocations; ++i )
    {
        suballocator.free( blocks[i] );
    }
    EXPECT_EQ( suballocator.freeSpace(), numAllocations * 128 );
}
