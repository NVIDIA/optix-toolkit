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
