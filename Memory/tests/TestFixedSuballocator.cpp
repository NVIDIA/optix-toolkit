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
