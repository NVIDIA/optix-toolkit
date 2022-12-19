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

#include <algorithm>
#include <map>
#include <vector>

#include "Memory/Allocators.h"
#include "Memory/BinnedSuballocator.h"
#include "Memory/FixedSuballocator.h"
#include "Memory/HeapSuballocator.h"
#include "Memory/MemoryBlockDesc.h"
#include "Memory/MemoryPool.h"
#include "Memory/RingSuballocator.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class TestMemoryPool : public testing::Test
{
  public:
    void SetUp() override
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( 0 ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );
    }
};

TEST_F( TestMemoryPool, AddressPool )
{
    // An AddressPool is a MemoryPool with no allocator. It keeps track of
    // ranges of addresses in a hypothetical memory space

    AddressPool<HeapSuballocator> pool( new HeapSuballocator() );
    pool.track( 0, 1 << 20 );

    for( unsigned int i = 1; i <= 2000; i++ )
    {
        uint64_t        freeSpaceBefore = pool.currentFreeSpace();
        MemoryBlockDesc m               = pool.alloc( i, 1 );
        EXPECT_EQ( freeSpaceBefore, pool.currentFreeSpace() + m.size );
    }
}

TEST_F( TestMemoryPool, NullSuballocator )
{
    // With a null suballocator, the memory pool allocates and frees directly from the allocator

    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator* allocator = new HostAllocator();

    MemoryPool<HostAllocator, HeapSuballocator> pool( allocator, nullptr, allocSize, maxSize );
    MemoryBlockDesc m = pool.alloc( 1024, 1 );

    EXPECT_TRUE( m.ptr != 0 && m.size == 1024 );

    pool.free( m );
}

TEST_F( TestMemoryPool, RingSuballocator )
{
    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator*    allocator    = new HostAllocator();
    RingSuballocator* suballocator = new RingSuballocator( allocSize );

    MemoryPool<HostAllocator, RingSuballocator> pool( allocator, suballocator, allocSize, maxSize );
    MemoryBlockDesc m = pool.alloc( 1024, 1 );

    EXPECT_TRUE( m.ptr != 0 );

    pool.free( m );

    EXPECT_EQ( pool.allocatableSpace(), maxSize );
}

TEST_F( TestMemoryPool, FixedSuballocator )
{
    uint64_t allocSize = 1 << 18;
    uint64_t maxSize   = 1 << 20;
    uint64_t itemSize  = 1024;

    HostAllocator*     allocator    = new HostAllocator();
    FixedSuballocator* suballocator = new FixedSuballocator( itemSize, itemSize );

    MemoryPool<HostAllocator, FixedSuballocator> pool( allocator, suballocator, allocSize, maxSize );

    std::vector<uint64_t> items;
    unsigned int          count = 0;
    while( count < 100000 )
    {
        uint64_t m = pool.allocItem();
        if( m == BAD_ADDR )
            break;
        items.push_back( m );
        ++count;
    }

    EXPECT_EQ( pool.allocatableSpace(), 0ull );

    for( uint64_t m : items )
    {
        pool.freeItem( m );
    }

    EXPECT_EQ( pool.allocatableSpace(), items.size() * itemSize );
}

TEST_F( TestMemoryPool, HeapSuballocator )
{
    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator*    allocator    = new HostAllocator();
    HeapSuballocator* suballocator = new HeapSuballocator();

    MemoryPool<HostAllocator, HeapSuballocator> pool( allocator, suballocator, allocSize, maxSize );
    MemoryBlockDesc m = pool.alloc( 1024, 101 );
    EXPECT_TRUE( ( m.ptr % 101 ) == 0 );
    EXPECT_TRUE( m.size == 1024 );
    pool.free( m );
}

TEST_F( TestMemoryPool, BinnedSuballocator )
{
    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator*      allocator    = new HostAllocator();
    BinnedSuballocator* suballocator = new BinnedSuballocator( {16, 32, 64}, {16, 16, 16} );

    MemoryPool<HostAllocator, BinnedSuballocator> pool( allocator, suballocator, allocSize, maxSize );
    std::vector<MemoryBlockDesc> blocks;
    blocks.push_back( pool.alloc( 16 ) );
    blocks.push_back( pool.alloc( 64 ) );
    blocks.push_back( pool.alloc( 32 ) );
    blocks.push_back( pool.alloc( 128, 128 ) );

    EXPECT_EQ( uint64_t( 16 ), blocks[0].size );
    EXPECT_EQ( uint64_t( 64 ), blocks[1].size );
    EXPECT_EQ( uint64_t( 32 ), blocks[2].size );
    EXPECT_EQ( uint64_t( 128 ), blocks[3].size );
}

TEST_F( TestMemoryPool, RingSuballocatorFreeAsync )
{
    const unsigned int deviceIndex = 0;
    CUstream           stream;
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
    DEMAND_CUDA_CHECK( cudaStreamCreate( &stream ) );

    uint64_t arenaSize = 1 << 19;
    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator*    allocator    = new HostAllocator();
    RingSuballocator* suballocator = new RingSuballocator( arenaSize );

    MemoryPool<HostAllocator, RingSuballocator> pool( allocator, suballocator, allocSize, maxSize );
    MemoryBlockDesc m = pool.alloc( 1024, 1 );

    pool.freeAsync( m, deviceIndex, stream );
}

TEST_F( TestMemoryPool, AllocFreeAsync )
{
    const unsigned int deviceIndex = 0;
    CUstream           stream;
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
    DEMAND_CUDA_CHECK( cudaStreamCreate( &stream ) );

    uint64_t arenaSize = 1 << 19;
    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator*    allocator    = new HostAllocator();
    RingSuballocator* suballocator = new RingSuballocator( arenaSize );

    MemoryPool<HostAllocator, RingSuballocator> pool( allocator, suballocator, allocSize, maxSize );
    std::deque<MemoryBlockDesc> blocks;

    for( int i = 0; i < 1000; ++i )
    {
        blocks.push_back( pool.alloc( 65536, 1 ) );
        if( blocks.size() > 30 )
        {
            pool.freeAsync( blocks.front(), deviceIndex, stream );
            blocks.pop_front();
        }
    }
}

TEST_F( TestMemoryPool, TextureTile )
{
    const unsigned int deviceIndex = 0;
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
    TextureTileAllocator* allocator    = new TextureTileAllocator( deviceIndex );
    HeapSuballocator*     suballocator = new HeapSuballocator();

    uint64_t allocationSize = TextureTileAllocator::getRecommendedAllocationSize( deviceIndex );
    uint64_t maxSize        = 1 << 25;

    MemoryPool<TextureTileAllocator, HeapSuballocator> pool( allocator, suballocator, allocationSize, maxSize );

    std::vector<TileBlockHandle> tileBlocks;
    for( unsigned int i = 0; i < 100; ++i )
    {
        // Do mostly single tiles, with some 4x in size
        unsigned int size = ( ( i % 10 ) == 0 ) ? TILE_SIZE_IN_BYTES * 4 : TILE_SIZE_IN_BYTES;
        tileBlocks.push_back( pool.allocTextureTiles( size ) );
        EXPECT_TRUE( static_cast<uint64_t>( tileBlocks.back().handle ) != BAD_ADDR );
        EXPECT_TRUE( tileBlocks.back().block.numTiles == ( size / TILE_SIZE_IN_BYTES ) );
    }

    for( TileBlockHandle tbh : tileBlocks )
    {
        uint64_t beforeFreeSpace = pool.currentFreeSpace();
        pool.freeTextureTiles( tbh.block );
        EXPECT_TRUE( beforeFreeSpace + tbh.block.numTiles * TILE_SIZE_IN_BYTES == pool.currentFreeSpace() );
    }
}
