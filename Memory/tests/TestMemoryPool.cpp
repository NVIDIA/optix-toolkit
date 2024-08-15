// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//


#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/BinnedSuballocator.h>
#include <OptiXToolkit/Memory/FixedSuballocator.h>
#include <OptiXToolkit/Memory/HeapSuballocator.h>
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>
#include <OptiXToolkit/Memory/MemoryPool.h>
#include <OptiXToolkit/Memory/RingSuballocator.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <vector>

using namespace otk;

class TestMemoryPool : public testing::Test
{
  public:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
    }
};

TEST_F( TestMemoryPool, TestNullAllocator )
{
    // With a null allocator, the pool keeps track address ranges given to it by calling track

    MemoryPool<NullAllocator, HeapSuballocator> pool( nullptr, new HeapSuballocator() );
    pool.track( 0, 1 << 20 );

    for( unsigned int i = 1; i <= 2000; i++ )
    {
        uint64_t        freeSpaceBefore = pool.currentFreeSpace();
        MemoryBlockDesc m               = pool.alloc( i, 1 );
        EXPECT_EQ( freeSpaceBefore, pool.currentFreeSpace() + m.size );
    }
}

TEST_F( TestMemoryPool, TestNullSuballocator )
{
    // With a null suballocator, the memory pool allocates and frees directly from the allocator

    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator* allocator = new HostAllocator();

    MemoryPool<HostAllocator, NullSuballocator> pool( allocator, nullptr, allocSize, maxSize );
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
    OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
    OTK_ERROR_CHECK( cudaStreamCreate( &stream ) );

    uint64_t arenaSize = 1 << 19;
    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator*    allocator    = new HostAllocator();
    RingSuballocator* suballocator = new RingSuballocator( arenaSize );

    MemoryPool<HostAllocator, RingSuballocator> pool( allocator, suballocator, allocSize, maxSize );
    MemoryBlockDesc m = pool.alloc( 1024, 1 );

    pool.freeAsync( m, stream );
}

TEST_F( TestMemoryPool, AllocFreeAsync )
{
    const unsigned int deviceIndex = 0;
    CUstream           stream;
    OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
    OTK_ERROR_CHECK( cudaStreamCreate( &stream ) );

    uint64_t arenaSize = 1 << 19;
    uint64_t allocSize = 1 << 20;
    uint64_t maxSize   = 10 * ( 1 << 20 );

    HostAllocator*    allocator    = new HostAllocator();
    RingSuballocator* suballocator = new RingSuballocator( arenaSize );

    MemoryPool<HostAllocator, RingSuballocator> pool( allocator, suballocator, allocSize, maxSize );
    std::deque<MemoryBlockDesc> blocks;

    for( int i = 0; i < 1000; ++i )
    {
        MemoryBlockDesc block = pool.alloc( 65536, 1 );
        ASSERT_TRUE( block.isGood() );  // It should always return a good block in this test
        blocks.push_back( block );
        if( blocks.size() > 30 )
        {
            pool.freeAsync( blocks.front(), stream );
            blocks.pop_front();
        }
    }
}

TEST_F( TestMemoryPool, TextureTile )
{
    const unsigned int deviceIndex = 0;
    OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
    TextureTileAllocator* allocator    = new TextureTileAllocator;
    HeapSuballocator*     suballocator = new HeapSuballocator;

    uint64_t allocationSize = TextureTileAllocator::getRecommendedAllocationSize();
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

TEST_F( TestMemoryPool, TestAllocItem )
{
    MemoryPool<HostAllocator, FixedSuballocator> pool( new FixedSuballocator( 32, 1 ) );
    uint64_t a = pool.allocItem();

    char* p = (char*)a;
    p[31]   = 'x';
    EXPECT_TRUE( p[31] == 'x' );

    pool.freeItem( a );
}

TEST_F( TestMemoryPool, TestAllocObject )
{
    MemoryPool<HostAllocator, HeapSuballocator> pool;

    int* i = pool.allocObject<int>();
    *i     = 1;
    EXPECT_TRUE( *i == 1 );
    pool.freeObject<int>( i );

    float4* f = pool.allocObject<float4>();
    f->z      = 1.0f;
    EXPECT_TRUE( f->z == 1.0f );
    pool.freeObject<float4>( f );
}

TEST_F( TestMemoryPool, TestAllocObjects )
{
    MemoryPool<HostAllocator, HeapSuballocator> pool;

    float4* f = pool.allocObjects<float4>( 1024 );
    f[1023].z = 1.0f;
    EXPECT_TRUE( f[1023].z == 1.0f );
    pool.freeObjects<float4>( f, 1024 );
}
