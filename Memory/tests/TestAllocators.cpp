// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/Allocators.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <vector>

using namespace otk;

class TestAllocators : public testing::Test
{
  public:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
    }
};

TEST_F( TestAllocators, HostAllocator )
{
    HostAllocator allocator;
    void*         ptr = allocator.allocate( 1024 );
    EXPECT_TRUE( ptr != nullptr );
    allocator.free( ptr );
}

TEST_F( TestAllocators, PinnedAllocator )
{
    PinnedAllocator allocator;
    void*           ptr = allocator.allocate( 1024 );
    EXPECT_TRUE( ptr != nullptr );
    allocator.free( ptr );
}

TEST_F( TestAllocators, DeviceAllocator )
{
    DeviceAllocator allocator;
    void*           ptr = allocator.allocate( 1024 );
    EXPECT_TRUE( ptr != nullptr );
    allocator.free( ptr );
}

TEST_F( TestAllocators, DeviceAsyncAllocator )
{
    DeviceAsyncAllocator allocator;

    CUstream stream;
    cuStreamCreate( &stream, 0U );

    void* ptr = allocator.allocate( 1024 );
    EXPECT_TRUE( ptr != nullptr );
    allocator.free( ptr );

    cuStreamDestroy( stream );
}

TEST_F( TestAllocators, TextureTileAllocator )
{
    TextureTileAllocator allocator;
    size_t               allocSize = TextureTileAllocator::getRecommendedAllocationSize();
    EXPECT_TRUE( allocSize > 0 );
    void* ptr = allocator.allocate( allocSize );
    EXPECT_TRUE( ptr != 0ull );
    allocator.free( ptr );
}

TEST_F( TestAllocators, DeviceAllocatorSpeed )
{
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );

    unsigned int       numBytes       = 1024;
    unsigned int       numAllocations = 10000;
    std::vector<void*> allocations( numAllocations, nullptr );

    DeviceAllocator allocator;

    for( unsigned int i = 0; i < numAllocations; ++i )
    {
        allocations[i] = allocator.allocate( numBytes );
    }

    for( unsigned int i = 0; i < numAllocations; ++i )
    {
        allocator.free( allocations[i] );
    }
}

TEST_F( TestAllocators, DeviceAsyncAllocatorSpeed )
{
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
    CUstream stream;
    OTK_ERROR_CHECK( cudaStreamCreate( &stream ) );

    unsigned int       numBytes       = 1024;
    unsigned int       numAllocations = 10000;
    std::vector<void*> allocations( numAllocations, nullptr );

    DeviceAsyncAllocator allocator;

    for( unsigned int i = 0; i < numAllocations; ++i )
    {
        allocations[i] = allocator.allocate( numBytes, stream );
    }

    for( unsigned int i = 0; i < numAllocations; ++i )
    {
        allocator.free( allocations[i], stream );
    }
}
