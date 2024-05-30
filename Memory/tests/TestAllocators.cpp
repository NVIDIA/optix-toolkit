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
