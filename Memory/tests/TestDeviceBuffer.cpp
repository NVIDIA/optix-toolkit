// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// Validate that header stands alone.
#include <OptiXToolkit/Memory/DeviceBuffer.h>

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <gtest/gtest.h>

#include <cuda.h>

#include <utility>

using namespace testing;

class DeviceBufferTest : public Test
{
public:
    ~DeviceBufferTest() override = default;

protected:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cuInit( 0 ) );
        OTK_ERROR_CHECK( cuCtxCreate( &m_context, 0, 0 ) );
    }
    void TearDown() override
    {
        // rip down CUDA driver context
        OTK_ERROR_CHECK( cuCtxDestroy( m_context ) );
    }

    CUcontext m_context{};
};

TEST_F( DeviceBufferTest, constructFromSize )
{
    otk::DeviceBuffer buffer( 512U );

    ASSERT_EQ( 512u, buffer.size() );
    ASSERT_EQ( 512u, buffer.capacity() );
    ASSERT_NE( CUdeviceptr{}, static_cast<CUdeviceptr>( buffer ) );
}

TEST_F( DeviceBufferTest, constructFromPointerAndSize )
{
    const std::size_t size{ 512U };
    CUdeviceptr       storage{};
    OTK_ERROR_CHECK( cuMemAlloc( &storage, size ) );
    otk::DeviceBuffer buffer( storage, size );

    ASSERT_EQ( size, buffer.size() );
    ASSERT_EQ( size, buffer.capacity() );
    ASSERT_EQ( storage, static_cast<CUdeviceptr>( buffer ) );
}

TEST_F( DeviceBufferTest, moveAssignable )
{
    otk::DeviceBuffer rhs( 512U );
    const CUdeviceptr address = rhs;
    otk::DeviceBuffer lhs( 128U );

    lhs = std::move( rhs );

    ASSERT_EQ( 0U, rhs.size() );
    ASSERT_EQ( 0U, rhs.capacity() );
    ASSERT_EQ( 512U, lhs.size() );
    ASSERT_EQ( 512U, lhs.capacity() );
    ASSERT_EQ( address, static_cast<CUdeviceptr>( lhs ) );
}

TEST_F( DeviceBufferTest, moveConstructable )
{
    otk::DeviceBuffer rhs( 512U );
    const CUdeviceptr address = rhs;

    otk::DeviceBuffer lhs( std::move( rhs ) );

    ASSERT_EQ( 0U, rhs.size() );
    ASSERT_EQ( 0U, rhs.capacity() );
    ASSERT_EQ( 512U, lhs.size() );
    ASSERT_EQ( 512U, lhs.capacity() );
    ASSERT_EQ( address, static_cast<CUdeviceptr>( lhs ) );
}

TEST_F( DeviceBufferTest, attach )
{
    CUdeviceptr storage{};
    std::size_t size{ 128U };
    OTK_ERROR_CHECK( cuMemAlloc( &storage, size ) );
    otk::DeviceBuffer buff;

    buff.attach( storage, size );

    ASSERT_EQ( storage, static_cast<CUdeviceptr>( buff ) );
    ASSERT_EQ( size, buff.size() );
    ASSERT_EQ( size, buff.capacity() );
}

TEST_F( DeviceBufferTest, detach )
{
    otk::DeviceBuffer buff{ 128U };
    const CUdeviceptr data = buff;

    CUdeviceptr storage = buff.detach();

    ASSERT_EQ( data, storage );
    ASSERT_EQ( 0U, buff.size() );
    ASSERT_EQ( 0U, buff.capacity() );

    OTK_ERROR_CHECK( cuMemFree( storage ) );
}
