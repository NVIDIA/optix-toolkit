//
//  Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include "Memory/EventPool.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class TestEventPool : public testing::Test
{
public:
    void SetUp() override
    {
        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        cudaFree( nullptr );
    }

    unsigned int m_deviceIndex = 0;
};

TEST_F( TestEventPool, TestEmpty )
{
    EventPool pool( m_deviceIndex );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_EQ( 0U, pool.capacity() );
}

TEST_F( TestEventPool, TestUnusedCapacity )
{
    EventPool pool( m_deviceIndex, 1 );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );
}

TEST_F( TestEventPool, TestWithinCapacity )
{
    EventPool pool( m_deviceIndex, 1 );
    CUevent   event = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );

    pool.free( event );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );

    pool.allocate();
    EXPECT_EQ( 1U, pool.size() );
    EXPECT_EQ( 1U, pool.capacity() );
}

TEST_F( TestEventPool, TestGrowth )
{
    EventPool pool( m_deviceIndex );
    CUevent   event = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );
    EXPECT_GE( 1U, pool.capacity() );

    pool.free( event );
    EXPECT_EQ( 0U, pool.size() );
    EXPECT_GE( 1U, pool.capacity() );

    pool.allocate();
    pool.allocate();
    EXPECT_EQ( 2U, pool.size() );
    EXPECT_GE( 2U, pool.capacity() );
}
