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

#include "Memory/FixedPool.h"

#include <gtest/gtest.h>

#include <vector>

using namespace demandLoading;

class FixedIntPool : public FixedPool<int>
{
  public:
    FixedIntPool( size_t capacity )
        : m_items( capacity )
    {
        FixedPool<int>::init( m_items.data(), m_items.size() );
    }

  private:
    std::vector<int> m_items;
};


class TestFixedPool : public testing::Test
{
  public:
    void SetUp() override { m_pool.reset( new FixedIntPool( m_maxItems ) ); }

    void TearDown() override { m_pool.reset( nullptr ); }

  protected:
    const size_t m_maxItems = 1;

    std::unique_ptr<FixedPool<int>> m_pool;
};

TEST_F( TestFixedPool, Unused )
{
    EXPECT_EQ( 0U, m_pool->size() );
    EXPECT_EQ( m_maxItems, m_pool->capacity() );
}

TEST_F( TestFixedPool, AllocateAndFree )
{
    int* item = m_pool->allocate();
    EXPECT_NE( nullptr, item );
    EXPECT_EQ( 1U, m_pool->size() );

    m_pool->free( item );
    EXPECT_EQ( 0U, m_pool->size() );
}

TEST_F( TestFixedPool, CapacityExceeded )
{
    int* item = m_pool->allocate();
    EXPECT_NE( nullptr, item );
    EXPECT_EQ( 1U, m_pool->size() );
    EXPECT_EQ( 1U, m_pool->capacity() );

    int* item2 = m_pool->allocate();
    EXPECT_EQ( nullptr, item2 );
}
