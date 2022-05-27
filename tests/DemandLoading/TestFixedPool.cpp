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
