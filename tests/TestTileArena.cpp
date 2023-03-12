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

#include "Memory/Buffers.h"
#include "Memory/TileArena.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class TestTileArena : public testing::Test
{
  public:
    void SetUp() override
    {
        size_t arenaSize = TileArena::getRecommendedSize();
        m_arena          = TileArena::create( arenaSize );
    }

    void TearDown() override { m_arena.destroy(); }

  protected:
    unsigned int m_deviceIndex = 0;
    TileArena    m_arena;
};

TEST_F( TestTileArena, CreateDestroy )
{
    EXPECT_EQ( 0U, m_arena.size() );
    EXPECT_GT( m_arena.capacity(), 0U );
}


TEST_F( TestTileArena, Allocate )
{
    size_t offset = m_arena.allocate( sizeof( TileBuffer ) );
    EXPECT_EQ( 0U, offset );
    EXPECT_EQ( sizeof( TileBuffer ), m_arena.size() );

    offset = m_arena.allocate( sizeof( TileBuffer ) );
    EXPECT_EQ( sizeof( TileBuffer ), offset );
    EXPECT_EQ( 2 * sizeof( TileBuffer ), m_arena.size() );
}
