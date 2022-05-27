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

#include "Memory/Buffers.h"
#include "Memory/TileArena.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class TestTileArena : public testing::Test
{
  public:
    void SetUp() override
    {
        size_t arenaSize = TileArena::getRecommendedSize( m_deviceIndex );
        m_arena          = TileArena::create( m_deviceIndex, arenaSize );
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
