//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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


#include <cuda.h>
#include <vector>

#include "WhiteBlackTileCheck.h"

#include <gtest/gtest.h>

using namespace demandLoading;
using namespace imageSource;
using namespace otk;

class TestWhiteBlackTileCheck : public testing::Test
{
};

TEST_F( TestWhiteBlackTileCheck, FloatTiles )
{
    std::vector<float> ftile( TILE_SIZE_IN_BYTES / sizeof(float), 1.0f );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ftile.data(), CU_AD_FORMAT_FLOAT, 1 ), F_1 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ftile.data(), CU_AD_FORMAT_FLOAT, 2 ), F2_11 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ftile.data(), CU_AD_FORMAT_FLOAT, 4 ), F4_1111 );

    std::fill( ftile.begin(), ftile.end(), 0.0f );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ftile.data(), CU_AD_FORMAT_FLOAT, 1 ), F_0 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ftile.data(), CU_AD_FORMAT_FLOAT, 2 ), F2_00 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ftile.data(), CU_AD_FORMAT_FLOAT, 4 ), F4_0000 );

    ftile[0] = 0.5f;
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ftile.data(), CU_AD_FORMAT_FLOAT, 1 ), WB_NONE );

    std::vector<float4> f4tile( TILE_SIZE_IN_BYTES / sizeof(float4), float4{1.0f, 1.0f, 1.0f, 0.0f} );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)f4tile.data(), CU_AD_FORMAT_FLOAT, 4 ), F4_1110 );
}

TEST_F( TestWhiteBlackTileCheck, HalfTiles )
{
    std::vector<half> htile( TILE_SIZE_IN_BYTES / sizeof(half), (half)1.0f );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)htile.data(), CU_AD_FORMAT_HALF, 1 ), H_1 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)htile.data(), CU_AD_FORMAT_HALF, 2 ), H2_11 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)htile.data(), CU_AD_FORMAT_HALF, 4 ), H4_1111 );

    std::fill( htile.begin(), htile.end(), (half)0.0f );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)htile.data(), CU_AD_FORMAT_HALF, 1 ), H_0 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)htile.data(), CU_AD_FORMAT_HALF, 2 ), H2_00 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)htile.data(), CU_AD_FORMAT_HALF, 4 ), H4_0000 );

    htile[25] = (half)0.5f;
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)htile.data(), CU_AD_FORMAT_HALF, 1 ), WB_NONE );

    std::vector<half4> h4tile( TILE_SIZE_IN_BYTES / sizeof(half4), half4{1.0f, 1.0f, 1.0f, 0.0f} );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)h4tile.data(), CU_AD_FORMAT_HALF, 4 ), H4_1110 );
}

TEST_F( TestWhiteBlackTileCheck, UcharTiles )
{
    std::vector<uchar> ubtile( TILE_SIZE_IN_BYTES / sizeof(uchar), 255 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ubtile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 1 ), UB_1 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ubtile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 2 ), UB2_11 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ubtile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 4 ), UB4_1111 );

    std::fill( ubtile.begin(), ubtile.end(), 0 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ubtile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 1 ), UB_0 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ubtile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 2 ), UB2_00 );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ubtile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 4 ), UB4_0000 );

    ubtile[25] = 10;
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ubtile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 1 ), WB_NONE );

    std::vector<uchar4> ub4tile( TILE_SIZE_IN_BYTES / sizeof(uchar4), uchar4{0,0,0,255} );
    EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)ub4tile.data(), CU_AD_FORMAT_UNSIGNED_INT8, 4 ), UB4_0001 );
}

TEST_F( TestWhiteBlackTileCheck, SpeedTest )
{
    std::vector<float> tile( TILE_SIZE_IN_BYTES / sizeof(float), 1.0f );
    for( int i=0; i<1000; ++i )
        EXPECT_EQ( classifyTileAsWhiteOrBlack( (char*)tile.data(), CU_AD_FORMAT_FLOAT, 1 ), F_1 );
}