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

#include <stdio.h>
#include <OptiXToolkit/ShaderUtil/ISummedAreaTable.h>
#include <gtest/gtest.h>

const float EPS = 0.00001f;

class TestISummedAreaTable : public testing::Test
{
  public:
    void printTable( ISummedAreaTable& sat );
    void verifyTable( ISummedAreaTable& sat );
};

void TestISummedAreaTable::printTable( ISummedAreaTable& sat )
{
    for( int j = 0; j < sat.height; ++j )
    {
        for( int i = 0; i < sat.width; ++i )
        {
            printf( "%1.4f ", static_cast<float>( sat.val( i, j ) ) / static_cast<float>( 0xffffffffU ) );
        }
        printf( "\n" );
    }
}

void TestISummedAreaTable::verifyTable( ISummedAreaTable& sat )
{
    for( int j = 0; j < sat.height; ++j )
    {
        for( int i = 0; i < sat.width; ++i )
        {
            EXPECT_EQ( sat.val( i, j ), sat.tval( i, j ) );
        }
    }
}

TEST_F( TestISummedAreaTable, TestMakeTable )
{
    float pdf[20] = {1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1};
    ISummedAreaTable sat;
    allocHost( sat, 5, 4 );
    init( sat, pdf );
    //printTable( sat );
    freeHost( sat );
}

TEST_F( TestISummedAreaTable, TestMakeTableSpeed )
{
    int width = 8192;
    int height = 4096;
    float* pdf = (float*)malloc( width * height * sizeof(float) );
    for( int i=0; i<width*height; ++i ) 
        pdf[i]=1.0f;

    ISummedAreaTable sat;
    allocHost( sat, width, height );
    init( sat, pdf );
    printf( "%1.4f\n", static_cast<float>( sat.val( width-1, height-1 ) ) / static_cast<float>( 0xffffffffU ) );
    freeHost( sat );
    free(pdf);
}

TEST_F( TestISummedAreaTable, TestSearchUniform )
{
    int width = 256;
    int height = 128;
    float* pdf = (float*)malloc( width * height * sizeof(float) );
    for( int i=0; i<width*height; ++i ) 
        pdf[i]=1.0f;

    ISummedAreaTable sat;
    allocHost( sat, width, height );
    init( sat, pdf );

    const int gridSize = 11;
    for( int j=0; j<=gridSize; ++j )
    {
        for( int i=0; i<=gridSize; ++i )
        {
            float2 s = float2{float(i)/gridSize, float(j)/gridSize};
            float2 t = sampleRect( sat, 0, 0, width-1, height-1, s.x, s.y );
            EXPECT_NEAR( s.x, t.x, EPS );
            EXPECT_NEAR( s.y, t.y, EPS );
        }
    }

    freeHost( sat );
}

TEST_F( TestISummedAreaTable, TestSearchNonUniform )
{
    // Make table with one quarter filled
    int width = 10;
    int height = 10;
    float pdf[100] = {
        1,1,1,1,1,0,0,0,0,0,
        1,1,1,1,1,0,0,0,0,0,
        1,1,1,1,1,0,0,0,0,0,
        1,1,1,1,1,0,0,0,0,0,
        1,1,1,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    };

    ISummedAreaTable sat;
    allocHost( sat, width, height );
    init( sat, pdf );
    //printTable( sat );

    // Sample over full table
    const int gridSize = 10;
    for( int j=0; j<=gridSize; ++j )
    {
        for( int i=0; i<=gridSize; ++i )
        {
            float2 s = float2{float(i)/gridSize, float(j)/gridSize};
            float2 t = sampleRect( sat, 0, 0, width-1, height-1, s.x, s.y );
            EXPECT_NEAR( s.x/2.0f, t.x, EPS );
            EXPECT_NEAR( s.y/2.0f, t.y, EPS );
        }
    }

    // Sample over partial table
    for( int j=0; j<=gridSize; ++j )
    {
        for( int i=0; i<=gridSize; ++i )
        {
            float2 s = float2{float(i)/gridSize, float(j)/gridSize};
            float2 t = sampleRect( sat, 2, 3, 7, 8, s.x, s.y );
            EXPECT_NEAR( t.x, 0.2f + (0.5f-0.2f)*s.x, EPS );
            EXPECT_NEAR( t.y, 0.3f + (0.5f-0.3f)*s.y, EPS );
        }
    }

    freeHost( sat );
}

