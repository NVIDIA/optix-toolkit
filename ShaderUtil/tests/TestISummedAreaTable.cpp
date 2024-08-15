// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>
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
    printf("SAT Table:\n");
    for( int j = 0; j < sat.height; ++j )
    {
        for( int i = 0; i < sat.width; ++i )
        {
            printf( "%1.4f ", static_cast<float>( sat.val( i, j ) ) / static_cast<float>( 0xffffffffU ) );
        }
        printf( "\n" );
    }

    printf("\nColumn Sums:\n");
    for( int j = 0; j < sat.height; ++j )
    {
        for( int i = 0; i < sat.width; ++i )
        {
            float val = sat.column(i)[j] / static_cast<float>( sat.column(i)[sat.height-1] );
            printf( "%1.4f ", val );
        }
        printf( "\n" );
    }
    printf("\n");
}

void TestISummedAreaTable::verifyTable( ISummedAreaTable& sat )
{
    // Check SAT table for consistency
    for( int j = 1; j < sat.height; ++j )
    {
        for( int i = 1; i < sat.width; ++i )
        {
            EXPECT_TRUE( sat.val( i, j ) > sat.val( i-1, j ) );
            EXPECT_TRUE( sat.val( i, j ) > sat.val( i, j-1 ) );
        }
    }

    // Check columns for consistency
    for( int i = 0; i < sat.width; ++i )
    {
        for( int j = 1; j < sat.height; ++j )
        {
            EXPECT_TRUE( sat.column(i)[j] > sat.column(i)[j-1] );
        }
    }
}

TEST_F( TestISummedAreaTable, TestMakeTable )
{
    float pdf[20] = {1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1};
    ISummedAreaTable sat;
    allocISummedAreaTableHost( sat, 5, 4 );
    initISummedAreaTable( sat, pdf );
    //printTable( sat );
    verifyTable( sat );
    freeISummedAreaTableHost( sat );
}

TEST_F( TestISummedAreaTable, TestMakeTableSpeed )
{
    int width = 8192;
    int height = 4096;
    float* pdf = (float*)malloc( width * height * sizeof(float) );
    for( int i=0; i<width*height; ++i ) 
        pdf[i]=1.0f;

    ISummedAreaTable sat;
    allocISummedAreaTableHost( sat, width, height );
    initISummedAreaTable( sat, pdf );
    printf( "%1.4f\n", static_cast<float>( sat.val( width-1, height-1 ) ) / static_cast<float>( 0xffffffffU ) );
    freeISummedAreaTableHost( sat );
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
    allocISummedAreaTableHost( sat, width, height );
    initISummedAreaTable( sat, pdf );
    verifyTable( sat );

    const int gridSize = 11;
    for( int j=0; j<=gridSize; ++j )
    {
        for( int i=0; i<=gridSize; ++i )
        {
            float2 s = float2{float(i)/gridSize, float(j)/gridSize};
            float2 t = sampleRect( sat, 0, 0, width-1, height-1, s );
            EXPECT_NEAR( s.x, t.x, EPS );
            EXPECT_NEAR( s.y, t.y, EPS );
        }
    }

    freeISummedAreaTableHost( sat );
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
    allocISummedAreaTableHost( sat, width, height );
    initISummedAreaTable( sat, pdf );
    //printTable( sat );

    // Sample over full table
    const int gridSize = 10;
    for( int j=0; j<=gridSize; ++j )
    {
        for( int i=0; i<=gridSize; ++i )
        {
            float2 s = float2{float(i)/gridSize, float(j)/gridSize};
            float2 t = sampleRect( sat, 0, 0, width-1, height-1, s );
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
            float2 t = sampleRect( sat, 2, 3, 7, 8, s );
            EXPECT_NEAR( t.x, 0.2f + (0.5f-0.2f)*s.x, EPS );
            EXPECT_NEAR( t.y, 0.3f + (0.5f-0.3f)*s.y, EPS );
        }
    }

    freeISummedAreaTableHost( sat );
}
