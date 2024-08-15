// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>
#include <stdio.h>
#include <OptiXToolkit/ShaderUtil/PdfTable.h>
#include <gtest/gtest.h>
#include <vector_types.h>

class TestPdfTable : public testing::Test
{
  public:
    void printPdfTable( float* table, int width, int height );
};

void TestPdfTable::printPdfTable( float* table, int width, int height )
{
    printf("PDF\n");
    for( int j = 0; j < height; ++j )
    {
        for( int i = 0; i < width; ++i )
        {
            printf( "%1.3f ", table[j * width + i] );
        }
        printf("\n");
    }
    printf("\n");
}

TEST_F( TestPdfTable, TestAngleModes )
{
    int w = 5;
    int h = 5;
    std::vector<float4> emap(w*h, float4{1,1,1,0});
    std::vector<float> pdf(w*h);
    float aveBrightness = 0.0f;

    makePdfTable<float4>( pdf.data(), emap.data(), &aveBrightness, w, h, pbLUMINANCE, paLATLONG );
    EXPECT_EQ( pdf[0], pdf[1] ); // rows have same value
    EXPECT_NEAR( pdf[(1)*w], pdf[(h-2)*w], 0.00001f ); // opposite side rows have same value
    EXPECT_TRUE( pdf[0] != 1.0f ); // corner value is not 1
    EXPECT_EQ( pdf[(h/2)*w + (w/2)], 1.0f ); // center value is 1

    makePdfTable<float4>( pdf.data(), emap.data(), &aveBrightness, w, h, pbLUMINANCE, paCUBEMAP );
    EXPECT_EQ( pdf[w/2], pdf[(h/2)*w] ); // top center, and center left are the same
    EXPECT_TRUE( pdf[0] != 1.0f ); // corner value is not 1
    EXPECT_EQ( pdf[(h/2)*w + (w/2)], 1.0f ); // center value is 1

    makePdfTable<float4>( pdf.data(), emap.data(), &aveBrightness, w, h, pbLUMINANCE, paNONE );
    EXPECT_EQ( pdf[0], 1.0f ); // corner value is 1
    EXPECT_EQ( pdf[(h/2)*w + (w/2)], 1.0f ); // center value is 1 
}

TEST_F( TestPdfTable, TestBrightnessModes )
{
    int w = 5;
    int h = 5;
    std::vector<float4> emap(w*h, float4{1,2,3,0});
    std::vector<float> pdf(w*h);
    float aveBrightness = 0.0f;

    makePdfTable<float4>( pdf.data(), emap.data(), &aveBrightness, w, h, pbLUMINANCE, paNONE );
    float lum = LUMINANCE( emap[0] );
    EXPECT_EQ( pdf[0], lum );

    makePdfTable<float4>( pdf.data(), emap.data(), &aveBrightness, w, h, pbRGBSUM, paNONE );
    EXPECT_EQ( pdf[0], 1+2+3 );
}
