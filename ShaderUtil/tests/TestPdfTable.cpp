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
