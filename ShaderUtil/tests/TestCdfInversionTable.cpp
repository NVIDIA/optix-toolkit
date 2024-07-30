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
#include <OptiXToolkit/ShaderUtil/CdfInversionTable.h>
#include <gtest/gtest.h>

class TestCdfInversionTable : public testing::Test
{
  public:
    void printInversionTable( CdfInversionTable& cit );
    const float EPS = 0.0001f;
};

void TestCdfInversionTable::printInversionTable( CdfInversionTable& cit )
{
    printf("CDF\n");
    for( int j = 0; j < cit.height; ++j )
    {
        printf( "%1.3f | ", cit.cdfMarginal[j] );
        for( int i = 0; i < cit.width; ++i )
        {
            printf( "%1.3f ", cit.cdfRows[j*cit.width + i] );
        }
        printf("\n");
    }
    printf("\n");

    printf("INV-CDF\n");
    for( int j = 0; j < cit.height; ++j )
    {
        printf( "%2d  | ", cit.invCdfMarginal[j] );
        for( int i = 0; i < cit.width; ++i )
        {
            printf( "%2d ", cit.invCdfRows[j*cit.width + i] );
        }
        printf("\n");
    }
    printf("\n");
}

TEST_F( TestCdfInversionTable, TestUniform )
{
    const int w = 5;
    const int h = 5;
    std::vector<float> pdf( w * h, 1.0f);

    CdfInversionTable cit;
    allocCdfInversionTableHost( cit, w, h );
    memcpy( cit.cdfRows, pdf.data(), w * h * sizeof(float) );
    invertPdf2D( cit );
    invertCdf2D( cit );

    // bin search sampling
    float2 xy = sampleCdfBinSearch( cit, float2{0.5f, 0.5f} );
    EXPECT_NEAR( xy.x, 0.5f, EPS );
    EXPECT_NEAR( xy.y, 0.5f, EPS );

    xy = sampleCdfBinSearch( cit, float2{0.25f, 0.75f} );
    EXPECT_NEAR( xy.x, 0.25f, EPS );
    EXPECT_NEAR( xy.y, 0.75f, EPS );

    xy = sampleCdfBinSearch( cit, float2{0.0f, 0.0f} );
    EXPECT_NEAR( xy.x, 0.0f, EPS );
    EXPECT_NEAR( xy.y, 0.0f, EPS );

    xy = sampleCdfBinSearch( cit, float2{1.0f, 1.0f} );
    EXPECT_NEAR( xy.x, 1.0f, EPS );
    EXPECT_NEAR( xy.y, 1.0f, EPS );

    // linear search sampling
    xy = sampleCdfLinSearch( cit, float2{0.5f, 0.5f} );
    EXPECT_NEAR( xy.x, 0.5f, EPS );
    EXPECT_NEAR( xy.y, 0.5f, EPS );

    xy = sampleCdfLinSearch( cit, float2{0.25f, 0.75f} );
    EXPECT_NEAR( xy.x, 0.25f, EPS );
    EXPECT_NEAR( xy.y, 0.75f, EPS );

    xy = sampleCdfLinSearch( cit, float2{0.0f, 0.0f} );
    EXPECT_NEAR( xy.x, 0.0f, EPS );
    EXPECT_NEAR( xy.y, 0.0f, EPS );

    xy = sampleCdfLinSearch( cit, float2{1.0f, 1.0f} );
    EXPECT_NEAR( xy.x, 1.0f, EPS );
    EXPECT_NEAR( xy.y, 1.0f, EPS );
    
    freeCdfInversionTableHost( cit );
}

TEST_F( TestCdfInversionTable, TestContinuous )
{
    const int w = 5;
    const int h = 5;
    std::vector<float> pdf( w * h, 1.0f);

    CdfInversionTable cit;
    allocCdfInversionTableHost( cit, w, h );
    memcpy( cit.cdfRows, pdf.data(), w * h * sizeof(float) );
    invertPdf2D( cit );
    invertCdf2D( cit );

    float2 xy = sampleCdfDirectLookup( cit, float2{0.5f, 0.5f} );
    EXPECT_NEAR( xy.x, 0.5f, EPS );
    EXPECT_NEAR( xy.y, 0.5f, EPS );
    
    xy = sampleCdfDirectLookup( cit, float2{0.75f, 0.25f} );
    EXPECT_NEAR( xy.x, 0.75f, EPS );
    EXPECT_NEAR( xy.y, 0.25f, EPS );

    xy = sampleCdfDirectLookup( cit, float2{0.0f, 0.0f} );
    EXPECT_NEAR( xy.x, 0.0f, EPS );
    EXPECT_NEAR( xy.y, 0.0f, EPS );

    xy = sampleCdfDirectLookup( cit, float2{1.0f, 1.0f} );
    EXPECT_NEAR( xy.x, 1.0f, EPS );
    EXPECT_NEAR( xy.y, 1.0f, EPS );

    freeCdfInversionTableHost( cit );
}

void fillExamplePdf( std::vector<float>& pdf, int w, int h )
{
    pdf.resize( w * h, 0.0f );
    for( int j = 0; j < h; ++j )
    {
        float y = ( j + 0.5f ) / h;
        for( int i = 0; i < w; ++i )
        {
            float x = ( i + 0.5f ) / w;
            float d2 = x * x + y * y;
            pdf[j * w + i] = ( d2 < 0.25f ) ? 1.0f : 0.1f;
        }
    }
}

TEST_F( TestCdfInversionTable, LinearAndBinarySearch )
{
    unsigned int w = 256;
    unsigned int h = 256;
    std::vector<float> pdf;
    fillExamplePdf( pdf, w, h );

    CdfInversionTable cit;
    allocCdfInversionTableHost( cit, w, h );
    memcpy( cit.cdfRows, pdf.data(), w * h * sizeof(float) );
    invertPdf2D( cit );
    invertCdf2D( cit );

    // Compare binary and linear search methods
    const int s = 16;
    for( int j = 0; j < s; ++j )
    {
        for( int i = 0; i < s; ++i )
        {
            float x = (i + ( ( j * 7 ) % s + 0.5f ) / s ) / s;
            float y = (j + ( ( i * 11 ) % s + 0.5f ) / s ) / s;
            float2 p = sampleCdfBinSearch( cit, float2{x, y} );
            float2 q = sampleCdfLinSearch( cit, float2{x, y} );
            EXPECT_NEAR( p.x, q.x, EPS );
            EXPECT_NEAR( p.y, q.y, EPS );
        }
    }

    freeCdfInversionTableHost( cit );
}
