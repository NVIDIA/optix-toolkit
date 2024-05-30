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
#include <OptiXToolkit/ShaderUtil/AliasTable.h>
#include <gtest/gtest.h>

class TestAliasTable : public testing::Test
{
  public:
    void printAliasTable( AliasTable& at );
    void verifyAliasTable( AliasTable& at, float* pdf, float maxDiff );
};

void TestAliasTable::printAliasTable( AliasTable& at )
{
    for( int i = 0; i < at.size; ++i )
    {
        AliasRecord r = at.table[i];
        printf("{%0.3f,%d}, ", r.prob, r.alias);
    }
    printf("\n");
}

void TestAliasTable::verifyAliasTable( AliasTable& at, float* pdf, float maxDiff )
{
    std::vector<float> atPdf( at.size, 0.0f );
    float ave = 1.0f / at.size;

    // Make sure probabilities and aliases are within proper range
    for( int i = 0; i < at.size; ++i )
    {
        AliasRecord ar = at.table[i];
        EXPECT_TRUE( ar.alias >= 0 && ar.alias < at.size );
        EXPECT_TRUE( ar.prob >= 0.0f && ar.prob <= 1.0f );
    }

    // Make pdf defined by alias table.
    for( int i = 0; i < at.size; ++i )
    {
        atPdf[i] += at.table[i].prob * ave;
        atPdf[at.table[i].alias] += (1.0f - at.table[i].prob) * ave;
    }

    // Make sure the probabilities are the same.
    for( int i = 0; i < at.size; ++i )
    {
        EXPECT_NEAR( atPdf[i], pdf[i], maxDiff );
    }
}

TEST_F( TestAliasTable, TestMakeTable )
{
    float pdf[5] = {0.1f, 0.1f, 0.3f, 0.5f, 0.0f};
    float pdfCopy[5];
    AliasTable at;
    allocAliasTableHost( at, 5 );

    memcpy( pdfCopy, pdf, 5 * sizeof(float) );
    makeAliasTable( at, pdfCopy );
    verifyAliasTable( at, pdf, 0.0001f );

    freeAliasTableHost( at );
}

TEST_F( TestAliasTable, testTableEdges )
{
    std::vector<float> pdf( 25, 1.0f / 25.0f );
    std::vector<float> pdfCopy = pdf;
    AliasTable at;
    allocAliasTableHost( at, 25 );
    makeAliasTable( at, pdf.data() );

    freeAliasTableHost( at );
}

