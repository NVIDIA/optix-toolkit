//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "SourceDir.h"  // generated from SourceDir.h.in

#include <OptiXToolkit/Util/EXRInputFile.h>

#include <gtest/gtest.h>
#include <half.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <vector>

using namespace otk;

class TestEXRInputFile : public testing::Test
{
  public:
    void testRead( const std::string& input, const std::string& output )
    {
        EXRInputFile file;
        file.open( input );
        std::vector<half> pixels( 4 * sizeof( half ) * file.getWidth() * file.getHeight() );
        file.read( pixels.data(), pixels.size() );

        // Convert to float and write PNG for manual validation.
        std::vector<float> floats( pixels.begin(), pixels.end() );
        int                stride = 4 * sizeof( half ) * file.getWidth();
        stbi_write_png( output.c_str(), file.getWidth(), file.getHeight(), 4, pixels.data(), stride );
    }
};

TEST_F( TestEXRInputFile, TestOpenError )
{
    EXPECT_THROW( EXRInputFile().open( "no-such-file.exr" ), std::exception );
}

TEST_F( TestEXRInputFile, TestOpen )
{
    EXRInputFile file;
    file.open( getSourceDir() + "/Textures/TiledMipMappedHalf.exr" );
    EXPECT_NE( 0, file.getWidth() );
    EXPECT_NE( 0, file.getHeight() );
}

TEST_F( TestEXRInputFile, TestReadHalf )
{
    testRead( getSourceDir() + "/Textures/TiledMipMappedHalf.exr", "TiledMipMappedHalf.png" );
}

TEST_F( TestEXRInputFile, TestReadFloat )
{
    testRead( getSourceDir() + "/Textures/TiledMipMappedFloat.exr", "TiledMipMappedFloat.png" );
}
