// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "SourceDir.h"  // generated from SourceDir.h.in

#include <OptiXToolkit/Util/EXRInputFile.h>

#include <gtest/gtest.h>
#include <half.h>

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
    EXPECT_NE( 0U, file.getWidth() );
    EXPECT_NE( 0U, file.getHeight() );
}

TEST_F( TestEXRInputFile, TestReadHalf )
{
    testRead( getSourceDir() + "/Textures/TiledMipMappedHalf.exr", "TiledMipMappedHalf.png" );
}

TEST_F( TestEXRInputFile, TestReadFloat )
{
    testRead( getSourceDir() + "/Textures/TiledMipMappedFloat.exr", "TiledMipMappedFloat.png" );
}
