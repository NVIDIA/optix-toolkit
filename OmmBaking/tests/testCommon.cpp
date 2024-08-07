//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "testCommon.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

void TestCommon::SetUp()
{
    // Initialize CUDA runtime
    OTK_ERROR_CHECK( cudaFree( 0 ) );

    // Create optix context

    OptixDeviceContextOptions optixOptions = {};

    OTK_ERROR_CHECK( optixInit() );

    CUcontext cuCtx = nullptr;  // zero means take the current context
    OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &optixOptions, &optixContext ) );
}

void TestCommon::TearDown()
{
    OTK_ERROR_CHECK( optixDeviceContextDestroy( optixContext ) );
}

cuOmmBaking::Result TestCommon::saveImageToFile( std::string imageNamePrefix, const std::vector<uchar3>& image, uint32_t width, uint32_t height )
{
    m_imageNamePrefix = imageNamePrefix;

    try
    {
#ifdef GENERATE_GOLD_IMAGES
        std::string imageName = m_imageNamePrefix + "_gold.ppm";
#else
        std::string imageName = m_imageNamePrefix + ".ppm";
#endif

        ImagePPM ppm( (const void*)image.data(), width, height, IMAGE_PIXEL_FORMAT_UCHAR3 );
        ppm.writePPM( imageName, false );
    }
    catch( ... )
    {
        EXPECT_TRUE( false );
        return cuOmmBaking::Result::ERROR_INTERNAL;
    }

    return cuOmmBaking::Result::SUCCESS;
}

void TestCommon::compareImage()
{
#ifdef GENERATE_GOLD_IMAGES
    EXPECT_TRUE( !"Generating gold image, no test result" );
    return;
#endif
    float tolerance = 2.0f / 255;
    int   numErrors;
    float avgError;
    float maxError;

    std::string imageName     = m_imageNamePrefix + ".ppm";
    std::string goldImageName = TEST_OMM_BAKING_GOLD_DIR + m_imageNamePrefix + "_gold.ppm";

    try
    {
        ImagePPM::compare( imageName, goldImageName, tolerance, numErrors, avgError, maxError );
    }
    catch( ... )
    {
        ASSERT_TRUE( !"Failed to compare with gold image" );
    }

    EXPECT_GT( 32, numErrors );
}
