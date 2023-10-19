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

#include <cuda.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include "DemandPageLoaderImpl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Check whether sparse textures are supported on at least one device.
bool sparseTexturesSupported()
{
    cudaFree(nullptr);
    int numDevices;
    OTK_ERROR_CHECK( cuDeviceGetCount( &numDevices ) );
    for( int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
    {
        if( demandLoading::deviceSupportsSparseTextures( deviceIndex ) )
            return true;
    }
    return false;
}

int main( int argc, char** argv )
{
    testing::InitGoogleMock( &argc, argv );
    if( !testing::GTEST_FLAG(list_tests) && !sparseTexturesSupported() )
    {
        std::string filter = ::testing::GTEST_FLAG( filter );
        if( filter.find( "-" ) == filter.npos )
            filter += '-';  // start negative filters
        else
            filter += ':';  // extend negative filters
        filter +=
            "DeferredImageLoadingTest.deferredTileIsLoadedAgain"
            ":TestDemandLoader.TestTextureVariants"
            ":TestDemandTexture.TestFillTile"
            ":TestDemandTexture.TestReadMipTail"
            ":TestDemandTexture.TestFillMipTail"
            ":TestDemandTexture.TestSparseNonMipmappedTexture";
        testing::GTEST_FLAG( filter ) = filter;
    }
    return RUN_ALL_TESTS();
}
