// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPageLoaderImpl.h"

#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <cuda.h>
#include <cuda_runtime.h>

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
