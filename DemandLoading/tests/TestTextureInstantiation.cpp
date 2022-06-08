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

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define CHK( status ) ASSERT_TRUE( ( status ) == CUDA_SUCCESS )

//------------------------------------------------------------------------------
// Test Fixtures
namespace {

class TestTextureInstantiation : public testing::Test
{
  public:
    TestTextureInstantiation() {}

    virtual ~TestTextureInstantiation() noexcept {}

    void SetUp() override
    {
        // Initialize CUDA.
        cudaFree( nullptr );
    }
};

}  // end namespace

//------------------------------------------------------------------------------
// Test functions

void mapTile( CUmipmappedArray& array, CUmemGenericAllocationHandle& tileHandle, CUstream& stream )
{
    // Map tile backing storage into array
    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = array;

    mapInfo.subresourceType               = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level = 0;

    mapInfo.subresource.sparseLevel.offsetX = 0;
    mapInfo.subresource.sparseLevel.offsetY = 0;

    mapInfo.subresource.sparseLevel.extentWidth  = 64;  // float4 size
    mapInfo.subresource.sparseLevel.extentHeight = 64;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = tileHandle;
    mapInfo.offset              = 0;
    mapInfo.deviceBitMask       = 1U;  // device 0

    CHK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void mapMipTail( CUmipmappedArray& array, CUmemGenericAllocationHandle& tileHandle, CUstream& stream )
{
    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = array;

    mapInfo.subresourceType            = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = 1 << 16;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = tileHandle;
    mapInfo.offset              = 0;
    mapInfo.deviceBitMask       = 1U;  // device 0

    CHK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void testMipmapTextures( unsigned int width, unsigned int height, unsigned int numTextures, unsigned int flags )
{
    int numLevels = static_cast<int>( 1.f + std::log2( static_cast<float>( std::max( width, height ) ) ) );

    std::vector<CUmipmappedArray> mipmapArrays( numTextures );
    std::vector<CUtexObject>      textures( numTextures );

    CUmemAllocationProp props{};
    props.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location         = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
    props.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;

    CUstream stream;
    CUresult result = cuStreamCreate( &stream, 0U );
    CHK( result );

    CUmemGenericAllocationHandle tileHandle;
    CHK( cuMemCreate( &tileHandle, 1 << 21, &props, 0 ) );

    size_t freeBefore, freeAfter, totalMem;
    cuMemGetInfo( &freeBefore, &totalMem );

    for( unsigned int i = 0; i < numTextures; ++i )
    {
        // Create CUDA array
        CUDA_ARRAY3D_DESCRIPTOR ad{};
        ad.Width       = width;
        ad.Height      = height;
        ad.Format      = CU_AD_FORMAT_FLOAT;
        ad.NumChannels = 4;
        ad.Flags       = flags;
        CHK( cuMipmappedArrayCreate( &mipmapArrays[i], &ad, numLevels ) );

        // Create CUDA texture descriptor
        CUDA_TEXTURE_DESC td{};
        td.addressMode[0]      = CU_TR_ADDRESS_MODE_WRAP;
        td.addressMode[1]      = CU_TR_ADDRESS_MODE_WRAP;
        td.filterMode          = CU_TR_FILTER_MODE_LINEAR;
        td.flags               = CU_TRSF_NORMALIZED_COORDINATES;
        td.maxAnisotropy       = 16;
        td.mipmapFilterMode    = CU_TR_FILTER_MODE_LINEAR;
        td.maxMipmapLevelClamp = static_cast<float>( numLevels - 1 );
        td.minMipmapLevelClamp = 0.f;

        // Create texture object.
        CUDA_RESOURCE_DESC rd{};
        rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        rd.res.mipmap.hMipmappedArray = mipmapArrays[i];
        CHK( cuTexObjectCreate( &textures[i], &rd, &td, 0 ) );

        // Code to map a tile or mip tail
        if( flags )
        {
            mapMipTail( mipmapArrays[i], tileHandle, stream );
            //mapTile( mipmapArrays[i], tileHandle, stream );
        }
    }

    cuMemGetInfo( &freeAfter, &totalMem );

// #define PRINTSTATS 1
#ifdef PRINTSTATS
    // Constants for float4 textures
    const double tileWidth = 64.0;
    const double tileHeight = 64.0;
    const double textureBytesPerTexel = 16.0;
    const double bytesPerTileRef = 8.0; 

    double totalBytes = freeBefore - freeAfter;
    double bytesPerTexture = totalBytes / numTextures;
    double texelsPerTexture = width * height * 4.0 / 3.0;
    double tilesPerTexture = (width / tileWidth) * (height / tileHeight) * 4.0 / 3.0;
    double bytesPerTexel = bytesPerTexture / texelsPerTexture;
    double bytesPerTile = bytesPerTexture / tilesPerTexture;

    double expectedBytesPerTexture = textureBytesPerTexel * texelsPerTexture;
    double expectedTileRefBytesPerTexture = bytesPerTileRef * tilesPerTexture;

    printf("MemUsage per texture                 %1.1f\n", bytesPerTexture);
    printf("MemUsage per texel                   %1.1f\n", bytesPerTexel);
    printf("MemUsage per tile                    %1.1f\n", bytesPerTile);
    printf("Memory overhead per standard texture %1.1f\n", bytesPerTexture - expectedBytesPerTexture );
    printf("Memory overhead per sparse texture   %1.1f\n", bytesPerTexture - expectedTileRefBytesPerTexture );
    printf("\n");
#endif

    for( unsigned int i = 0; i < numTextures; ++i )
    {
        // Free resources
        CHK( cuTexObjectDestroy( textures[i] ) );
        CHK( cuMipmappedArrayDestroy( mipmapArrays[i] ) );
    }

    CHK( cuStreamDestroy( stream ) );
    CHK( cuMemRelease( tileHandle ) );
}

//------------------------------------------------------------------------------
// Timing Tests for different numbers and sizes of textures. 
// Small Tests

TEST_F( TestTextureInstantiation, t32_50000_dense_1GB )
{
    testMipmapTextures( 32, 32, 50000, 0 );
}

TEST_F( TestTextureInstantiation, t64_20000_sparse_1_6GB )
{
    testMipmapTextures( 64, 64, 20000, CUDA_ARRAY3D_SPARSE );
}

TEST_F( TestTextureInstantiation, t512_10000_sparse_52GB )
{
    testMipmapTextures( 512, 512, 10000, CUDA_ARRAY3D_SPARSE );
}

// TODO: this reports a spurious CUDA_ERROR_OUT_OF_MEMORY error in
// cudaMallocAsync under Windows in single Ampere configurations only.
TEST_F( TestTextureInstantiation, DISABLED_t4096_1000_sparse_333GB )
{
    testMipmapTextures( 4096, 4096, 1000, CUDA_ARRAY3D_SPARSE );
}

//------------------------------------------------------------------------------
// Large tests

// #define LARGE_TESTS
#ifdef LARGE_TESTS

// 3892 ms
TEST_F( TestTextureInstantiation, t16_500000_dense_2_5GB )
{
    testMipmapTextures( 16, 16, 500000, 0 );
}

// 5910 ms
TEST_F( TestTextureInstantiation, t64_100000_sparse_8GB )
{
    testMipmapTextures( 64, 64, 100000, CUDA_ARRAY3D_SPARSE );
}

// 7412 ms
TEST_F( TestTextureInstantiation, t512_50000_sparse_260GB )
{
    testMipmapTextures( 512, 512, 50000, CUDA_ARRAY3D_SPARSE );
}

// 11939 ms
TEST_F( TestTextureInstantiation, t4096_10000_sparse_3333GB )
{
    testMipmapTextures( 4096, 4096, 10000, CUDA_ARRAY3D_SPARSE );
}

// 60557 ms
TEST_F( TestTextureInstantiation, t2048_40000_sparse_3333GB )
{
    testMipmapTextures( 2048, 2048, 40000, CUDA_ARRAY3D_SPARSE );
}
#endif
