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

#include <OptiXToolkit/ImageSource/DeviceConstantImage.h>

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define CHK( status ) ASSERT_TRUE( static_cast<unsigned int>( status ) == static_cast<unsigned int>( CUDA_SUCCESS ) ) 

//------------------------------------------------------------------------------
// Test Fixtures
namespace {

class TestTextureFill : public testing::Test
{
  public:
    TestTextureFill()
        : testImage( 4096, 4096, colors )
    {
    }
    virtual ~TestTextureFill() noexcept {}

    void SetUp() override
    {
        // Initialize CUDA.
        CHK( cudaFree( nullptr ) );
    }

    static const unsigned int pixelSize  = 16;
    static const unsigned int tileWidth  = 64;
    static const unsigned int tileHeight = 64;
    std::vector<float4> colors = {{1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.5f, 0.0f, 0.0f}, {1.0f, 0.0f, .0f, 0.0f}, {1.0f, 1.0f, 0.0f, 0.0f}};
    imageSource::DeviceConstantImage testImage; 

    void fillMapInfo( CUmipmappedArray&             array,
                      unsigned int                  mipLevel,
                      unsigned int                  offsetX,
                      unsigned int                  offsetY,
                      unsigned int                  width,
                      unsigned int                  height,
                      CUmemGenericAllocationHandle& tilePool,
                      unsigned int                  tilePoolOffset,
                      CUarrayMapInfo&               mapInfo );

    void mapMultipleTiles( CUarrayMapInfo* mapInfos, unsigned int numTiles, CUstream& stream );

    void mapTile( CUmipmappedArray&             array,
                  unsigned int                  mipLevel,
                  unsigned int                  offsetX,
                  unsigned int                  offsetY,
                  CUmemGenericAllocationHandle& tilePool,
                  unsigned int                  tilePoolOffset,
                  CUstream&                     stream );

    void fillTile( CUmipmappedArray& array,
                   unsigned int      mipLevel,
                   unsigned int      tileX,
                   unsigned int      tileY,
                   const char*       tileData,
                   CUmemorytype      tileMemoryType,
                   CUstream          stream );

    void doTextureFillTest( int                       numStreams,
                            imageSource::ImageSource& imageSource,
                            bool                      readImage,
                            bool                      asyncMalloc,
                            bool                      mapTiles,
                            bool                      transferTexture,
                            bool                      batchMode,
                            unsigned int              mapBatchSize = 1 );

    void doMapTilesTest( imageSource::ImageSource& imageSource, unsigned int tileBlockWidth, unsigned int tileBlockHeight, bool unmap );
};

}  // end namespace

//------------------------------------------------------------------------------
// Test functions

void TestTextureFill::fillMapInfo( CUmipmappedArray&             array,
                                   unsigned int                  mipLevel,
                                   unsigned int                  offsetX,
                                   unsigned int                  offsetY,
                                   unsigned int                  width,
                                   unsigned int                  height,
                                   CUmemGenericAllocationHandle& tilePool,
                                   unsigned int                  tilePoolOffset,
                                   CUarrayMapInfo&               mapInfo )
{
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = array;

    mapInfo.subresourceType                      = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level        = mipLevel;
    mapInfo.subresource.sparseLevel.offsetX      = offsetX;
    mapInfo.subresource.sparseLevel.offsetY      = offsetY;
    mapInfo.subresource.sparseLevel.extentWidth  = width;
    mapInfo.subresource.sparseLevel.extentHeight = height;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = tilePool;
    mapInfo.offset              = tilePoolOffset;
    mapInfo.deviceBitMask       = 1U;  // device 0
}

void TestTextureFill::mapMultipleTiles( CUarrayMapInfo* mapInfos, unsigned int numTiles, CUstream& stream )
{
    CHK( cuMemMapArrayAsync( mapInfos, numTiles, stream ) );
}

void TestTextureFill::mapTile( CUmipmappedArray&             array,
                               unsigned int                  mipLevel,
                               unsigned int                  offsetX,
                               unsigned int                  offsetY,
                               CUmemGenericAllocationHandle& tilePool,
                               unsigned int                  tilePoolOffset,
                               CUstream&                     stream )
{
    CUarrayMapInfo mapInfo{};
    fillMapInfo( array, mipLevel, offsetX, offsetY, tileWidth, tileHeight, tilePool, tilePoolOffset, mapInfo );
    CHK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void TestTextureFill::fillTile( CUmipmappedArray& array,
                                unsigned int      mipLevel,
                                unsigned int      tileX,
                                unsigned int      tileY,
                                const char*       tileData,
                                CUmemorytype      tileMemoryType,
                                CUstream          stream )
{
    CUarray mipLevelArray{};
    CHK( cuMipmappedArrayGetLevel( &mipLevelArray, array, mipLevel ) );

    CUDA_MEMCPY2D copyArgs{};
    copyArgs.srcMemoryType = tileMemoryType;
    copyArgs.srcHost       = ( tileMemoryType == CU_MEMORYTYPE_HOST ) ? tileData : nullptr;
    copyArgs.srcDevice     = ( tileMemoryType == CU_MEMORYTYPE_DEVICE ) ? reinterpret_cast<CUdeviceptr>( tileData ) : 0;
    copyArgs.srcPitch      = tileWidth * pixelSize;

    copyArgs.dstXInBytes = tileX * tileWidth * pixelSize;
    copyArgs.dstY        = tileY * tileHeight;

    copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyArgs.dstArray      = mipLevelArray;

    copyArgs.WidthInBytes = tileWidth * pixelSize;
    copyArgs.Height       = tileHeight;

    CHK( cuMemcpy2DAsync( &copyArgs, stream ) );
}

void TestTextureFill::doTextureFillTest( int                       numStreams,
                                         imageSource::ImageSource& imageSource,
                                         bool                      readImage,
                                         bool                      asyncMalloc,
                                         bool                      mapTiles,
                                         bool                      transferTexture,
                                         bool                      batchMode,
                                         unsigned int              mapBatchSize )
{
    // Get image properties
    imageSource::TextureInfo texInfo;
    imageSource.open( &texInfo );

    // Create a CUDA stream
    std::vector<CUstream> streams(numStreams);
    for( int i=0; i<numStreams; ++i )
    {
        CHK( cuStreamCreate( &streams[i], 0U ) );
    }

    // Create temporary ring buffer for transfers
    const unsigned int ringBufferSize = 1 << 24; //texInfo.width * texInfo.height * pixelSize;
    CUdeviceptr ringBuffer = 0L;
    if( !asyncMalloc )
    {
        CHK( cuMemAlloc( &ringBuffer, ringBufferSize) );
    }

    // Create tile pool to hold all tiles on mip level 0
    CUmemAllocationProp props{};
    props.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location         = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
    props.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;

    CUmemGenericAllocationHandle tilePool = 0;
    CHK( cuMemCreate( &tilePool, texInfo.width * texInfo.height * 16, &props, 0 ) );

    // Create sparse array for texture
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = texInfo.width;
    ad.Height      = texInfo.height;
    ad.Format      = texInfo.format;;
    ad.NumChannels = texInfo.numChannels;
    ad.Flags       = CUDA_ARRAY3D_SPARSE;

    CUmipmappedArray mipmapArray;
    CHK( cuMipmappedArrayCreate( &mipmapArray, &ad, texInfo.numMipLevels ) );

    // Create CUDA texture descriptor
    CUDA_TEXTURE_DESC td{};
    td.addressMode[0]      = CU_TR_ADDRESS_MODE_WRAP;
    td.addressMode[1]      = CU_TR_ADDRESS_MODE_WRAP;
    td.filterMode          = CU_TR_FILTER_MODE_LINEAR;
    td.flags               = CU_TRSF_NORMALIZED_COORDINATES;
    td.maxAnisotropy       = 16;
    td.mipmapFilterMode    = CU_TR_FILTER_MODE_LINEAR;
    td.maxMipmapLevelClamp = static_cast<float>( texInfo.numMipLevels - 1 );
    td.minMipmapLevelClamp = 0.f;

    // Create texture object.
    CUtexObject       texture;
    CUDA_RESOURCE_DESC rd{};
    rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    rd.res.mipmap.hMipmappedArray = mipmapArray;
    CHK( cuTexObjectCreate( &texture, &rd, &td, 0 ) );

    unsigned int streamId = 0;
    cudaEvent_t event;
    CHK( cudaEventCreate( &event ) );

    // Batch mode: Reading/Mapping/Copying separated
    if( batchMode )
    {
        std::vector<CUarrayMapInfo> mapInfos;
        
        unsigned int tileSize = tileWidth * tileHeight * pixelSize;
        streamId = 0;
        for( unsigned int y = 0; y < texInfo.height; y += tileHeight )
        {
            // Allocate tile buffer
            char* tempTileBuffs = (char*)ringBuffer; 
            if( asyncMalloc )
            {
#if CUDA_VERSION >= 11020
                CHK( cudaMallocAsync( &tempTileBuffs, texInfo.width * tileHeight * pixelSize, streams[streamId] ) );
#else
                CHK( cudaMalloc( &tempTileBuffs, texInfo.width * tileHeight * pixelSize ) );
#endif
            }

            for( unsigned int x = 0; x < texInfo.width; x += tileWidth )
            {
                // Read the image tile into the tile buffer
                char* tempTileBuff = &tempTileBuffs[tileSize*(x/tileWidth)];
                if( readImage )
                    imageSource.readTile( tempTileBuff, 0, x / tileWidth, y / tileHeight, tileWidth, tileHeight, streams[streamId] );
            }

            for( unsigned int x = 0; x < texInfo.width; x += tileWidth )
            {
                // Map the tile in the sparse texture
                unsigned int tilePoolOffset = y * texInfo.width * pixelSize + x * tileWidth * pixelSize;
                if( mapTiles )
                {
                    mapInfos.emplace_back();
                    fillMapInfo( mipmapArray, 0, x, y, tileWidth, tileHeight, tilePool, tilePoolOffset, mapInfos.back() );
                    if( mapInfos.size() >= mapBatchSize )
                    {
                        mapMultipleTiles( &mapInfos.front(), static_cast<unsigned int>( mapInfos.size() ), streams[streamId] );
                        mapInfos.clear();
                    }
                }
            }
            if( mapInfos.size() > 0 )
            {
                mapMultipleTiles( &mapInfos.front(), static_cast<unsigned int>( mapInfos.size() ), streams[streamId] );
                mapInfos.clear();
            }

            for( unsigned int x = 0; x < texInfo.width; x += tileWidth )
            {
                // Copy tile data from tile buffer to sparse texture
                char* tempTileBuff = &tempTileBuffs[tileSize*(x/tileWidth)];
                if( transferTexture )
                    fillTile( mipmapArray, 0, x / tileWidth, y / tileHeight, tempTileBuff, CU_MEMORYTYPE_DEVICE, streams[streamId] );
            }

            if( asyncMalloc )
            {
#if CUDA_VERSION >= 11020
                CHK( cudaFreeAsync( tempTileBuffs, streams[streamId] ) );
#else
                CHK( cudaFree( tempTileBuffs ) );
#endif
            }

            streamId = (streamId + 1) % numStreams;
        }
    }


    // Regular mode: Reading/Mapping/Copying together
    else 
    {
        unsigned int ringBufferPos = 0;
        for( unsigned int y = 0; y < texInfo.height; y += tileHeight )
        {
            for( unsigned int x = 0; x < texInfo.width; x += tileWidth )
            {
                // Map the tile in the sparse texture
                unsigned int tilePoolOffset = y * texInfo.width * pixelSize + x * tileWidth * pixelSize;
                if( mapTiles )
                    mapTile( mipmapArray, 0, x, y, tilePool, tilePoolOffset, streams[streamId] );

                // Allocate tile buffer
                char* tempTileBuff = nullptr; 
                if( asyncMalloc )
#if CUDA_VERSION >= 11020
                    CHK( cudaMallocAsync( &tempTileBuff, tileWidth * tileHeight * pixelSize, streams[streamId] ) );
#else
                    CHK( cudaMalloc( &tempTileBuff, tileWidth * tileHeight * pixelSize ) );
#endif
                else
                    tempTileBuff = reinterpret_cast<char*>( ringBuffer + ringBufferPos );

                // Read the image tile into the tile buffer
                if( readImage )
                    imageSource.readTile( tempTileBuff, 0, x / tileWidth, y / tileHeight, tileWidth, tileHeight, streams[streamId] );
                
                // Copy tile data from tile buffer to sparse texture
                if( transferTexture )
                    fillTile( mipmapArray, 0, x / tileWidth, y / tileHeight, tempTileBuff, CU_MEMORYTYPE_DEVICE, streams[streamId] );

                // Free tile
                if( asyncMalloc )
#if CUDA_VERSION >= 11020
                    CHK( cudaFreeAsync( tempTileBuff, streams[streamId] ) );
#else
                    CHK( cudaFree( tempTileBuff ) );
#endif
                else 
                    ringBufferPos = ( ringBufferPos + tileWidth * tileHeight * pixelSize ) % ringBufferSize;

                streamId = (streamId + 1) % numStreams;
            }
        }
    }

    for( int i=0; i<numStreams; ++i )
    {
        CHK( cudaStreamSynchronize( streams[i] ) );
    }
    
    // Free resources
    CHK( cudaEventDestroy( event ) );
    if( !asyncMalloc )
    {
        CHK( cuMemFree( ringBuffer ) );
    }
    CHK( cuTexObjectDestroy( texture ) );
    CHK( cuMipmappedArrayDestroy( mipmapArray ) );
    for( int i=0; i<numStreams; ++i )
    {
        CHK( cuStreamDestroy( streams[i] ) );
    }
    CHK( cuMemRelease( tilePool ) );
}

void TestTextureFill::doMapTilesTest( imageSource::ImageSource& imageSource, unsigned int tileBlockWidth, unsigned int tileBlockHeight, bool unmap )
{
    // Get image properties
    imageSource::TextureInfo texInfo;
    imageSource.open( &texInfo );

    // Create tile pool to hold all tiles on mip level 0
    CUmemAllocationProp props{};
    props.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location         = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
    props.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;

    CUmemGenericAllocationHandle tilePool = 0;
    CHK( cuMemCreate( &tilePool, texInfo.width * texInfo.height * 16, &props, 0 ) );

    // Create sparse array for texture
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = texInfo.width;
    ad.Height      = texInfo.height;
    ad.Format      = texInfo.format;;
    ad.NumChannels = texInfo.numChannels;
    ad.Flags       = CUDA_ARRAY3D_SPARSE;

    CUmipmappedArray mipmapArray;
    CHK( cuMipmappedArrayCreate( &mipmapArray, &ad, texInfo.numMipLevels ) );

    CUstream stream;
    CHK( cuStreamCreate( &stream, 0U ) );

    // Map the tiles of the texture in blocks
    for( unsigned int y = 0; y < texInfo.height; y += tileBlockHeight )
    {
        for( unsigned int x = 0; x + tileBlockWidth <= texInfo.width; x += tileBlockWidth )
        {
            CUarrayMapInfo mapInfo{};
            unsigned int tilePoolOffset = y * texInfo.width * pixelSize + x * tileBlockHeight * pixelSize;
            fillMapInfo( mipmapArray, 0, x, y, tileBlockWidth, tileBlockHeight, tilePool, tilePoolOffset, mapInfo );
            mapMultipleTiles( &mapInfo, 1, stream );
        }
    }

    // Unmap individual tiles
    if( unmap )
    {
        for( unsigned int y = 0; y < texInfo.height; y += tileHeight )
        {
            for( unsigned int x = 0; x < texInfo.width; x += tileWidth )
            {
                CUarrayMapInfo mapInfo{};
                unsigned int tilePoolOffset = y * texInfo.width * pixelSize + x * tileHeight * pixelSize;
                fillMapInfo( mipmapArray, 0, x, y, tileWidth, tileHeight, tilePool, tilePoolOffset, mapInfo );
                mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_UNMAP;
                mapInfo.memHandle.memHandle = 0ULL;
                mapInfo.offset              = 0ULL;
                CHK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
            }
        }
    }

    CHK( cuStreamDestroy( stream ) );
    CHK( cuMemRelease( tilePool ) );
}

//------------------------------------------------------------------------------
// Test Instances

TEST_F( TestTextureFill, start )
{
    // Make sure that starting up CUDA does not affect the timing of the first test
}

// Test image reading speed for cheap ringbuffer allocator
TEST_F( TestTextureFill, ringBufferAlloc_readImage )
{
    int  numStreams      = 1;
    bool readImage       = true;
    bool asyncAlloc      = false;
    bool mapTiles        = false;
    bool transferTexture = false;
    bool batchMode       = false;
    doTextureFillTest( numStreams, testImage, readImage, asyncAlloc, mapTiles, transferTexture, batchMode );
}

// Test lreading speed using cudaMallocAsync
TEST_F( TestTextureFill, asyncAlloc_readImage )
{
    int  numStreams      = 1;
    bool readImage       = true;
    bool asyncAlloc      = true;
    bool mapTiles        = false;
    bool transferTexture = false;
    bool batchMode       = false;
    doTextureFillTest( numStreams, testImage, readImage, asyncAlloc, mapTiles, transferTexture, batchMode );
}

// Single stream (slow)
TEST_F( TestTextureFill, readImage_mapTiles_transfer )
{
    int  numStreams      = 1;
    bool readImage       = true;
    bool asyncAlloc      = true;
    bool mapTiles        = true;
    bool transferTexture = true;
    bool batchMode       = false;
    doTextureFillTest( numStreams, testImage, readImage, asyncAlloc, mapTiles, transferTexture, batchMode );
}

// Multi-stream (faster than single stream)
TEST_F( TestTextureFill, readImage_mapTiles_transfer_multistream )
{
    int  numStreams      = 4;
    bool readImage       = true;
    bool asyncAlloc      = true;
    bool mapTiles        = true;
    bool transferTexture = true;
    bool batchMode       = false;
    doTextureFillTest( numStreams, testImage, readImage, asyncAlloc, mapTiles, transferTexture, batchMode );
}

// Batch mode (mapping many, then reading many, then transferring many, rather than interleaving) (faster than non-batch mode)
TEST_F( TestTextureFill, readImage_mapTiles_transfer_batch )
{
    int  numStreams      = 1;
    bool readImage       = true;
    bool asyncAlloc      = true;
    bool mapTiles        = true;
    bool transferTexture = true;
    bool batchMode       = true;
    doTextureFillTest( numStreams, testImage, readImage, asyncAlloc, mapTiles, transferTexture, batchMode );
}

// Just mapping, with each mapping call made separately
TEST_F( TestTextureFill, mapTiles_batch1 )
{
    int          numStreams      = 1;
    bool         readImage       = false;
    bool         asyncAlloc      = false;
    bool         mapTiles        = true;
    bool         transferTexture = false;
    bool         batchMode       = true;
    unsigned int mapBatchSize    = 1;
    doTextureFillTest( numStreams, testImage, readImage, asyncAlloc, mapTiles, transferTexture, batchMode, mapBatchSize );
}

// Just mapping , with 16 mappings made per call (almost no speedup under Linux compared to separate calls)
TEST_F( TestTextureFill, mapTiles_batch16 )
{
    int          numStreams      = 1;
    bool         readImage       = false;
    bool         asyncAlloc      = false;
    bool         mapTiles        = true;
    bool         transferTexture = false;
    bool         batchMode       = true;
    unsigned int mapBatchSize    = 16;
    doTextureFillTest( numStreams, testImage, readImage, asyncAlloc, mapTiles, transferTexture, batchMode, mapBatchSize );
}

// Mapping: Regular size tiles (baseline)
// This is faster than the previous mapping tests because we don't create a texture or allocate any extra memory
TEST_F( TestTextureFill, mapTest_64_64 )
{
    doMapTilesTest( testImage, 64, 64, false );
}

// Mapping and unmapping
TEST_F( TestTextureFill, mapTest_64_64_unmap )
{
    doMapTilesTest( testImage, 64, 64, true );
}

// Mapping: double wide tiles (About twice as fast as 64x64)
TEST_F( TestTextureFill, mapTest_128_64 )
{
    doMapTilesTest( testImage, 128, 64, false );
}

// Mapping: triple wide tiles. (About 3x faster than 64x64)
TEST_F( TestTextureFill, mapTest_192_64 )
{
    doMapTilesTest( testImage, 192, 64, false );
}

// Mapping: 4 wide tiles. (About 4x faster than 64x64)
TEST_F( TestTextureFill, mapTest_256_64 )
{
    doMapTilesTest( testImage, 256, 64, false );
}

// Mapping and unmapping: mapping 4 wide tiles and unmapping individual. 
TEST_F( TestTextureFill, mapTest_256_64_unmap )
{
    doMapTilesTest( testImage, 256, 64, true );
}

// Mapping: double high tiles (almost no speedup compared to 64x64)
TEST_F( TestTextureFill, mapTest_64_128 )
{
    doMapTilesTest( testImage, 64, 128, false );
}
