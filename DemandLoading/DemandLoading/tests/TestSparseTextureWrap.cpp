// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Textures/SparseTexture.h"

#include "TestSparseTexture.h"

#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/FixedSuballocator.h>
#include <OptiXToolkit/Memory/HeapSuballocator.h>
#include <OptiXToolkit/Memory/MemoryPool.h>

#include <gtest/gtest.h>

#include <cuda.h>

#include <fstream>
#include <math.h>
#include <ostream>

using namespace demandLoading;
using namespace imageSource;
using namespace otk;

namespace {  // anonymous

// Return ceil(x/y) for integers x and y
inline unsigned int idivCeil( unsigned int x, unsigned int y )
{
    return ( x + y - 1 ) / y;
}

// Get a vector of some handy colors.
std::vector<float4> getColors()
{
    std::vector<float4> colors{
        {255, 0, 0, 0},    // red
        {255, 127, 0, 0},  // orange
        {255, 255, 0, 0},  // yellow
        {0, 255, 0, 0},    // green
        {0, 0, 255, 0},    // blue
        {127, 0, 0, 0},    // dark red
        {127, 63, 0, 0},   // dark orange
        {127, 127, 0, 0},  // dark yellow
        {0, 127, 0, 0},    // dark green
        {0, 0, 127, 0},    // dark blue
    };
    // Normalize the miplevel colors to [0,1]
    for( float4& color : colors )
    {
        color.x /= 255.f;
        color.y /= 255.f;
        color.z /= 255.f;
    }
    return colors;
}

bool savePPM( const char* filename, size_t width, size_t height, const float4* buffer )
{
    std::vector<unsigned char> tmp( width * height * 3 );

    const float*   src        = reinterpret_cast<const float*>( buffer );
    unsigned char* dst        = &tmp[0];
    int            stride     = 4;
    int            row_stride = 0;

    for( int y = static_cast<int>( height ) - 1; y >= 0; --y )
    {
        for( size_t x = 0; x < width; ++x )
        {
            for( int k = 0; k < 3; ++k )
            {
                int chanVal = static_cast<int>( ( *( src + k ) * 255.0f ) );
                *dst++      = static_cast<unsigned char>( chanVal < 0 ? 0 : chanVal > 0xff ? 0xff : chanVal );
            }
            src += stride;
        }
        src += row_stride;
    }

    std::ofstream file( filename, std::ios::out | std::ios::binary );
    if( !file.is_open() )
        return false;

    file << "P6" << std::endl;
    file << width << " " << height << std::endl;
    file << 255 << std::endl;
    file.write( reinterpret_cast<char*>( &tmp[0] ), tmp.size() );
    file.close();
    return true;
}

}  // anonymous namespace

class TestSparseTextureWrap : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA.
        m_deviceIndex = demandLoading::getFirstSparseTextureDevice();
        if( m_deviceIndex == demandLoading::MAX_DEVICES )
            return;

        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );

        m_tilePool.reset(                                              //
            new MemoryPool<TextureTileAllocator, HeapSuballocator>(    //
                new TextureTileAllocator(),                            // allocator
                new HeapSuballocator(),                                // suballocator
                TextureTileAllocator::getRecommendedAllocationSize(),  // allocation granularity
                m_options.maxTexMemPerDevice                           // max size
                ) );
    }

  protected:
    void testLargeSparseTexture( CUstream stream, unsigned int res, unsigned int mipLevel, const char* outFileName );

    unsigned int m_deviceIndex = 0;
    CUstream m_stream{};
    Options m_options{};
    std::unique_ptr<MemoryPool<TextureTileAllocator, HeapSuballocator>> m_tilePool;
};

TEST_F( TestSparseTextureWrap, Test )
{
    // Skip test if sparse textures not supported
    if( m_deviceIndex == demandLoading::MAX_DEVICES )
        return;

    // Initialize CUDA.
    OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    // Create sparse texture.
    TextureDescriptor desc;
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 16;

    TextureInfo info{
        1024, 1024, CU_AD_FORMAT_FLOAT, 4 /*numChannels*/, 11 /*numMipLevels*/, true /*isValid*/, true /*isTiled*/
    };

    SparseTexture texture;
    texture.init( desc, info, nullptr );

    // Allocate tile buffer.
    const unsigned int  tileWidth  = texture.getTileWidth();
    const unsigned int  tileHeight = texture.getTileHeight();
    std::vector<float4> tileBuffer( tileWidth * tileHeight );
    ASSERT_TRUE( tileBuffer.size() * sizeof( float4 ) <= TILE_SIZE_IN_BYTES );

    const unsigned int  mipLevel = 0;
    std::vector<float4> colors( getColors() );

    // Fill all the tiles on the finest miplevel.
    const unsigned int tilesWide = idivCeil( info.width, tileWidth );
    const unsigned int tilesHigh = idivCeil( info.height, tileHeight );
    for( unsigned int j = 0; j < tilesHigh; ++j )
    {
        for( unsigned int i = 0; i < tilesWide; ++i )
        {
            // Fill tile buffer with a solid color.
            unsigned int colorIndex = ( j * tilesWide + i ) % static_cast<unsigned int>( colors.size() );
            std::fill( tileBuffer.begin(), tileBuffer.end(), colors[colorIndex] );

            // Allocate tile backing storage.
            TileBlockHandle bh = m_tilePool->allocTextureTiles( TILE_SIZE_IN_BYTES );

            // Map and fill tile.
            texture.fillTile( m_stream,                                                  // stream to fill tile on
                              mipLevel, i, j,                                            // tile to fill
                              reinterpret_cast<const char*>( tileBuffer.data() ),        // src data
                              CU_MEMORYTYPE_HOST, tileBuffer.size() * sizeof( float4 ),  // src type and size
                              bh.handle, bh.block.offset()                               // dest
                              );
        }
    }

    // Set up kernel output buffer.
    const int    outWidth  = 512;
    const int    outHeight = 512;
    float4*      devOutput;
    const size_t outputSize = outWidth * outHeight * sizeof( float4 );
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const float       lod           = static_cast<float>( mipLevel );
    const CUtexObject textureObject = texture.getTextureObject();
    launchWrapTestKernel( textureObject, devOutput, outWidth, outHeight, lod );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    OTK_ERROR_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Save the output buffer as a PPM.
    savePPM( "testWrap.ppm", outWidth, outHeight, hostOutput.data() );
    std::cout << "Wrote testWrap.ppm" << std::endl;

    OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );
}

void TestSparseTextureWrap::testLargeSparseTexture( CUstream stream, unsigned int res, unsigned int mipLevel, const char* outFileName )
{
    // Create sparse texture.
    TextureDescriptor desc;
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 16;

    unsigned int numMipLevels = (unsigned int)( log2( res ) + 1 );

    TextureInfo info{res, res, CU_AD_FORMAT_FLOAT, 4 /*numChannels*/, numMipLevels, /*isValid=*/true, /*isTiled=*/true};

    SparseTexture texture;
    texture.init( desc, info, nullptr );

    // Allocate tile buffer.
    const unsigned int  tileWidth  = texture.getTileWidth();
    const unsigned int  tileHeight = texture.getTileHeight();
    std::vector<float4> tileBuffer( tileWidth * tileHeight );
    ASSERT_TRUE( tileBuffer.size() * sizeof( float4 ) <= TILE_SIZE_IN_BYTES );
    std::vector<float4> colors( getColors() );

    // Fill all the tiles on the specified miplevel.
    const unsigned int tilesWide = idivCeil( info.width >> mipLevel, tileWidth );
    const unsigned int tilesHigh = idivCeil( info.height >> mipLevel, tileHeight );
    for( unsigned int j = 0; j < tilesHigh; ++j )
    {
        if( j * tilesWide > 50000 )
            break;  // Only load up to 50000 tiles

        for( unsigned int i = 0; i < tilesWide; ++i )
        {
            // Fill tile buffer with a solid color.
            unsigned int colorIndex = ( j * tilesWide + i ) % static_cast<unsigned int>( colors.size() );
            std::fill( tileBuffer.begin(), tileBuffer.end(), colors[colorIndex] );

            // Allocate tile backing storage.
            TileBlockHandle bh = m_tilePool->allocTextureTiles( TILE_SIZE_IN_BYTES );

            // Map and fill tile.
            texture.fillTile( stream,                                              // stream
                              mipLevel, i, j,                                      // Tile to fill
                              reinterpret_cast<const char*>( tileBuffer.data() ),  // source data
                              CU_MEMORYTYPE_HOST, TILE_SIZE_IN_BYTES,              // source type and size
                              bh.handle, bh.block.offset()                         // dest
                              );
        }
    }

    // Set up kernel output buffer.
    const int    outWidth  = 256;
    const int    outHeight = 256;
    float4*      devOutput;
    const size_t outputSize = outWidth * outHeight * sizeof( float4 );
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const float       lod           = static_cast<float>( mipLevel );
    const CUtexObject textureObject = texture.getTextureObject();
    launchWrapTestKernel( textureObject, devOutput, outWidth, outHeight, lod );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    OTK_ERROR_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Save the output buffer as a PPM.
    savePPM( outFileName, outWidth, outHeight, hostOutput.data() );
    std::cout << "wrote " << outFileName << std::endl;

    OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );
}

// This test is too slow for inclusion in the smoke tests.
TEST_F( TestSparseTextureWrap, DISABLED_largeTextures )
{
    // Skip test if sparse textures not supported
    if( m_deviceIndex == demandLoading::MAX_DEVICES )
        return;

    testLargeSparseTexture( m_stream, 16384, 0, "largeSparse-16k-l0.ppm" );
    testLargeSparseTexture( m_stream, 16384, 2, "largeSparse-16k-l2.ppm" );
    testLargeSparseTexture( m_stream, 8194, 2, "largeSparse-8k-l2.ppm" );
}
