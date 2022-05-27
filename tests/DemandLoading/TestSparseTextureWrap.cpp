//
//  Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include "TestSparseTexture.h"

#include "Memory/Buffers.h"
#include "Memory/TilePool.h"
#include "Textures/SparseTexture.h"
#include "Util/Exception.h"

#include <ImageSource/CheckerBoardImage.h>

#include <gtest/gtest.h>

#include <cuda.h>

#include <fstream>
#include <math.h>
#include <ostream>

using namespace demandLoading;
using namespace imageSource;

namespace {  // anonymous

const unsigned long long TEX_MEM_PER_DEVICE = 1u << 30; // 1 GB

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
  protected:
    CUstream     m_stream{};
};

TEST_F( TestSparseTextureWrap, Test )
{
    // Initialize CUDA.
    const unsigned int deviceIndex = 0;
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
    DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

    // Create sparse texture.
    TextureDescriptor desc;
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 16;

    TextureInfo info{
        1024, 1024, CU_AD_FORMAT_FLOAT, 4 /*numChannels*/, 11 /*numMipLevels*/
    };

    SparseTexture texture( deviceIndex );
    texture.init( desc, info );

    // Allocate tile buffer.
    const unsigned int  tileWidth  = texture.getTileWidth();
    const unsigned int  tileHeight = texture.getTileHeight();
    std::vector<float4> tileBuffer( tileWidth * tileHeight );
    ASSERT_TRUE( tileBuffer.size() <= sizeof(TileBuffer) );

    const unsigned int  mipLevel = 0;
    TilePool            tilePool( deviceIndex, TEX_MEM_PER_DEVICE );
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
            CUmemGenericAllocationHandle handle;
            size_t                       offset;
            TileBlockDesc                tileBlock = tilePool.allocate( sizeof( TileBuffer ) );
            tilePool.getHandle( tileBlock, &handle, &offset );

            // Map and fill tile.
            texture.fillTile( m_stream, mipLevel, i, j, reinterpret_cast<const char*>( tileBuffer.data() ),
                              CU_MEMORYTYPE_HOST, tileBuffer.size(), handle, offset );
        }
    }

    // Set up kernel output buffer.
    const int    outWidth  = 512;
    const int    outHeight = 512;
    float4*      devOutput;
    const size_t outputSize = outWidth * outHeight * sizeof( float4 );
    DEMAND_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );
    const float       lod           = static_cast<float>( mipLevel );
    const CUtexObject textureObject = texture.getTextureObject();
    launchWrapTestKernel( textureObject, devOutput, outWidth, outHeight, lod );
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Save the output buffer as a PPM.
    savePPM( "testWrap.ppm", outWidth, outHeight, hostOutput.data() );
    std::cout << "Wrote testWrap.ppm" << std::endl;

    DEMAND_CUDA_CHECK( cudaFree( devOutput ) );
}

void testLargeSparseTexture( CUstream stream, unsigned int res, unsigned int mipLevel, const char* outFileName )
{
    // Initialize CUDA.
    const unsigned int deviceIndex = 0;
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
    DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

    // Create sparse texture.
    TextureDescriptor desc;
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 16;

    unsigned int numMipLevels = (unsigned int)( log2( res ) + 1 );

    TextureInfo info{res, res, CU_AD_FORMAT_FLOAT, 4 /*numChannels*/, numMipLevels};

    SparseTexture texture( deviceIndex );
    texture.init( desc, info );

    // Allocate tile buffer.
    const unsigned int  tileWidth  = texture.getTileWidth();
    const unsigned int  tileHeight = texture.getTileHeight();
    std::vector<float4> tileBuffer( tileWidth * tileHeight );
    ASSERT_TRUE( tileBuffer.size() <= sizeof( TileBuffer ) );

    TilePool            tilePool( deviceIndex, TEX_MEM_PER_DEVICE );
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
            CUmemGenericAllocationHandle handle;
            size_t                       offset;
            TileBlockDesc                tileBlock = tilePool.allocate( sizeof( TileBuffer ) );
            tilePool.getHandle( tileBlock, &handle, &offset );

            // Map and fill tile.
            texture.fillTile( stream, mipLevel, i, j, reinterpret_cast<const char*>( tileBuffer.data() ),
                              CU_MEMORYTYPE_HOST, tileBuffer.size(), handle, offset );
        }
    }

    // Set up kernel output buffer.
    const int    outWidth  = 256;
    const int    outHeight = 256;
    float4*      devOutput;
    const size_t outputSize = outWidth * outHeight * sizeof( float4 );
    DEMAND_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );
    const float       lod           = static_cast<float>( mipLevel );
    const CUtexObject textureObject = texture.getTextureObject();
    launchWrapTestKernel( textureObject, devOutput, outWidth, outHeight, lod );
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Save the output buffer as a PPM.
    savePPM( outFileName, outWidth, outHeight, hostOutput.data() );
    std::cout << "wrote " << outFileName << std::endl;

    DEMAND_CUDA_CHECK( cudaFree( devOutput ) );
}

// This test is too slow for inclusion in the smoke tests.
TEST_F( TestSparseTextureWrap, DISABLED_largeTextures )
{
    testLargeSparseTexture( m_stream, 16384, 0, "largeSparse-16k-l0.ppm" );
    testLargeSparseTexture( m_stream, 16384, 2, "largeSparse-16k-l2.ppm" );
    testLargeSparseTexture( m_stream, 8194, 2, "largeSparse-8k-l2.ppm" );
}
