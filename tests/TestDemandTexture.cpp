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

#include "TestSparseTexture.h"

#include "DemandLoaderImpl.h"
#include "Memory/TilePool.h"
#include "PageTableManager.h"
#include "Textures/DemandTextureImpl.h"
#include "Util/Exception.h"

#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>

#include <gtest/gtest.h>

#include <cuda.h>

#include <memory>

using namespace demandLoading;
using namespace imageSource;

const unsigned long long TEX_MEM_PER_DEVICE = 1u << 30; // 1 GB

class TestDemandTexture : public testing::Test
{
  public:
    void SetUp()
    {
        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cuInit( 0 ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );
    }

    void initTexture( unsigned int width, unsigned int height, bool useMipMaps = true, bool tiledImage = true )
    {
        m_width = width;
        m_height = height;

        // Construct DemandLoaderImpl.  DemandTexture needs it to construct a TextureRequestHandler,
        // and it's provides a PageTableManager that's needed by initSampler().
        demandLoading::Options options{};
        options.useSmallTextureOptimization = true;
        m_loader.reset( new DemandLoaderImpl( options ) );

        // Use the first capable device.
        m_deviceIndex = m_loader->getDevices().at(0);
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

        // Create TextureDescriptor.
        m_desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
        m_desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
        m_desc.filterMode       = CU_TR_FILTER_MODE_POINT;
        m_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        m_desc.maxAnisotropy    = 16;

        // Create CheckerBoardImage.
        std::shared_ptr<ImageSource> image =
            std::make_shared<CheckerBoardImage>( m_width, m_height, /*squaresPerSide*/ 4, useMipMaps, tiledImage );

        // Construct and initialize DemandTexture
        m_texture.reset( new DemandTextureImpl( /*id*/ 0, m_numDevices, m_desc, image, m_loader.get() ) );
        m_texture->open();
        m_texture->init( m_deviceIndex );
    }

  protected:
    unsigned int                       m_deviceIndex = 0;
    CUstream                           m_stream{};
    unsigned int                       m_numDevices = 1;
    unsigned int                       m_width      = 256;
    unsigned int                       m_height     = 256;
    TextureDescriptor                  m_desc{};
    std::unique_ptr<DemandTextureImpl> m_texture;
    std::unique_ptr<DemandLoaderImpl>  m_loader;
};

TEST_F( TestDemandTexture, TestInit )
{
    initTexture(256, 256);
}

TEST_F( TestDemandTexture, TestInitNonMipMapped )
{
    initTexture(256, 256, false);
}

TEST_F( TestDemandTexture, TestFillTile )
{
    initTexture(256, 256);

    // Read and fill the corner tiles, leaving the others non-resident.
    TilePool tilePool( TEX_MEM_PER_DEVICE );
    unsigned int      mipLevel      = 0;
    uint2             tileCoords[4] = {{0, 0}, {0, 3}, {3, 0}, {3, 3}};
    TileBuffer        tileBuffer;
    for( unsigned int i = 0; i < 4; ++i )
    {
        // Read tile.
        unsigned int tileX = tileCoords[i].x;
        unsigned int tileY = tileCoords[i].y;
        EXPECT_EQ( true, m_texture->readTile( mipLevel, tileX, tileY, tileBuffer.data, sizeof( TileBuffer ), CUstream{} ) );

        // Fill tile.
        TileBlockDesc                tileBlock = tilePool.allocate( sizeof( TileBuffer ) );
        CUmemGenericAllocationHandle handle;
        size_t                       offset;
        tilePool.getHandle( tileBlock, &handle, &offset );

        m_texture->fillTile( m_deviceIndex, m_stream, mipLevel, tileX, tileY, reinterpret_cast<char*>( tileBuffer.data ),
                             CU_MEMORYTYPE_HOST, sizeof( TileBuffer ), handle, offset );
    }

    // Set up kernel output buffer.
    const int outWidth  = 4;
    const int outHeight = 4;
    float4*   devOutput;
    size_t    outputSize = outWidth * outHeight * sizeof( float4 );
    DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );
    float       lod           = static_cast<float>( mipLevel );
    CUtexObject textureObject = m_texture->getTextureObject( m_deviceIndex );
    launchSparseTextureKernel( textureObject, devOutput, outWidth, outHeight, lod );
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Validate the red channels of the output.  When resident, the red channel is a 0/1 checkerboard.
    // The output channels are -1 if the tile is not resident.
    float expected[4][4] = {{1.f, -1.f, -1.f, 0.f}, {-1.f, -1.f, -1.f, -1.f}, {-1.f, -1.f, -1.f, -1.f}, {0.f, -1.f, -1.f, 1.f}};
    for( int j = 0; j < outHeight; ++j )
    {
        for( int i = 0; i < outWidth; ++i )
        {
            const float4& pixel = hostOutput[j * outWidth + i];
            EXPECT_EQ( expected[j][i], pixel.x );
        }
    }

    DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );
}

TEST_F( TestDemandTexture, TestReadMipTail )
{
    initTexture(256, 256); 

    // Read the entire mip tail.
    std::unique_ptr<MipTailBuffer> buffer( new MipTailBuffer );
    EXPECT_NO_THROW( m_texture->readMipTail( buffer->data, sizeof( MipTailBuffer ), CUstream{} ) );

    // For now we print the levels in the mip tail for visual validation.
    const TextureInfo& info      = m_texture->getInfo();
    EXPECT_TRUE( info.isValid );
    unsigned int       pixelSize = info.numChannels * getBytesPerChannel( info.format );
    size_t             offset    = 0;
    for( unsigned int mipLevel = m_texture->getMipTailFirstLevel(); mipLevel < info.numMipLevels; ++mipLevel )
    {
        const float4* texels = reinterpret_cast<const float4*>( buffer->data + offset );

        uint2 levelDims = m_texture->getMipLevelDims( mipLevel );
        for( unsigned int y = 0; y < levelDims.y; ++y )
        {
            for( unsigned int x = 0; x < levelDims.x; ++x )
            {
                float4 texel = texels[y * levelDims.x + x];

                // Quantize components to [0..9] for compact output.
                printf( "%i%i%i ", static_cast<int>( 9 * texel.x ), static_cast<int>( 9 * texel.y ),
                        static_cast<int>( 9 * texel.z ) );
            }
            printf( "\n" );
        }

        offset += levelDims.x * levelDims.y * pixelSize;
    }
}

TEST_F( TestDemandTexture, TestFillMipTail )
{
    initTexture(256, 256);

    // Read the entire mip tail.
    std::unique_ptr<MipTailBuffer> buffer( new MipTailBuffer );
    EXPECT_NO_THROW( m_texture->readMipTail( buffer->data, sizeof( MipTailBuffer ), CUstream{} ) );

    // Map the backing storage and fill it.
    TilePool tilePool( TEX_MEM_PER_DEVICE );
    TileBlockDesc                tileBlock = tilePool.allocate( sizeof( MipTailBuffer ) );
    CUmemGenericAllocationHandle handle;
    size_t                       offset;
    tilePool.getHandle( tileBlock, &handle, &offset );
    m_texture->fillMipTail( m_deviceIndex, m_stream, buffer->data, CU_MEMORYTYPE_HOST, sizeof( MipTailBuffer ), handle, offset );

    // Set up kernel output buffer.
    const int outWidth  = 4;
    const int outHeight = 4;
    float4*   devOutput;
    size_t    outputSize = outWidth * outHeight * sizeof( float4 );
    DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );
    float       lod           = static_cast<float>( m_texture->getMipTailFirstLevel() );
    CUtexObject textureObject = m_texture->getTextureObject( m_deviceIndex );
    launchSparseTextureKernel( textureObject, devOutput, outWidth, outHeight, lod );
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Validate output.  The first level of the mip tail is a green/black checkerboard
    unsigned int pattern[4][4] = {{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}};
    for( int j = 0; j < outHeight; ++j )
    {
        for( int i = 0; i < outWidth; ++i )
        {
            const float4& pixel = hostOutput[j * outWidth + i];
            EXPECT_EQ( 0.f, pixel.x );
            EXPECT_EQ( pattern[j][i] ? 1.f : 0.f, pixel.y );
            EXPECT_EQ( 0.f, pixel.z );
            EXPECT_EQ( 0.f, pixel.w );
        }
    }

    DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );
}

TEST_F( TestDemandTexture, TestDenseTexture )
{
    initTexture( 32, 32 ); // small enough to be a dense texture
    const TextureInfo& info = m_texture->getInfo();
    EXPECT_TRUE( info.isValid );

    // Make sure it is using a dense texture
    EXPECT_FALSE( m_texture->useSparseTexture() );

    // Read the entire texture (as a mip tail), and fill it.
    std::unique_ptr<MipTailBuffer> buffer( new MipTailBuffer );
    EXPECT_NO_THROW( m_texture->readMipTail( buffer->data, sizeof( MipTailBuffer ), CUstream{} ) );
    m_texture->fillDenseTexture( m_deviceIndex, m_stream, buffer->data, info.width, info.height, true );

    // Set up kernel output buffer.
    const int outWidth  = 4;
    const int outHeight = 4;
    float4*   devOutput;
    size_t    outputSize = outWidth * outHeight * sizeof( float4 );
    DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );
    float       lod           = 0;
    CUtexObject textureObject = m_texture->getTextureObject( m_deviceIndex );
    launchSparseTextureKernel( textureObject, devOutput, outWidth, outHeight, lod );
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Validate output. (Red checkerboard)
    float4 pattern[16] = {
        {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f},  //
        {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f},  //
        {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f},  //
        {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};

    for( int j = 0; j < outHeight; ++j )
    {
        for( int i = 0; i < outWidth; ++i )
        {
            int idx = j * outWidth + i;
            EXPECT_EQ( pattern[idx].x, hostOutput[idx].x );
            EXPECT_EQ( pattern[idx].y, hostOutput[idx].y );
            EXPECT_EQ( pattern[idx].z, hostOutput[idx].z );
            EXPECT_EQ( pattern[idx].w, hostOutput[idx].w );
        }
    }

    DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );
}

TEST_F( TestDemandTexture, TestDenseNonMipMappedTexture )
{
    initTexture( 256, 256, false, false ); 
    const TextureInfo& info = m_texture->getInfo();
    EXPECT_TRUE( info.isValid );

    // Make sure it is using a dense texture
    EXPECT_FALSE( m_texture->useSparseTexture() );

    // Read the entire texture, and fill it.
    std::vector<char> buffer( 256 * 256 * imageSource::getBytesPerChannel( info.format ) * info.numChannels );
    EXPECT_NO_THROW( m_texture->readNonMipMappedData( buffer.data(), buffer.size(), CUstream{} ) );
    m_texture->fillDenseTexture( m_deviceIndex, m_stream, buffer.data(), info.width, info.height, true );

    // Set up kernel output buffer.
    const int outWidth  = 4;
    const int outHeight = 4;
    float4*   devOutput;
    size_t    outputSize = outWidth * outHeight * sizeof( float4 );
    DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );
    float       lod           = 0;
    CUtexObject textureObject = m_texture->getTextureObject( m_deviceIndex );
    launchSparseTextureKernel( textureObject, devOutput, outWidth, outHeight, lod );
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Validate output. (Red checkerboard)
    float4 pattern[16] = {
        {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f},  //
        {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f},  //
        {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f},  //
        {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};

    for( int j = 0; j < outHeight; ++j )
    {
        for( int i = 0; i < outWidth; ++i )
        {
            int idx = j * outWidth + i;
            EXPECT_EQ( pattern[idx].x, hostOutput[idx].x );
            EXPECT_EQ( pattern[idx].y, hostOutput[idx].y );
            EXPECT_EQ( pattern[idx].z, hostOutput[idx].z );
            EXPECT_EQ( pattern[idx].w, hostOutput[idx].w );
        }
    }

    DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );
}

TEST_F( TestDemandTexture, TestSparseNonMipmappedTexture )
{
    initTexture(256, 256, false, true);

    // Read and fill the corner tiles, leaving the others non-resident.
    TilePool tilePool( TEX_MEM_PER_DEVICE );
    unsigned int      mipLevel      = 0;
    uint2             tileCoords[4] = {{0, 0}, {0, 3}, {3, 0}, {3, 3}};
    TileBuffer        tileBuffer;
    for( unsigned int i = 0; i < 4; ++i )
    {
        // Read tile.
        unsigned int tileX = tileCoords[i].x;
        unsigned int tileY = tileCoords[i].y;
        EXPECT_EQ( true, m_texture->readTile( mipLevel, tileX, tileY, tileBuffer.data, sizeof( TileBuffer ), CUstream{} ) );

        // Fill tile.
        TileBlockDesc                tileBlock = tilePool.allocate( sizeof( TileBuffer ) );
        CUmemGenericAllocationHandle handle;
        size_t                       offset;
        tilePool.getHandle( tileBlock, &handle, &offset );

        m_texture->fillTile( m_deviceIndex, m_stream, mipLevel, tileX, tileY, reinterpret_cast<char*>( tileBuffer.data ),
                             CU_MEMORYTYPE_HOST, sizeof( TileBuffer ), handle, offset );
    }

    // Set up kernel output buffer.
    const int outWidth  = 4;
    const int outHeight = 4;
    float4*   devOutput;
    size_t    outputSize = outWidth * outHeight * sizeof( float4 );
    DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Launch the worker.
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );
    float       lod           = 1.0f; // Give a non-zero lod, and test if level 0 is still retrieved in tex2D
    CUtexObject textureObject = m_texture->getTextureObject( m_deviceIndex );
    launchSparseTextureKernel( textureObject, devOutput, outWidth, outHeight, lod );
    DEMAND_CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy output buffer to host.
    std::vector<float4> hostOutput( outWidth * outHeight );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput, outputSize, cudaMemcpyDeviceToHost ) );

    // Validate the red channels of the output.  When resident, the red channel is a 0/1 checkerboard.
    // The output channels are -1 if the tile is not resident.
    float expected[4][4] = {{1.f, -1.f, -1.f, 0.f}, {-1.f, -1.f, -1.f, -1.f}, {-1.f, -1.f, -1.f, -1.f}, {0.f, -1.f, -1.f, 1.f}};
    for( int j = 0; j < outHeight; ++j )
    {
        for( int i = 0; i < outWidth; ++i )
        {
            const float4& pixel = hostOutput[j * outWidth + i];
            EXPECT_EQ( expected[j][i], pixel.x );
        }
    }

    DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );
}
