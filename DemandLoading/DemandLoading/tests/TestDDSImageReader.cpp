// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TestDrawTexture.h"

#include "SourceDir.h"  // generated from SourceDir.h.in

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/DDSImageReader.h>

#include <gtest/gtest.h>
#include <cuda.h>

using namespace imageSource;

inline uint8_t quantize( float f )
{
    f = std::min( 1.0f, std::max( 0.0f, f ));
    float value = f * 255;
    return uint8_t( value );
}

void writePPM( const char* filename, float* pixels, int width, int height, int numComponents )
{
    FILE* fp = fopen( filename, "wb" ); 
    fprintf( fp, "P6\n%d %d\n255\n", width, height );
    for( int j = 0; j < height; ++j )
    {
        for( int i = 0; i < width; ++i )
        {
            float r = (numComponents >= 1) ? pixels[ (j*width + i) * numComponents + 0 ] : 0;
            float g = (numComponents >= 2) ? pixels[ (j*width + i) * numComponents + 1 ] : 0;
            float b = (numComponents >= 3) ? pixels[ (j*width + i) * numComponents + 2 ] : 0;
            uint8_t color[3] = {quantize( r ), quantize( g ), quantize( b )};
            fwrite( color, sizeof( uint8_t ), 3, fp );
        }
    }
    fclose( fp );
}

class TestDDSImageReader : public testing::Test
{
  public:
    void SetUp() override
    {
        unsigned int deviceIndex = 0;
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
    }

    void TearDown() override {}

    void loadDDSTexture( const char* imageFileName, imageSource::TextureInfo& imageInfo, CUtexObject& texObj )
    {
        // Load image into a DDSImageReader
        imageSource::DDSImageReader image( imageFileName, false );
        image.open( &imageInfo );

        // Create the mipmapped cuda array
        CUmipmappedArray mipmappedArray;
        CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
        arrayDesc.Format = imageInfo.format;
        arrayDesc.NumChannels = imageInfo.numChannels;
        arrayDesc.Width = imageInfo.width;
        arrayDesc.Height = imageInfo.height;
        arrayDesc.Depth = 0;
        OTK_ERROR_CHECK( cuMipmappedArrayCreate(&mipmappedArray, &arrayDesc, imageInfo.numMipLevels) );

        // Read the mip level data from file and copy to cuda arrays
        for( uint32_t mipLevel = 0; mipLevel < imageInfo.numMipLevels; ++mipLevel )
        {
            uint32_t mipWidth = imageInfo.width >> mipLevel;
            uint32_t mipHeight = imageInfo.height >> mipLevel;
            std::vector<char> texData( image.getMipLevelSizeInBytes( mipLevel ) );
            image.readMipLevel( texData.data(), mipLevel, mipWidth, mipHeight, CUstream{0} );

            CUarray cuArray;
            OTK_ERROR_CHECK( cuMipmappedArrayGetLevel(&cuArray, mipmappedArray, mipLevel) );
            CUDA_MEMCPY2D copyParams = {};
            copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
            copyParams.srcHost = texData.data();
            copyParams.srcPitch = image.getMipLevelWidthInBytes( mipLevel );
            copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copyParams.dstArray = cuArray;
            copyParams.WidthInBytes = image.getMipLevelWidthInBytes( mipLevel );
            copyParams.Height = mipHeight / BC_BLOCK_HEIGHT;
            OTK_ERROR_CHECK( cuMemcpy2D( &copyParams ) );
        }

        // Specify the resource descriptor
        CUDA_RESOURCE_DESC resDesc = {};
        resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        resDesc.res.mipmap.hMipmappedArray = mipmappedArray;

        // Specify the texture descriptor
        CUDA_TEXTURE_DESC texDesc = {};
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
        texDesc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
        texDesc.maxAnisotropy = 16;
        texDesc.minMipmapLevelClamp = 0;
        texDesc.maxMipmapLevelClamp = imageInfo.numMipLevels - 1;
        texDesc.mipmapLevelBias = 0;
        texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;

        // Create the texture object
        OTK_ERROR_CHECK( cuTexObjectCreate( &texObj, &resDesc, &texDesc, nullptr ) );
    }

    void renderTexture( CUtexObject texObj, std::vector<float4>& image, int imageWidth, int imageHeight,
                        float2 ddx, float2 ddy )
    {
        // Allocate memory for the image on the host and device
        image.resize( imageWidth * imageHeight, float4{0.0f, 0.0f, 0.0f, 0.0f} );
        CUdeviceptr devImage = 0;
        OTK_ERROR_CHECK( cuMemAlloc( &devImage, imageWidth * imageHeight * sizeof(float4) ) );

        // Draw the image
        CUstream stream{};
        OTK_ERROR_CHECK( cudaStreamCreate( &stream) );
        launchDrawTextureKernel( stream, reinterpret_cast<float4*>( devImage ), imageWidth, imageHeight, texObj, 
            float2{0.0f, 0.0f}, float2{1.0f, 1.0f}, ddx, ddy );
    
        // Copy image back to host
        size_t imageSize = imageWidth * imageHeight * sizeof(float4);
        OTK_ERROR_CHECK( cudaMemcpy( (void*)image.data(), (void*)devImage, imageSize, cudaMemcpyDeviceToHost ) );
        OTK_ERROR_CHECK( cuMemFree( devImage ) );
    }

    void renderDDSImage( const char* imageFile, const char* outFile, 
                         std::vector<float4>& image, int imageWidth, int imageHeight, float2 ddx, float2 ddy )
    {
        imageSource::TextureInfo imageInfo = {};
        CUtexObject texObj = 0;
        loadDDSTexture( imageFile, imageInfo, texObj );

        renderTexture( texObj, image, imageWidth, imageHeight, ddx, ddy );
        if( outFile != nullptr )
            writePPM( outFile, reinterpret_cast<float*>( image.data() ), imageWidth, imageHeight, 4 );
    }
};

TEST_F( TestDDSImageReader, bc1 )
{
    std::vector<float4> image;
    std::string textureName = getSourceDir() + "/Textures/colors256-bc1.dds";
    renderDDSImage( textureName.c_str(), nullptr, image, 256, 256, float2{0.001f, 0.0f}, float2{0.0f, 0.001f} );

    // Check beginning pixel (should be white)
    EXPECT_EQ( image[0].x, 1.0f );
    EXPECT_EQ( image[0].y, 1.0f );
    EXPECT_EQ( image[0].z, 1.0f );
}
