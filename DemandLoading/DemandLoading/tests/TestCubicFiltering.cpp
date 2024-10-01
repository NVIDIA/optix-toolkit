// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "TestSparseTexture.h"

#include "DemandLoaderImpl.h"
#include "Memory/DeviceMemoryManager.h"
#include "PageTableManager.h"
#include "Textures/DemandTextureImpl.h"

#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/texture.h>

using namespace demandLoading;
using namespace imageSource;
using namespace otk;

//------------------------------------------------------------------------------

class TestCubicFiltering : public testing::Test
{
  public:
    void makeOIIOImages();
    void makeDemandTextureImages();
    void makeDiffImages();
    void combineImages();
    void writeImages( const char* fileName );
    void runTest();

    void makeTestImage();

  protected:
    // Input image
    std::string imageFileName;
    int inputImageWidth;
    int inputImageHeight;

    // Output image parameters that will be used by both OIIO and DemandLoading
    int width = 512;
    int height = 512;
    float2 uv00, uv11;
    float2 ddx, ddy;
    unsigned int filterMode;
    unsigned int mipmapFilterMode;

    float imageScale = 1.0f;
    float derivativeScale = 1.0f;
    float diffImageScale = 50.0f;

    // Output images
    std::vector<float4> oiioImage;
    std::vector<float4> oiioDerivativeImage;

    std::vector<float4> demandTextureImage;
    std::vector<float4> demandTextureDerivativeImage;

    std::vector<float4> diffImage;
    std::vector<float4> diffDerivativeImage;

    std::vector<float4> combinedImage;

    float rmseImages;
    float rmseDerivativeImages;
};

//------------------------------------------------------------------------------

inline float mix( float a, float b, float x )
{
    return ( 1.0f - x ) * a + x * b;
}

//------------------------------------------------------------------------------

inline uint8_t quantize( float f )
{
    f = std::min( 1.0f, std::max( 0.0f, f ));
    float value = f * 255;
    return uint8_t( value );
}

void writePPM( const char* filename, float* pixels, int width, int height, int numComponents )
{
    FILE* fp = fopen( filename, "wb" ); /* b - binary mode */
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

//------------------------------------------------------------------------------

void scaleAndBiasImage( std::vector<float4>& img, float4 scale, float4 bias )
{
    for( unsigned int i = 0; i < img.size(); ++i )
    {
        img[i].x = img[i].x * scale.x + bias.x;
        img[i].y = img[i].y * scale.y + bias.y;
        img[i].z = img[i].z * scale.z + bias.z;
        img[i].w = img[i].w * scale.w + bias.w;
    }
}

void makeDiffImage( std::vector<float4>& imgA, std::vector<float4>& imgB, std::vector<float4>&imgDiff, float scale )
{
    imgDiff.resize( imgA.size() );
    for( unsigned int i = 0; i < imgDiff.size(); ++i )
    {
        imgDiff[i].x = scale * ( imgA[i].x - imgB[i].x ) + 0.5f;
        imgDiff[i].y = scale * ( imgA[i].y - imgB[i].y ) + 0.5f;
        imgDiff[i].z = scale * ( imgA[i].z - imgB[i].z ) + 0.5f;
        imgDiff[i].w = scale * ( imgA[i].w - imgB[i].w ) + 0.5f;
    }
}

float imageDiff( std::vector<float4>& imgA, std::vector<float4>& imgB )
{
    double diff = 0.0f;
    for( unsigned int i = 0; i < imgA.size(); ++i )
    {
        float d = (imgA[i].x-imgB[i].x) * (imgA[i].x-imgB[i].x) + 
                  (imgA[i].y-imgB[i].y) * (imgA[i].y-imgB[i].y) + 
                  (imgA[i].z-imgB[i].z) * (imgA[i].z-imgB[i].z);
        diff += d / 3.0f;
    }
    return static_cast<float>( sqrt( diff / imgA.size() ) );
}

void copyImage( std::vector<float4>& dest, int dw, int dh, std::vector<float4>& src, int sw, int sh, int x, int y )
{
    for( int j=0; j<sh; ++j )
    {
        for( int i=0; i<sw; ++i )
        {
            int px = x + i;
            int py = y + j;
            if( px < 0 || px >= dw || py < 0 || py > dh )
                continue;
            dest[py * dw + px] = src[j * sw + i];
        }
    }
}

//------------------------------------------------------------------------------

void TestCubicFiltering::makeOIIOImages()
{
    OIIO::ustring fileName( imageFileName );
    OIIO::TextureOpt options;
    options.interpmode = (OIIO::TextureOpt::InterpMode)filterMode;
    OIIO::TextureSystem* ts = OIIO::TextureSystem::create( true );

    oiioImage.resize(width * height, float4{});
    oiioDerivativeImage.resize(width * height, float4{});

    for( int j=0; j<height; ++j )
    {
        for( int i=0; i<width; ++i )
        {
            float x = ( i + 0.5f ) / width;
            float y = 1.0f - ( j + 0.5f ) / height;
            float s = mix( uv00.x, uv11.x, x );
            float t = mix( uv00.y, uv11.y, y );
            
            float pxVal[4];
            float pxDrds[4];
            float pxDrdt[4];
            ts->texture( fileName, options, s, t, ddx.x, ddx.y, ddy.x, ddy.y, 4, pxVal, pxDrds, pxDrdt );

            // Combine derivative images
            int pixelId = ( height - 1 - j ) * width + i;
            oiioImage[pixelId] = float4{pxVal[0], pxVal[1], pxVal[2], pxVal[3]};
            oiioDerivativeImage[pixelId] = float4{pxDrds[0], pxDrdt[0], 0.0f, 0.0f};
        }
    }

    scaleAndBiasImage( oiioDerivativeImage, float4{derivativeScale/inputImageWidth, derivativeScale/inputImageHeight, 1.0f, 1.0f}, 
                       float4{0.5f, 0.5f, 0.5f, 0.5f} );
}

//------------------------------------------------------------------------------

void TestCubicFiltering::makeDemandTextureImages()
{
    // Initialize CUDA, and init first capable device
    OTK_ERROR_CHECK( cuInit( 0 ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    // Use the first capable device.
    unsigned int deviceIndex = getFirstSparseTextureDevice();
    if( deviceIndex == demandLoading::MAX_DEVICES )
        return;

    // Initialize cuda
    CUstream stream{};
    OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
    OTK_ERROR_CHECK( cuInit( 0 ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( cudaStreamCreate( &stream) );

    // Construct DemandLoaderImpl. 
    DemandLoaderImpl* demandLoader = new DemandLoaderImpl( demandLoading::Options{} );

    // Create TextureDescriptor.
    TextureDescriptor texDesc{};
    texDesc.addressMode[0]   = CU_TR_ADDRESS_MODE_BORDER;
    texDesc.addressMode[1]   = CU_TR_ADDRESS_MODE_BORDER;
    texDesc.filterMode       = filterMode;
    texDesc.mipmapFilterMode = (CUfilter_mode)mipmapFilterMode;
    texDesc.maxAnisotropy    = 16;

    // Create a texture for the textureName
    std::shared_ptr<imageSource::ImageSource> image = imageSource::createImageSource( imageFileName.c_str() );
    image->open(0);
    inputImageWidth = image->getInfo().width;
    inputImageHeight = image->getInfo().height;
    //TextureInfo& getInfo()
    const demandLoading::DemandTexture& texture = demandLoader->createTexture( image, texDesc );
    unsigned int textureId = texture.getId();

    // Allocate memory for the image on the host and device
    demandTextureImage.resize( width * height );
    demandTextureDerivativeImage.resize( width * height );
    CUdeviceptr devImage = 0;
    OTK_ERROR_CHECK( cuMemAlloc( &devImage, width * height * sizeof(float4) ) );
    CUdeviceptr devDerivativeImage = 0;
    OTK_ERROR_CHECK( cuMemAlloc( &devDerivativeImage, width * height * sizeof(float4) ) );

    // Perform launches
    unsigned int totalRequests = 0;
    const unsigned int numLaunches = 4;
    for( unsigned int launchNum = 0; launchNum < numLaunches; ++launchNum )
    {
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        demandLoading::DeviceContext context;
        demandLoader->launchPrepare( stream, context );

        launchCubicTextureSubimageDrawKernel( stream, context, textureId, (float4*)devImage, (float4*)devDerivativeImage, 
                                              width, height, uv00, uv11, ddx, ddy );
        Ticket ticket = demandLoader->processRequests( stream, context );
        ticket.wait();

        totalRequests += ticket.numTasksTotal();
        printf( "Launch %d: %d requests.\n", launchNum, ticket.numTasksTotal() );
    }

    size_t imageSize = width * height * sizeof(float4);
    OTK_ERROR_CHECK( cudaMemcpy( (void*)demandTextureImage.data(), (void*)devImage, imageSize, cudaMemcpyDeviceToHost ) );
    OTK_ERROR_CHECK( cuMemFree( devImage ) );

    OTK_ERROR_CHECK( cudaMemcpy( (void*)demandTextureDerivativeImage.data(), (void*)devDerivativeImage, imageSize, cudaMemcpyDeviceToHost ) );
    OTK_ERROR_CHECK( cuMemFree( devDerivativeImage ) );

    scaleAndBiasImage( demandTextureDerivativeImage, float4{derivativeScale/inputImageWidth, derivativeScale/inputImageHeight, 1.0f, 1.0f}, 
                       float4{0.5f, 0.5f, 0.5f, 0.5f} );
}

//------------------------------------------------------------------------------

void TestCubicFiltering::makeDiffImages()
{
    makeDiffImage( oiioImage, demandTextureImage, diffImage, diffImageScale );
    makeDiffImage( oiioDerivativeImage, demandTextureDerivativeImage, diffDerivativeImage, diffImageScale );
}

//------------------------------------------------------------------------------

void TestCubicFiltering::writeImages( const char* fileName )
{
#define COMBINED_IMAGES
#ifndef COMBINED_IMAGES // Separate images
    writePPM( "image_A_OIIO.ppm", (float*)oiioImage.data(), width, height, 4 );
    writePPM( "image_B_DemandTexture.ppm", (float*)demandTextureImage.data(), width, height, 4 );
    writePPM( "image_C_Diff.ppm", (float*)diffImage.data(), width, height, 4 );

    writePPM( "imageDerivative_A_OIIO.ppm", (float*)oiioDerivativeImage.data(), width, height, 4 );
    writePPM( "imageDerivative_B_DemandTexture.ppm", (float*)demandTextureDerivativeImage.data(), width, height, 4 );
    writePPM( "imageDerivative_C_Diff.ppm", (float*)diffDerivativeImage.data(), width, height, 4 );
#endif

#ifdef COMBINED_IMAGES
    int combinedWidth = width * 3 + 2;
    int combinedHeight = height * 2 + 1;
    std::vector<float4> combinedImage( combinedWidth * combinedHeight, float4{1.0f,1.0f,1.0f,0.0f} );

    copyImage( combinedImage, combinedWidth, combinedHeight, oiioImage, width, height, 0, 0 );
    copyImage( combinedImage, combinedWidth, combinedHeight, demandTextureImage, width, height, width+1, 0 );
    copyImage( combinedImage, combinedWidth, combinedHeight, diffImage, width, height, 2*width+2, 0 );

    copyImage( combinedImage, combinedWidth, combinedHeight, oiioDerivativeImage, width, height, 0, height+1 );
    copyImage( combinedImage, combinedWidth, combinedHeight, demandTextureDerivativeImage, width, height, width+1, height+1 );
    copyImage( combinedImage, combinedWidth, combinedHeight, diffDerivativeImage, width, height, 2*width+2, height+1 );
    
    writePPM( fileName, (float*)combinedImage.data(), combinedWidth, combinedHeight, 4 );
#endif
}

//------------------------------------------------------------------------------

void TestCubicFiltering::runTest()
{
printf("A\n");
    makeDemandTextureImages();
printf("B\n");
    makeOIIOImages();
printf("C\n");
    makeDiffImages();
    rmseImages = imageDiff( oiioImage, demandTextureImage );
    rmseDerivativeImages = imageDiff( oiioDerivativeImage, demandTextureDerivativeImage );
    scaleAndBiasImage( oiioImage, float4{imageScale, imageScale, imageScale, imageScale}, float4{0.0f, 0.0f, 0.0f, 0.0f} );
    scaleAndBiasImage( demandTextureImage, float4{imageScale, imageScale, imageScale, imageScale}, float4{0.0f, 0.0f, 0.0f, 0.0f} );
}

//------------------------------------------------------------------------------

void TestCubicFiltering::makeTestImage()
{
    int w = 128;
    int h = 128;
    std::vector<float4> testImage(w*h, float4{0,0,0,0});
    testImage[64*w + 64] = float4{1,1,1,0};
    writePPM( "testImage.ppm", (float*)testImage.data(), w, h, 4 );
}

//------------------------------------------------------------------------------

TEST_F( TestCubicFiltering, bilinearLvl0 )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.48f, 0.48f};
    uv11 = float2{0.53f, 0.53f};
    ddx = float2{1.0f/128, 0.0f};
    ddy = float2{0.0f, 1.0f/128};
    filterMode = OIIO::TextureOpt::InterpBilinear;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 1.0f;
    derivativeScale = 0.5f;
    diffImageScale = 50.0f;

    runTest();
    writeImages( "image_bilinearLvl0.ppm" );
}

TEST_F( TestCubicFiltering, bilinearLvl0Horizontal )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.40f, 0.40f};
    uv11 = float2{0.60f, 0.60f};
    ddx = float2{0.08f, 0.00f};
    ddy = float2{0.0f, 0.01f};
    filterMode = OIIO::TextureOpt::InterpBilinear;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 10.0f;
    derivativeScale = 1.0f;
    diffImageScale = 2.0f;

    runTest();
    writeImages( "image_bilinearLvl0Horizontal.ppm" );
}

TEST_F( TestCubicFiltering, bilinearLvl0Diagonal )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.4f, 0.4f};
    uv11 = float2{0.6f, 0.6f};
    ddx = float2{0.08f, 0.08f};
    ddy = float2{0.005f, -0.005f};
    filterMode = OIIO::TextureOpt::InterpBilinear;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 10.0f;
    derivativeScale = 1.0f;
    diffImageScale = 2.0f;

    runTest();
    writeImages( "image_bilinearLvl0Diagonal.ppm" );
}

TEST_F( TestCubicFiltering, bilinearMidMip )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.4f, 0.4f};
    uv11 = float2{0.7f, 0.7f};
    ddx = float2{0.08f, 0.08f};
    ddy = float2{0.08f, -0.08f};
    filterMode = OIIO::TextureOpt::InterpBilinear;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 200.0f;
    derivativeScale = 200.0f;
    diffImageScale = 2.0f;

    runTest();
    writeImages( "image_bilinearMidMip.ppm" );
}

TEST_F( TestCubicFiltering, bilinearMidMip2 )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.3f, 0.3f};
    uv11 = float2{0.8f, 0.8f};
    ddx = float2{0.16f, 0.16f};
    ddy = float2{0.16f, -0.16f};
    filterMode = OIIO::TextureOpt::InterpBilinear;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 500.0f;
    derivativeScale = 1000.0f;
    diffImageScale = 2.0f;

    runTest();
    writeImages( "image_bilinearMidMip2.ppm" );
}


TEST_F( TestCubicFiltering, smartBicubicNoMip )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.48f, 0.48f};
    uv11 = float2{0.53f, 0.53f};
    ddx = float2{1.0f/1000, 0.0f};
    ddy = float2{0.0f, 1.0f/1000};
    filterMode = OIIO::TextureOpt::InterpSmartBicubic;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 2.0f;
    derivativeScale = 1.0f;
    diffImageScale = 50.0f;

    runTest();
    writeImages( "image_smartBicubicNoMip.ppm" );
}

/*
TEST_F( TestCubicFiltering, smartBicubicNoMipHorizontal )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.45f, 0.45f};
    uv11 = float2{0.55f, 0.55f};
    ddx = float2{0.05f, 0.05f};
    ddy = float2{0.0f, 0.0f};
    filterMode = OIIO::TextureOpt::InterpSmartBicubic;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 5.0f;
    derivativeScale = 1.0f;
    diffImageScale = 2.0f;

    runTest();
    writeImages( "image_smartBicubicNoMipHorizontal.ppm" );
}

TEST_F( TestCubicFiltering, smartBicubicNoMipDiagonal )
{
    imageFileName = "testImage.exr";
    uv00 = float2{0.45f, 0.45f};
    uv11 = float2{0.55f, 0.55f};
    ddx = float2{1.0f/90, 1.0f/90};
    ddy = float2{1.0f/128, -1.0f/128};
    filterMode = OIIO::TextureOpt::InterpBilinear;
    mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;

    imageScale = 1.0f;
    derivativeScale = 1.0f;
    diffImageScale = 2.0f;

    runTest();
    writeImages( "image_smartBicubicNoMipDiagonal.ppm" );
}
*/
