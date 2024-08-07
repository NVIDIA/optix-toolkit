
#include "TestSparseVsDenseTextures.h"

#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <math.h>

class TestSparseVsDenseTextures : public testing::Test
{
  protected:
    unsigned int     m_deviceIndex = 0;
    CUmipmappedArray m_sparseArray;
    CUmipmappedArray m_denseArray;
    CUtexObject      m_sparseTexture;
    CUtexObject      m_denseTexture;

  public:
    void SetUp() override
    {
        m_deviceIndex = demandLoading::getFirstSparseTextureDevice();
        if( m_deviceIndex == demandLoading::MAX_DEVICES )
            return;

        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
    }

    void createSparseArray( int width, int height );
    void createSparseTexture( int width, int height );
    void createDenseArray( int width, int height );
    void createDenseTexture( int width, int height );

    void TearDown() override
    {
    }

    static int getNumMipLevels( int width, int height );
};

int TestSparseVsDenseTextures::getNumMipLevels( int width, int height )
{
    return static_cast<int>( 1 + ceil( log2( std::max( width, height ) ) ) );
}

void TestSparseVsDenseTextures::createSparseArray( int width, int height )
{
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = width;
    ad.Height      = height;
    ad.Format      = CU_AD_FORMAT_FLOAT;
    ad.NumChannels = 4;
    ad.Flags       = CUDA_ARRAY3D_SPARSE;
    OTK_ERROR_CHECK( cuMipmappedArrayCreate( &m_sparseArray, &ad, getNumMipLevels( width, height ) ) );
}

void TestSparseVsDenseTextures::createSparseTexture( int width, int height )
{
    // Create CUDA texture descriptor
    CUDA_TEXTURE_DESC td{};
    td.addressMode[0]      = CU_TR_ADDRESS_MODE_WRAP;
    td.addressMode[1]      = CU_TR_ADDRESS_MODE_WRAP;
    td.filterMode          = CU_TR_FILTER_MODE_LINEAR;
    td.flags               = CU_TRSF_NORMALIZED_COORDINATES;
    td.maxAnisotropy       = 16;
    td.mipmapFilterMode    = CU_TR_FILTER_MODE_LINEAR;
    td.maxMipmapLevelClamp = static_cast<float>( getNumMipLevels( width, height ) - 1 );
    td.minMipmapLevelClamp = 0.f;

    // Create texture object.
    CUDA_RESOURCE_DESC rd{};
    rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    rd.res.mipmap.hMipmappedArray = m_sparseArray;
    OTK_ERROR_CHECK( cuTexObjectCreate( &m_sparseTexture, &rd, &td, nullptr ) );
}

void TestSparseVsDenseTextures::createDenseArray( int width, int height )
{
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = width;
    ad.Height      = height;
    ad.Format      = CU_AD_FORMAT_FLOAT;
    ad.NumChannels = 4;

    OTK_ERROR_CHECK( cuMipmappedArrayCreate( &m_denseArray, &ad, getNumMipLevels( width, height ) ) );
}

void TestSparseVsDenseTextures::createDenseTexture( int width, int height )
{
    // Create CUDA texture descriptor
    CUDA_TEXTURE_DESC td{};
    td.addressMode[0]      = CU_TR_ADDRESS_MODE_WRAP;
    td.addressMode[1]      = CU_TR_ADDRESS_MODE_WRAP;
    td.filterMode          = CU_TR_FILTER_MODE_LINEAR;
    td.flags               = CU_TRSF_NORMALIZED_COORDINATES;
    td.maxAnisotropy       = 16;
    td.mipmapFilterMode    = CU_TR_FILTER_MODE_LINEAR;
    td.maxMipmapLevelClamp = static_cast<float>( getNumMipLevels( width, height ) - 1 );
    td.minMipmapLevelClamp = 0.f;

    // Create texture object
    CUDA_RESOURCE_DESC rd{};
    rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    rd.res.mipmap.hMipmappedArray = m_denseArray;
    OTK_ERROR_CHECK( cuTexObjectCreate( &m_denseTexture, &rd, &td, nullptr ) );
}

TEST_F( TestSparseVsDenseTextures, denseSparseSparse )
{
    // Skip test if device does not support sparse textures
    if( m_deviceIndex == demandLoading::MAX_DEVICES )
            return;

    createDenseArray( 8, 8 );
    createDenseTexture( 8, 8 );

    createSparseArray( 64, 64 );
    createSparseTexture( 64, 64 );

    launchSparseVsDenseTextureKernel( m_sparseTexture );
    cudaDeviceSynchronize();
}
