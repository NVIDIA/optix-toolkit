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

#include "DemandLoaderImpl.h"
#include "DemandLoaderTestKernels.h"
#include "Util/Exception.h"

#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <functional>

using namespace demandLoading;
using namespace imageSource;


class TestDemandLoader : public testing::Test
{
  public:
    void SetUp() override
    {
        // Create one stream per device.
        int numDevices;
        cudaGetDeviceCount( &numDevices );
        m_streams.resize( numDevices );
        for( int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
        {
            // Initialize CUDA
            DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
            DEMAND_CUDA_CHECK( cudaFree( nullptr ) );

            // Create a stream per device.
            DEMAND_CUDA_CHECK( cuStreamCreate( &m_streams[deviceIndex], 0 ) );
        }

        // Create DemandLoader
        m_loader = dynamic_cast<DemandLoaderImpl*>( createDemandLoader( Options() ) );

        // Create ImageSource
        m_imageSource =
            std::unique_ptr<ImageSource>( new CheckerBoardImage( 2048, 2048, 32 /*squaresPerSide*/, true /*useMipmaps*/ ) );

        // Create TextureDescriptor
        m_descriptor.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
        m_descriptor.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
        m_descriptor.filterMode       = CU_TR_FILTER_MODE_LINEAR;
        m_descriptor.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
        m_descriptor.maxAnisotropy    = 16;
    }

    void TearDown() override
    {
        for( unsigned int deviceIndex = 0; deviceIndex < static_cast<unsigned int>( m_streams.size() ); ++deviceIndex )
        {
            DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
            DEMAND_CUDA_CHECK( cuStreamDestroy( m_streams[deviceIndex] ) );
        }
        destroyDemandLoader( m_loader );
        m_loader = nullptr;
    }


    int launchKernel( unsigned int deviceIndex, CUstream stream, const std::function<void( const DeviceContext& )>& launchFunction )
    {
        // Prepare for launch, obtaining host-side DeviceContext.
        DeviceContext context;
        bool          ok = m_loader->launchPrepare( deviceIndex, stream, context );
        EXPECT_TRUE( ok );

        // Copy DeviceContext to device.
        DeviceContext* devContext;
        DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devContext ), sizeof( DeviceContext ) ) );
        DEMAND_CUDA_CHECK( cudaMemcpy( devContext, &context, sizeof( DeviceContext ), cudaMemcpyHostToDevice ) );

        // Call the given function to launch the kernel.
        launchFunction( context );

        // Process requests.
        Ticket ticket = m_loader->processRequests( deviceIndex, stream, context );
        ticket.wait();

        DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devContext ) ) );

        return ticket.numTasksTotal();
    }

  protected:
    std::vector<CUstream>        m_streams;
    DemandLoaderImpl*            m_loader{};
    std::shared_ptr<ImageSource> m_imageSource;
    TextureDescriptor            m_descriptor{};
};

TEST_F( TestDemandLoader, TestCreateDestroy )
{
    // No operations
}

TEST_F( TestDemandLoader, TestCreateTexture )
{
    m_loader->createTexture( m_imageSource, m_descriptor );
    // The texture is opaque, so we can't really validate it.
}

class TestDemandLoaderResident : public TestDemandLoader
{
  public:
    void SetUp() override
    {
        TestDemandLoader::SetUp();
        // Allocate device memory for kernel output.
        DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &m_devIsResident ), sizeof( bool ) ) );
    }

    void TearDown() override
    {
        DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( m_devIsResident ) ) );
        TestDemandLoader::TearDown();
    }

protected:
    int launchKernelAndSynchronize( unsigned int deviceIndex, unsigned int pageId, bool* isResident )
    {
        bool*    devIsResident = m_devIsResident;
        CUstream stream        = m_streams[deviceIndex];
        const int numFilled = launchKernel( deviceIndex, stream, [stream, pageId, devIsResident]( const DeviceContext& context ) {
            launchPageRequester( stream, context, pageId, devIsResident );
        } );
        // Copy isResident result to host.
        DEMAND_CUDA_CHECK( cuStreamSynchronize( stream ) );
        DEMAND_CUDA_CHECK( cudaMemcpy( isResident, m_devIsResident, sizeof( bool ), cudaMemcpyDeviceToHost ) );
        return numFilled;
    }

    bool* m_devIsResident{};
};

TEST_F( TestDemandLoaderResident, TestSamplerRequest )
{
    const DemandTexture& texture = m_loader->createTexture( m_imageSource, m_descriptor );
    const unsigned int   pageId  = texture.getId();

    // TODO: this fails with multiple GPUs under both Windows and Linux.
    // for( unsigned int deviceIndex : m_loader->getDevices() )
    for( unsigned int deviceIndex = 0; deviceIndex < 1; ++deviceIndex )
    {
        bool isResident1{true};
        bool isResident2{};
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );

        // Launch the kernel, which requests the texture sampler and returns a boolean indicating whether it's resident.
        // The helper function processes any requests.
        const int numFilled1 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident1 );
        // Launch the kernel again.  The sampler should now be resident.
        const int numFilled2 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident2 );

        EXPECT_EQ( 1, numFilled1 );
        EXPECT_FALSE( isResident1 );
        EXPECT_EQ( 0, numFilled2 );
        EXPECT_TRUE( isResident2 );
    }
}

class MockResourceLoader
{
  public:
    MOCK_METHOD( bool, loadResource, ( unsigned int deviceIndex, CUstream stream, unsigned int pageIndex, void** pageTableEntry ) );

    static bool callback( unsigned int deviceIndex, CUstream stream, unsigned int pageIndex, void* context, void** pageTableEntry )
    {
        return static_cast<MockResourceLoader*>( context )->loadResource( deviceIndex, stream, pageIndex, pageTableEntry );
    }
};

TEST_F( TestDemandLoaderResident, TestResourceRequest )
{
    // TODO: this fails with multiple GPUs under both Windows and Linux.
    //const std::vector<unsigned int> devices = m_loader->getDevices();
    const unsigned int devices[] = { 0 };
    using namespace testing;
    const unsigned int numPages  = 256;
    StrictMock<MockResourceLoader> resLoader;
    const unsigned int startPage = m_loader->createResource( numPages, StrictMock<MockResourceLoader>::callback, &resLoader );
    // Must configure mocks before making any method calls.
    for( unsigned int deviceIndex : devices )
    {
        EXPECT_CALL( resLoader, loadResource( deviceIndex, _, startPage, NotNull() ) )
           .WillOnce( DoAll( SetArgPointee<3>( nullptr ), Return( true ) ) );
    }

    for( unsigned int deviceIndex : devices )
    {
        bool isResident1{true};
        bool isResident2{};
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );

        // Launch the kernel, which requests a page and returns a boolean indicating whether it's
        // resident.  The helper function processes any requests.
        unsigned int pageId = startPage + deviceIndex;
        const int numFilled1 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident1 );
        // Launch the kernel again.  The page should now be resident.
        const int numFilled2 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident2 );

        EXPECT_EQ( 1, numFilled1 );
        EXPECT_FALSE( isResident1 );
        EXPECT_EQ( 0, numFilled2 );
        EXPECT_TRUE( isResident2 );
    }
}

TEST_F( TestDemandLoaderResident, TestDeferredResourceRequest )
{
    // TODO: this fails with multiple GPUs under both Windows and Linux.
    //const std::vector<unsigned int> devices = m_loader->getDevices();
    const unsigned int devices[] = { 0 };
    using namespace testing;
    StrictMock<MockResourceLoader> resLoader;
    const unsigned int             numPages = 256;
    const unsigned int startPage = m_loader->createResource( numPages, StrictMock<MockResourceLoader>::callback, &resLoader );
    // Must configure mocks before making any method calls.
    for( unsigned int deviceIndex : devices )
    {
        EXPECT_CALL( resLoader, loadResource( deviceIndex, _, startPage, NotNull() ) )
            .WillOnce( Return( false ) )
            .WillOnce( DoAll( SetArgPointee<3>( nullptr ), Return( true ) ) );
    }

    for( unsigned int deviceIndex : devices )
    {
        const unsigned int pageId = startPage + deviceIndex;
        bool isResident1{true};
        bool isResident2{true};
        bool isResident3{};
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );

        const int numFilled1 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident1 ); // request deferred
        const int numFilled2 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident2 ); // request fulfilled
        const int numFilled3 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident3 ); // resource already loaded

        EXPECT_EQ( 1, numFilled1 );
        EXPECT_FALSE( isResident1 );
        EXPECT_EQ( 1, numFilled2 );
        EXPECT_FALSE( isResident2 );
        EXPECT_EQ( 0, numFilled3 );
        EXPECT_TRUE( isResident3 );
    }
}

TEST_F( TestDemandLoader, TestTextureVariants )
{
    unsigned int deviceIndex = 0;

    // Make first texture
    TextureDescriptor  texDesc1 = m_descriptor;
    DemandTextureImpl* texture1 = m_loader->getTexture( m_loader->createTexture( m_imageSource, texDesc1 ).getId() );
    texture1->open();
    texture1->init( deviceIndex );

    // Make second texture with different descriptor
    TextureDescriptor texDesc2  = m_descriptor;
    texDesc2.addressMode[0]     = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc2.addressMode[1]     = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc2.filterMode         = CU_TR_FILTER_MODE_POINT;
    texDesc2.mipmapFilterMode   = CU_TR_FILTER_MODE_POINT;
    DemandTextureImpl* texture2 = m_loader->getTexture( m_loader->createTexture( m_imageSource, texDesc2 ).getId() );
    texture2->open();
    texture2->init( deviceIndex );

    // The image sources should be the same, but the texture id's different
    EXPECT_EQ( texture1->getInfo(), texture2->getInfo() );
    EXPECT_NE( texture1->getId(), texture2->getId() );

    // The texture descriptors should be what the textures were constructed with
    EXPECT_EQ( texture1->getDescriptor(), texDesc1 );
    EXPECT_EQ( texture2->getDescriptor(), texDesc2 );

    // Texture1 should have a request handler, but texture2 should not (uses the same one)
    EXPECT_TRUE( texture1->getRequestHandler() != nullptr );
    EXPECT_TRUE( texture2->getRequestHandler() == nullptr );

    // The textures should use the same demand load pages
    EXPECT_EQ( texture1->getSampler().startPage, texture2->getSampler().startPage );
    EXPECT_EQ( texture1->getSampler().numPages, texture2->getSampler().numPages ); 
}
