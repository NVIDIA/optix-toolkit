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

#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
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
        unsigned int numDevices = getCudaDeviceCount();
        m_streams.resize( numDevices );
        m_loaders.resize( numDevices );
        for( unsigned int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
        {
            // Initialize CUDA
            OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
            OTK_ERROR_CHECK( cudaFree( nullptr ) );

            // Create a stream per device.
            OTK_ERROR_CHECK( cuStreamCreate( &m_streams[deviceIndex], 0 ) );

            // Create DemandLoader per device
            m_loaders[deviceIndex] = dynamic_cast<DemandLoaderImpl*>( createDemandLoader( Options() ) );
        }

        // Create ImageSource
        m_imageSource.reset( new CheckerBoardImage( 2048, 2048, 32 /*squaresPerSide*/, true /*useMipmaps*/ ) );

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
            OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
            OTK_ERROR_CHECK( cuStreamDestroy( m_streams[deviceIndex] ) );
            destroyDemandLoader( m_loaders[deviceIndex] );
            m_loaders[deviceIndex] = nullptr;
        }
    }


    int launchKernel( DemandLoaderImpl* loader, CUstream stream, const std::function<void( const DeviceContext& )>& launchFunction )
    {
        // Prepare for launch, obtaining host-side DeviceContext.
        DeviceContext context;
        bool          ok = loader->launchPrepare( stream, context );
        EXPECT_TRUE( ok );

        // Call the given function to launch the kernel.  The device context will be copied to
        // device memory by the callee (i.e. by passing it by value in a triple-chevron launch).
        launchFunction( context );

        // Process requests.
        Ticket ticket = loader->processRequests( stream, context );
        ticket.wait();

        return ticket.numTasksTotal();
    }

  protected:
    std::vector<CUstream>          m_streams;
    std::vector<DemandLoaderImpl*> m_loaders;
    std::shared_ptr<ImageSource>   m_imageSource;
    TextureDescriptor              m_descriptor{};
};

TEST_F( TestDemandLoader, TestCreateDestroy )
{
    // No operations
}

TEST_F( TestDemandLoader, TestCreateTexture )
{
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
    m_loaders[0]->createTexture( m_imageSource, m_descriptor );
    // The texture is opaque, so we can't really validate it.
}

class TestDemandLoaderResident : public TestDemandLoader
{
  public:
    void SetUp() override
    {
        TestDemandLoader::SetUp();

        // Allocate per-device memory for kernel output.
        size_t numDevices = m_streams.size();
        m_devIsResident.resize( numDevices );
        m_devPageTableEntry.resize( numDevices );
        for( size_t deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
        {
            OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
            OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &m_devIsResident[deviceIndex] ), sizeof( bool ) ) );
            OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &m_devPageTableEntry[deviceIndex] ),
                                           sizeof( unsigned long long ) ) );
        }
    }

    void TearDown() override
    {
        size_t numDevices = m_devIsResident.size();
        for( size_t deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
        {
            OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
            OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( m_devIsResident[deviceIndex] ) ) );
            OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( m_devPageTableEntry[deviceIndex] ) ) );
        }
        TestDemandLoader::TearDown();
    }

  protected:
    int launchKernelAndSynchronize( unsigned int deviceIndex, unsigned int pageId, bool* isResident )
    {
        DemandLoaderImpl*   loader            = m_loaders[deviceIndex];
        CUstream            stream            = m_streams[deviceIndex];
        bool*               devIsResident     = m_devIsResident[deviceIndex];
        unsigned long long* devPageTableEntry = m_devPageTableEntry[deviceIndex];
        const int           numFilled =
            launchKernel( loader, stream, [stream, pageId, devIsResident, devPageTableEntry]( const DeviceContext& context ) {
                launchPageRequester( stream, context, pageId, devIsResident, devPageTableEntry );
            } );
        // Copy isResident result to host.
        OTK_ERROR_CHECK( cuStreamSynchronize( stream ) );
        OTK_ERROR_CHECK( cudaMemcpy( isResident, devIsResident, sizeof( bool ), cudaMemcpyDeviceToHost ) );
        return numFilled;
    }

    std::vector<bool*>               m_devIsResident;
    std::vector<unsigned long long*> m_devPageTableEntry;
};

TEST_F( TestDemandLoaderResident, TestSamplerRequest )
{
    // TODO: this fails with multiple GPUs under both Windows and Linux.
    //for( unsigned int deviceIndex : getSparseTextureDevices() )
    const std::vector<unsigned int> devices = getSparseTextureDevices();
    if( devices.empty() )
        return;

    const unsigned int deviceIndex = devices[0];
    {
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        const DemandTexture& texture = m_loaders[deviceIndex]->createTexture( m_imageSource, m_descriptor );
        const unsigned int   pageId  = texture.getId();

        bool isResident1{ true };
        bool isResident2{};

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
    MOCK_METHOD( bool, loadResource, ( CUstream stream, unsigned int pageIndex, void** pageTableEntry ) );

    static bool callback( CUstream stream, unsigned int pageIndex, void* context, void** pageTableEntry )
    {
        return static_cast<MockResourceLoader*>( context )->loadResource( stream, pageIndex, pageTableEntry );
    }
};

TEST_F( TestDemandLoaderResident, TestResourceRequest )
{
    // TODO: this fails with multiple GPUs under both Windows and Linux.
    //const std::vector<unsigned int> devices = m_loader->getDevices();
    const unsigned int devices[] = { 0 };
    using namespace testing;
    const unsigned int             numPages = 256;
    StrictMock<MockResourceLoader> resLoader;
    unsigned int startPage = 0; 
    for( unsigned int deviceIndex : devices )
    {
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        startPage = m_loaders[deviceIndex]->createResource( numPages, StrictMock<MockResourceLoader>::callback, &resLoader );
    }

    // Must configure mocks before making any method calls.
    for( unsigned int deviceIndex : devices )
    {
        (void)deviceIndex; // silence unused variable warning
        EXPECT_CALL( resLoader, loadResource( _, startPage, NotNull() ) ).WillOnce( DoAll( SetArgPointee<2>( nullptr ), Return( true ) ) );
    }

    for( unsigned int deviceIndex : devices )
    {
        bool isResident1{ true };
        bool isResident2{};
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );

        // Launch the kernel, which requests a page and returns a boolean indicating whether it's
        // resident.  The helper function processes any requests.
        unsigned int pageId     = startPage + deviceIndex;
        const int    numFilled1 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident1 );
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
    const unsigned int numPages = 256;
    unsigned int startPage = 0;
    for( unsigned int deviceIndex : devices )
    {
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        startPage = m_loaders[deviceIndex]->createResource( numPages, StrictMock<MockResourceLoader>::callback, &resLoader );
    }
    // Must configure mocks before making any method calls.
    for( unsigned int deviceIndex : devices )
    {
        (void)deviceIndex; // silence unused variable warning
        EXPECT_CALL( resLoader, loadResource( _, startPage, NotNull() ) )
            .WillOnce( Return( false ) )
            .WillOnce( DoAll( SetArgPointee<2>( nullptr ), Return( true ) ) );
    }

    for( unsigned int deviceIndex : devices )
    {
        const unsigned int pageId = startPage + deviceIndex;
        bool               isResident1{ true };
        bool               isResident2{ true };
        bool               isResident3{};
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );

        const int numFilled1 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident1 );  // request deferred
        const int numFilled2 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident2 );  // request fulfilled
        const int numFilled3 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident3 );  // resource already loaded

        EXPECT_EQ( 1, numFilled1 );
        EXPECT_FALSE( isResident1 );
        EXPECT_EQ( 1, numFilled2 );
        EXPECT_FALSE( isResident2 );
        EXPECT_EQ( 0, numFilled3 );
        EXPECT_TRUE( isResident3 );
    }
}

TEST_F( TestDemandLoaderResident, TestResourceResident )
{
    const std::vector<unsigned int> devices = getSparseTextureDevices();
    if( devices.empty() )
        return;
    const unsigned int deviceIndex = devices[0];
    OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );

    const ResourceCallback callback = []( CUstream /*stream*/, unsigned int /*pageIndex*/, void* /*context*/,
                                          void** /*pageTableEntry*/ ) { return true; };

    // Create a resource with a single page.
    const unsigned int pageId = m_loaders[deviceIndex]->createResource( 1, callback, nullptr );

    bool isResident1{true};
    bool isResident2{};

    // Launch the kernel, which requests the resource and returns a boolean indicating whether it's resident.
    // The helper function processes any requests.
    const int numFilled1 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident1 );
    // Launch the kernel again.  The resource should now be resident.
    const int numFilled2 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident2 );

    EXPECT_EQ( 1, numFilled1 );
    EXPECT_FALSE( isResident1 );
    EXPECT_EQ( 0, numFilled2 );
    EXPECT_TRUE( isResident2 );

    // Invalidate the resource and confirm that it's requested on a subsequent launch.
    m_loaders[deviceIndex]->invalidatePage( pageId );
    const int numFilled3 = launchKernelAndSynchronize( deviceIndex, pageId, &isResident2 );
    bool      isResident3{};
    EXPECT_EQ( 1, numFilled3 );
    EXPECT_FALSE( isResident3 );
}


TEST_F( TestDemandLoader, TestTextureVariants )
{
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );

    // Make first texture
    TextureDescriptor  texDesc1 = m_descriptor;
    DemandTextureImpl* texture1 = m_loaders[0]->getTexture( m_loaders[0]->createTexture( m_imageSource, texDesc1 ).getId() );
    texture1->open();
    texture1->init();

    // Make second texture with different descriptor
    TextureDescriptor texDesc2  = m_descriptor;
    texDesc2.addressMode[0]     = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc2.addressMode[1]     = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc2.filterMode         = CU_TR_FILTER_MODE_POINT;
    texDesc2.mipmapFilterMode   = CU_TR_FILTER_MODE_POINT;
    DemandTextureImpl* texture2 = m_loaders[0]->getTexture( m_loaders[0]->createTexture( m_imageSource, texDesc2 ).getId() );
    texture2->open();
    texture2->init();

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

    class TestDemandLoaderBatches : public TestDemandLoader
{
  protected:
    using PageTableEntry = unsigned long long;
    using PageTable      = std::vector<PageTableEntry>;

    void        validatePageTableEntries( const PageTable& pageTableEntries,
                                          unsigned int     currentPage,
                                          unsigned int     endPage,
                                          unsigned int     batchSize,
                                          unsigned int     numLaunches );
    static bool loadResourceCallback( CUstream stream, unsigned int pageId, void* context, void** pageTableEntry )
    {
        return static_cast<TestDemandLoaderBatches*>( context )->loadResource( stream, pageId, pageTableEntry );
    }
    bool loadResource( CUstream stream, unsigned int pageId, void** pageTableEntry );
    void testBatch( unsigned int deviceIndex, bool testAbort );

    std::atomic<int> m_numRequestsProcessed{ 0 };
};

void TestDemandLoaderBatches::validatePageTableEntries( const PageTable& pageTableEntries,
                                                        unsigned int     currentPage,
                                                        unsigned int     endPage,
                                                        unsigned int     batchSize,
                                                        unsigned int     numLaunches )
{
    for( unsigned int i = 0; i < batchSize; ++i )
    {
        if( i >= endPage )
            break;
        ASSERT_EQ( pageTableEntries[i], currentPage ) << "Launch " << numLaunches << " Mismatch pageTable[" << i << "]";
        ++currentPage;
    }
}

static void* toPageEntry( unsigned int value )
{
    void*         result{};
    std::intptr_t source = value;
    std::memcpy( &result, &source, sizeof( result ) );
    return result;
}

bool TestDemandLoaderBatches::loadResource( CUstream /*stream*/, unsigned int pageId, void** pageTableEntry )
{
    ++m_numRequestsProcessed;
    *pageTableEntry = toPageEntry( pageId );
    return true;
}

void TestDemandLoaderBatches::testBatch( unsigned int deviceIndex, bool testAbort )
{
    DemandLoaderImpl* loader = m_loaders[deviceIndex];
    CUstream stream = m_streams[deviceIndex];

    // Create a resource, using the given callback to handle page requests.
    const unsigned int numPages  = 128;
    unsigned int       startPage = loader->createResource( numPages, loadResourceCallback, this );

    // Process all the pages of the resource in batches.
    const unsigned int batchSize   = 32;
    unsigned int       numLaunches = 0;

    void* devPageTableEntries{};
    cudaMalloc( &devPageTableEntries, sizeof( PageTableEntry ) * numPages );

    std::vector<PageTableEntry> pageTableEntries;

    for( unsigned int currentPage = startPage; currentPage < startPage + numPages; )
    {
        // Prepare for launch, obtaining DeviceContext.
        DeviceContext context;
        loader->launchPrepare( stream, context );

        // Launch the kernel.
        launchPageBatchRequester( stream, context, currentPage, currentPage + batchSize,
                                  static_cast<PageTableEntry*>( devPageTableEntries ) );
        ++numLaunches;

        // Initiate request processing, which returns a Ticket.
        Ticket ticket = loader->processRequests( stream, context );

        // Test abort functionality, which halts request processing.  The request processor
        // should automatically restart on the next iteration.
        if( testAbort )
        {
            loader->abort();
            cudaDeviceSynchronize();
            currentPage += batchSize;
        }

        cudaDeviceSynchronize();

        // Wait for any page requests to be processed.
        ticket.wait();

        if( ticket.numTasksTotal() == 0 )
        {
            // Validate page table entries.
            pageTableEntries.resize( batchSize );
            cudaMemcpy( pageTableEntries.data(), devPageTableEntries, batchSize * sizeof( PageTableEntry ), cudaMemcpyDeviceToHost );
            validatePageTableEntries( pageTableEntries, currentPage, startPage + numPages, batchSize, numLaunches );

            // Advance the loop counter only when there were no page requests.
            currentPage += batchSize;
        }
    }

    // Clean up
    cudaFree( devPageTableEntries );
}

TEST_F( TestDemandLoaderBatches, LoopTest )
{
    for( unsigned int deviceIndex = 0; deviceIndex < m_loaders.size(); ++deviceIndex )
    {
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        for( int i = 0; i < 4; ++i )
            testBatch( deviceIndex, /*testAbort=*/false );
    }
}

TEST_F( TestDemandLoaderBatches, TestAbort )
{
    for( unsigned int deviceIndex = 0; deviceIndex < m_loaders.size(); ++deviceIndex )
    {
        OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
        testBatch( deviceIndex, /*testAbort=*/true );
    }
}
