//
//  Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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

#include "DemandLoaderImpl.h"
#include "DemandLoaderTestKernels.h"
#include "Util/Exception.h"

#include <ImageSource/CheckerBoardImage.h>

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
        DEMAND_CUDA_CHECK( cudaMalloc( &devContext, sizeof( DeviceContext ) ) );
        DEMAND_CUDA_CHECK( cudaMemcpy( devContext, &context, sizeof( DeviceContext ), cudaMemcpyHostToDevice ) );

        // Call the given function to launch the kernel.
        launchFunction( context );

        // Process requests.
        Ticket ticket = m_loader->processRequests( deviceIndex, stream, context );
        ticket.wait();

        DEMAND_CUDA_CHECK( cudaFree( devContext ) );

        return ticket.numTasksTotal();
    }

  protected:
    std::vector<CUstream>        m_streams;
    DemandLoaderImpl*            m_loader;
    std::shared_ptr<ImageSource> m_imageSource;
    TextureDescriptor            m_descriptor;
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

TEST_F( TestDemandLoader, TestSamplerRequest )
{
    const DemandTexture& texture = m_loader->createTexture( m_imageSource, m_descriptor );

    // TODO: this fails with multiple GPUs under both Windows and Linux.
    // for( unsigned int deviceIndex : m_loader->getDevices() )
    unsigned int deviceIndex = 0;
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
        CUstream stream = m_streams[deviceIndex];

        // Allocate device memory for kernel output.
        bool* devIsResident;
        DEMAND_CUDA_CHECK( cudaMalloc( &devIsResident, sizeof( bool ) ) );

        // Launch the kernel, which requests the texture sampler and returns a boolean indicating whether it's resident.
        // The helper function processes any requests.
        unsigned int pageId = texture.getId();
        int numFilled = launchKernel( deviceIndex, stream, [stream, pageId, devIsResident]( const DeviceContext& context ) {
            launchPageRequester( stream, context, pageId, devIsResident );
        } );
        EXPECT_EQ( 1, numFilled );

        // Copy isResident result to host.
        DEMAND_CUDA_CHECK( cuStreamSynchronize( stream ) );
        bool isResident;
        DEMAND_CUDA_CHECK( cudaMemcpy( &isResident, devIsResident, sizeof( bool ), cudaMemcpyDeviceToHost ) );
        EXPECT_FALSE( isResident );

        // Launch the kernel again.  The sampler should now be resident.
        numFilled = launchKernel( deviceIndex, stream, [stream, pageId, devIsResident]( const DeviceContext& context ) {
            launchPageRequester( stream, context, pageId, devIsResident );
        } );
        EXPECT_EQ( 0, numFilled );

        DEMAND_CUDA_CHECK( cuStreamSynchronize( stream ) );
        DEMAND_CUDA_CHECK( cudaMemcpy( &isResident, devIsResident, sizeof( bool ), cudaMemcpyDeviceToHost ) );
        DEMAND_CUDA_CHECK( cudaFree( devIsResident ) );
        EXPECT_TRUE( isResident );
    }
}

void* callback( unsigned int deviceIndex, CUstream stream, unsigned int pageIndex )
{
    return nullptr;
}

TEST_F( TestDemandLoader, TestResourceRequest )
{
    const unsigned int numPages  = 256;
    const unsigned int startPage = m_loader->createResource( numPages, callback );

    // TODO: this fails with multiple GPUs under both Windows and Linux.
    // for( unsigned int deviceIndex : m_loader->getDevices() )
    unsigned int deviceIndex = 0;
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
        CUstream stream = m_streams[deviceIndex];

        // Allocate device memory for kernel output.
        bool* devIsResident;
        DEMAND_CUDA_CHECK( cudaMalloc( &devIsResident, sizeof( bool ) ) );

        // Launch the kernel, which requests a page and returns a boolean indicating whether it's
        // resident.  The helper function processes any requests.
        unsigned int pageId = startPage + deviceIndex;
        int numFilled = launchKernel( deviceIndex, stream, [stream, pageId, devIsResident]( const DeviceContext& context ) {
            launchPageRequester( stream, context, pageId, devIsResident );
        } );
        EXPECT_EQ( 1, numFilled );

        // Copy isResident result to host.
        DEMAND_CUDA_CHECK( cuStreamSynchronize( stream ) );
        bool isResident;
        DEMAND_CUDA_CHECK( cudaMemcpy( &isResident, devIsResident, sizeof( bool ), cudaMemcpyDeviceToHost ) );
        EXPECT_FALSE( isResident );

        // Launch the kernel again.  The page should now be resident.
        numFilled = launchKernel( deviceIndex, stream, [stream, pageId, devIsResident]( const DeviceContext& context ) {
            launchPageRequester( stream, context, pageId, devIsResident );
        } );
        EXPECT_EQ( 0, numFilled );

        DEMAND_CUDA_CHECK( cuStreamSynchronize( stream ) );
        DEMAND_CUDA_CHECK( cudaMemcpy( &isResident, devIsResident, sizeof( bool ), cudaMemcpyDeviceToHost ) );
        DEMAND_CUDA_CHECK( cudaFree( devIsResident ) );
        EXPECT_TRUE( isResident );
    }
}
