//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include "DemandLoaderTestKernels.h"

#include <OptiXToolkit/DemandLoading/DemandPageLoader.h>
#include <OptiXToolkit/DemandLoading/RequestProcessor.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <iostream>

using namespace testing;

namespace {

class MockRequestProcessor : public demandLoading::RequestProcessor
{
public:
  MOCK_METHOD( void, addRequests, ( CUstream stream, unsigned int id, const unsigned int* pageIds, unsigned int numPageIds ) );
  MOCK_METHOD( void, stop, () );
};

class DemandPageLoaderTest : public Test
{
  public:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
        m_deviceIndex = 0;
        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0 ) );
        OTK_ERROR_CHECK( cudaMalloc( &m_devIsResident, sizeof( bool ) ) );
        OTK_ERROR_CHECK( cudaMalloc( &m_devPageTableEntry, sizeof( unsigned long long ) ) );
        m_loader = createDemandPageLoader( &m_processor, demandLoading::Options{} );
    }

    void TearDown() override
    {
        destroyDemandPageLoader( m_loader );
        OTK_ERROR_CHECK( cudaFree( m_devPageTableEntry ) );
        OTK_ERROR_CHECK( cudaFree( m_devIsResident ) );
        OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
    }

  protected:
    void setIsResident( bool value )
    {
        OTK_ERROR_CHECK( cudaMemcpy( m_devIsResident, &value, sizeof( value ), cudaMemcpyHostToDevice ) );
    }
    bool getIsResident() const
    {
        bool value;
        OTK_ERROR_CHECK( cudaMemcpy( &value, m_devIsResident, sizeof( value ), cudaMemcpyDeviceToHost ) );
        return value;
    }
    bool launchAndRequestPage( unsigned int pageId )
    {
        const bool supported = m_loader->pushMappings( m_stream, m_context );
        launchPageRequester( m_stream, m_context, pageId, static_cast<bool*>( m_devIsResident ),
                             static_cast<unsigned long long*>( m_devPageTableEntry ) );
        m_loader->pullRequests( m_stream, m_context, m_pullId++ );
        return supported;
    }

    CUstream                         m_stream{};
    StrictMock<MockRequestProcessor> m_processor;
    demandLoading::DemandPageLoader* m_loader{};
    unsigned int                     m_deviceIndex{};
    void*                            m_devIsResident{};
    void*                            m_devPageTableEntry{};
    demandLoading::DeviceContext     m_context{};
    unsigned int                     m_pullId{};
};

}  // namespace

TEST_F( DemandPageLoaderTest, create_destroy )
{
    // create done in SetUp, destroy done in TearDown
}

TEST_F( DemandPageLoaderTest, request_non_resident_page )
{
    unsigned int actualRequestedPage{};
    EXPECT_CALL( m_processor, addRequests( m_stream, _, NotNull(), 1 ) ).WillOnce( SaveArgPointee<2>( &actualRequestedPage ) );
    const unsigned int NUM_PAGES     = 10;
    const unsigned int startPage     = m_loader->allocatePages( NUM_PAGES, true );
    const unsigned int requestedPage = startPage + NUM_PAGES / 2;
    setIsResident( true );

    const bool supported = launchAndRequestPage( requestedPage );

    EXPECT_TRUE( supported );
    EXPECT_FALSE( getIsResident() );
    EXPECT_EQ( requestedPage, actualRequestedPage );
}

TEST_F( DemandPageLoaderTest, request_resident_page )
{
    unsigned int actualRequestedPage{};
    EXPECT_CALL( m_processor, addRequests( m_stream, 0, NotNull(), 1 ) ).WillOnce( SaveArgPointee<2>( &actualRequestedPage ) );
    EXPECT_CALL( m_processor, addRequests( m_stream, 1, NotNull(), 0 ) );
    const unsigned int NUM_PAGES     = 10;
    const unsigned int startPage     = m_loader->allocatePages( NUM_PAGES, true );
    const unsigned int requestedPage = startPage + NUM_PAGES / 2;

    const bool supported = launchAndRequestPage( requestedPage );
    setIsResident( false );
    m_loader->setPageTableEntry( requestedPage, true, nullptr );
    launchAndRequestPage( requestedPage );

    EXPECT_TRUE( supported );
    EXPECT_TRUE( getIsResident() );
    EXPECT_EQ( requestedPage, actualRequestedPage );
}
