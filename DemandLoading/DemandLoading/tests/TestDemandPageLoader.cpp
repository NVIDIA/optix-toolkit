// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandLoaderTestKernels.h"

#include <OptiXToolkit/DemandLoading/DemandPageLoader.h>
#include <OptiXToolkit/DemandLoading/RequestProcessor.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

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
    m_loader->setPageTableEntry( requestedPage, true, 0ULL );
    launchAndRequestPage( requestedPage );

    EXPECT_TRUE( supported );
    EXPECT_TRUE( getIsResident() );
    EXPECT_EQ( requestedPage, actualRequestedPage );
}
