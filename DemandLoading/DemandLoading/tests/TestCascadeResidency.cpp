// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Tests for cascading texture residency.

#include "DemandLoaderImpl.h"
#include "DemandLoaderTestKernels.h"
#include "PageTableManager.h"

#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TextureCascade.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/CheckerBoardImage.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

using namespace demandLoading;
using namespace imageSource;

// ---------------------------------------------------------------------------
// TestCascadePageAddressing
// Verify that the device-side cascade page calculation
//   cascadeStartIndex = pageTable.capacity
//   cascadePage = cascadeStartIndex + textureId * NUM_CASCADES + cascadeLevel
// matches the host-side reservation via PageTableManager::reserveUnbackedPages.
// ---------------------------------------------------------------------------

class DummyCascadeHandler : public RequestHandler
{
  public:
    void fillRequest( CUstream /*stream*/, unsigned int /*pageId*/ ) override {}
};

class TestCascadePageAddressing : public testing::Test
{
  protected:
    // Mirror the default option values
    static constexpr unsigned int numPages            = 64u * 1024u * 1024u;
    static constexpr unsigned int numPageTableEntries  = 1024u * 1024u;
    static constexpr unsigned int maxTextures          = 256u * 1024u;
};

TEST_F( TestCascadePageAddressing, cascadeStartMatchesPageTableCapacity )
{
    // Host: allocate sampler pages (backed), then cascade pages (unbacked)
    PageTableManager    mgr( numPages, numPageTableEntries );
    DummyCascadeHandler samplerHandler;
    DummyCascadeHandler cascadeHandler;

    mgr.reserveBackedPages( maxTextures * NUM_PAGES_PER_TEXTURE, &samplerHandler );
    unsigned int cascadeStartPage = mgr.reserveUnbackedPages( NUM_CASCADES * maxTextures, &cascadeHandler );

    // Device uses pageTable.capacity (== numPageTableEntries) as cascadeStartIndex
    unsigned int deviceCascadeStart = numPageTableEntries;

    EXPECT_EQ( deviceCascadeStart, cascadeStartPage )
        << "Device cascadeStartIndex must equal host cascadeStartPage";
}

TEST_F( TestCascadePageAddressing, cascadePageForTexture )
{
    PageTableManager    mgr( numPages, numPageTableEntries );
    DummyCascadeHandler handler;

    mgr.reserveBackedPages( maxTextures * NUM_PAGES_PER_TEXTURE, &handler );
    unsigned int cascadeStartPage = mgr.reserveUnbackedPages( NUM_CASCADES * maxTextures, &handler );

    // Verify that host handler lookup matches the device page calculation
    const unsigned int textureId    = 42;
    const unsigned int cascadeLevel = 3;
    unsigned int       devicePage   = cascadeStartPage + textureId * NUM_CASCADES + cascadeLevel;

    EXPECT_EQ( &handler, mgr.getRequestHandler( devicePage ) );
}

// ---------------------------------------------------------------------------
// TestCascadeResidency
// Verify that a texture becomes resident when cascading texture sizes are
// enabled.  This exercises the device-side requestCascade() code path via
// demand-loading tex2DGrad, ensuring that a NULL sampler does not generate
// cascade requests that starve the sampler page from loading.
// ---------------------------------------------------------------------------

class TestCascadeResidency : public testing::Test
{
  public:
    void SetUp() override
    {
        m_deviceIndex = getFirstSparseTextureDevice();
        if( m_deviceIndex == demandLoading::MAX_DEVICES )
            return;

        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
        OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0 ) );

        // Enable cascading texture sizes
        Options options{};
        options.useCascadingTextureSizes = true;
        m_loader = dynamic_cast<DemandLoaderImpl*>( createDemandLoader( options ) );

        // Allocate device memory for kernel output
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &m_devIsResident ), sizeof( bool ) ) );
    }

    void TearDown() override
    {
        if( m_deviceIndex == demandLoading::MAX_DEVICES )
            return;

        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( m_devIsResident ) ) );
        OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
        destroyDemandLoader( m_loader );
    }

  protected:
    // Launch a kernel that samples the texture and process any resulting requests.
    // Returns the number of tasks processed.
    int launchAndProcess( unsigned int textureId, float2 ddx, float2 ddy, bool* isResident )
    {
        DeviceContext context;
        bool ok = m_loader->launchPrepare( m_stream, context );
        EXPECT_TRUE( ok );

        launchTextureSampler( m_stream, context, textureId, ddx, ddy, m_devIsResident );

        Ticket ticket = m_loader->processRequests( m_stream, context );
        ticket.wait();

        OTK_ERROR_CHECK( cudaMemcpy( isResident, m_devIsResident, sizeof( bool ), cudaMemcpyDeviceToHost ) );
        return ticket.numTasksTotal();
    }

    unsigned int      m_deviceIndex = demandLoading::MAX_DEVICES;
    CUstream          m_stream{};
    DemandLoaderImpl* m_loader = nullptr;
    bool*             m_devIsResident = nullptr;
};

TEST_F( TestCascadeResidency, samplerBecomesResident )
{
    if( m_deviceIndex == demandLoading::MAX_DEVICES )
        return;

    // Create a texture through the demand loader
    auto imageSource = std::make_shared<CheckerBoardImage>( 2048, 2048, 32, true );
    TextureDescriptor desc{};
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.filterMode       = CU_TR_FILTER_MODE_LINEAR;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
    desc.maxAnisotropy    = 16;

    const DemandTexture& texture   = m_loader->createTexture( imageSource, desc );
    const unsigned int   textureId = texture.getId();

    // Use small derivatives (high magnification) — this is the scenario where
    // requestCascade would fire on a NULL sampler before the fix.
    float2 ddx = make_float2( 1.0f / 4096.0f, 0.0f );
    float2 ddy = make_float2( 0.0f, 1.0f / 4096.0f );

    // Iterate launch-process cycles until the texture becomes resident.
    // With cascading enabled, multiple rounds may be needed: load sampler,
    // then cascade to progressively larger sizes.  Before the fix, the
    // cascade request from the NULL-sampler path would knock out the sampler
    // page indefinitely, so the texture would never become resident.
    const int maxLaunches = 20;
    bool      isResident  = false;
    int       numLaunches = 0;

    for( ; numLaunches < maxLaunches && !isResident; ++numLaunches )
    {
        launchAndProcess( textureId, ddx, ddy, &isResident );
    }

    EXPECT_TRUE( isResident ) << "Texture failed to become resident after " << maxLaunches << " launches";
    EXPECT_LT( numLaunches, maxLaunches ) << "Should converge well before the limit";
}
