// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// Verifies that a request handler reports a failed request through the existing logger (DL_LOG)
// instead of letting the exception terminate the worker thread.  Without that handling the worker
// thread dies before calling Ticket::notify(), so Ticket::wait() would never return -- hence the
// test asserts both that wait() completes and that the error was captured by the logger.

#include "TestSparseTexture.h"  // launchTextureDrawKernel
#include "DemandLoaderImpl.h"

#include <OptiXToolkit/DemandLoading/DemandLoadLogger.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <gtest/gtest.h>

#include <cuda.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace demandLoading;
using namespace imageSource;

namespace {

// Thread-safe capturing log sink with static lifetime.  DemandLoadLogCallback is a raw function
// pointer (no closure) invoked from worker threads, and DemandLoadLogger::setLogFunction is one-shot,
// so the storage must be static + mutex-guarded and the installation must happen exactly once.
std::mutex                               g_logMutex;
std::vector<std::pair<int, std::string>> g_logMessages;

void capturingLogCallback( int level, const char* message )
{
    std::lock_guard<std::mutex> lock( g_logMutex );
    g_logMessages.emplace_back( level, message ? message : "" );
}

void installLoggerOnce()
{
    // Level 0 captures the errors the handlers report (DL_LOG level 0) while suppressing the
    // level 4-5 informational traces.  The function-local static guarantees a single install even if
    // the suite runs multiple test cases or is repeated.
    static bool installed = [] {
        DemandLoadLogger::setLogFunction( capturingLogCallback, 0 );
        return true;
    }();
    (void)installed;
}

void clearLog()
{
    std::lock_guard<std::mutex> lock( g_logMutex );
    g_logMessages.clear();
}

bool logContains( const std::string& substr, int& levelOut )
{
    std::lock_guard<std::mutex> lock( g_logMutex );
    for( const auto& entry : g_logMessages )
    {
        if( entry.second.find( substr ) != std::string::npos )
        {
            levelOut = entry.first;
            return true;
        }
    }
    return false;
}

// An ImageSource whose open() always throws, simulating a missing or unreadable image file.
class ThrowingImageSource : public ImageSourceBase
{
  public:
    void open( TextureInfo* /*info*/ ) override { throw std::runtime_error( "simulated open failure" ); }
    void close() override {}
    bool isOpen() const override { return false; }
    const TextureInfo& getInfo() const override { return m_info; }
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }
    bool readTile( char*, unsigned int, const Tile&, CUstream ) override { return false; }
    bool readMipLevel( char*, unsigned int, unsigned int, unsigned int, CUstream ) override { return false; }
    bool readBaseColor( float4& ) override { return false; }

  private:
    TextureInfo m_info{};
};

}  // namespace

class TestRequestHandlerLogging : public testing::Test
{
  public:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cuInit( 0 ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );

        m_deviceIndex = getFirstSparseTextureDevice();
        if( m_deviceIndex == MAX_DEVICES )
            return;

        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaStreamCreate( &m_stream ) );

        Options options{};
        m_loader.reset( new DemandLoaderImpl( options ) );

        installLoggerOnce();
        clearLog();
    }

    void TearDown() override
    {
        m_loader.reset();
        if( m_stream )
            OTK_ERROR_CHECK( cudaStreamDestroy( m_stream ) );
    }

  protected:
    unsigned int                      m_deviceIndex = 0;
    CUstream                          m_stream{};
    std::unique_ptr<DemandLoaderImpl> m_loader;
};

// A sampler request whose image fails to open must be reported through the logger (at level 0) and
// must not terminate the worker thread -- the request cycle still completes (ticket.wait() returns).
TEST_F( TestRequestHandlerLogging, FailedSamplerRequestIsLogged )
{
    if( m_deviceIndex == MAX_DEVICES )
        GTEST_SKIP() << "No device supports sparse textures";

    TextureDescriptor desc{};
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 1;

    std::shared_ptr<ImageSource> image = std::make_shared<ThrowingImageSource>();
    const DemandTexture&         texture   = m_loader->createTexture( image, desc );
    const unsigned int           textureId = texture.getId();

    const int outWidth  = 16;
    const int outHeight = 16;
    float4*   devOutput;
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &devOutput ), outWidth * outHeight * sizeof( float4 ) ) );

    // Drive a request for the texture's sampler; the worker calls open(), which throws.
    for( int i = 0; i < 2; ++i )
    {
        DeviceContext context;
        m_loader->launchPrepare( m_stream, context );
        launchTextureDrawKernel( m_stream, context, textureId, devOutput, outWidth, outHeight );
        Ticket ticket = m_loader->processRequests( m_stream, context );
        ticket.wait();  // must return: the worker logged the error rather than dying before notify()
    }

    OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( devOutput ) ) );

    int level = -1;
    EXPECT_TRUE( logContains( "simulated open failure", level ) );
    EXPECT_EQ( 0, level );
}
