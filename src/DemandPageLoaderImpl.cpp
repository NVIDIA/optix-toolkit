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

#include "DemandPageLoaderImpl.h"

#include "RequestProcessor.h"
#include "Util/Exception.h"
#include "Util/NVTXProfiling.h"
#include "Util/Stopwatch.h"
#include "Util/TraceFile.h"
#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>

#include <cuda.h>

#include <algorithm>
#include <memory>
#include <set>
#include <utility>

namespace {

unsigned int getCudaDeviceCount()
{
    int numDevices;
    DEMAND_CUDA_CHECK( cuDeviceGetCount( &numDevices ) );
    return static_cast<unsigned int>( numDevices );
}

demandLoading::Options configure( demandLoading::Options options )
{
    // If maxTexMemPerDevice is 0, consider it to be unlimited
    if( options.maxTexMemPerDevice == 0 )
        options.maxTexMemPerDevice = 0xfffffffffffffffful;

    // PagingSystem::pushMappings requires enough capacity to handle all the requested pages.
    if( options.maxFilledPages < options.maxRequestedPages )
        options.maxFilledPages = options.maxRequestedPages;

    // Anticipate at lease one active stream per device.
    options.maxActiveStreams = std::max( getCudaDeviceCount(), options.maxActiveStreams );

    return options;
}

bool supportsSparseTextures( unsigned int deviceIndex )
{
    int sparseSupport = 0;
    DEMAND_CUDA_CHECK( cuDeviceGetAttribute( &sparseSupport, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, deviceIndex ) );

    // Skip devices in TCC mode.  This guards against an "operation not supported" error when
    // querying the recommended allocation granularity via cuMemGetAllocationGranularity.
    int inTccMode = 0;
    DEMAND_CUDA_CHECK( cuDeviceGetAttribute( &inTccMode, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, deviceIndex ) );

    return sparseSupport && !inTccMode;
}

bool tccModeEnabled( unsigned int deviceIndex )
{
    // Skip devices in TCC mode.  This guards against an "operation not supported" error when
    // querying the recommended allocation granularity via cuMemGetAllocationGranularity.
    int inTccMode = 0;
    DEMAND_CUDA_CHECK( cuDeviceGetAttribute( &inTccMode, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, deviceIndex ) );
    return inTccMode;
}

}  // anonymous namespace

namespace demandLoading {

DemandPageLoaderImpl::DemandPageLoaderImpl( RequestProcessor* requestProcessor, const Options& options )
    : DemandPageLoaderImpl( std::make_shared<PageTableManager>( configure( options ).numPages, configure( options ).numPageTableEntries ), requestProcessor, options )
{
    // Determine which devices to use.  Look for devices supporting sparse textures first
    for( unsigned int deviceIndex = 0; deviceIndex < m_numDevices; ++deviceIndex )
    {
        if( m_options.useSparseTextures && supportsSparseTextures( deviceIndex ) && !tccModeEnabled( deviceIndex ) )
            m_devices.push_back( deviceIndex );
    }

    // Fall back to dense textures if no devices supporting sparse textures were found
    if( m_devices.empty() )
    {
        // FIXME: log a warning here that we are falling back to dense textures if m_options.useSparseTextures is true.
        //throw Exception( "No devices that support CUDA sparse textures were found (sm_60+ required)." );
        m_options.useSparseTextures = false;
        for( unsigned int deviceIndex = 0; deviceIndex < m_numDevices; ++deviceIndex )
        {
            if( !tccModeEnabled( deviceIndex ) )
                m_devices.push_back( deviceIndex );
        }
    }

    // Create deviceMemoryManagers and pagingSystems for the devices
    for( unsigned int deviceIndex : m_devices )
    {
        m_deviceMemoryManagers[deviceIndex].reset( new DeviceMemoryManager( deviceIndex, m_options ) );
        m_pagingSystems[deviceIndex].reset( new PagingSystem(
            deviceIndex, m_options, m_deviceMemoryManagers[deviceIndex].get(), &m_pinnedMemoryPool, m_requestProcessor ) );
    }
}

const static unsigned long long PINNED_ALLOC_SIZE = 2 * 1024 * 1024;

DemandPageLoaderImpl::DemandPageLoaderImpl( std::shared_ptr<PageTableManager> pageTableManager,
                                            RequestProcessor*                 requestProcessor,
                                            const Options&                    options )
    : m_options( configure( options ) )
    , m_numDevices( getCudaDeviceCount() )
    , m_deviceMemoryManagers( m_numDevices )
    , m_pagingSystems(m_numDevices)
    , m_pagesToInvalidate( m_numDevices )
    , m_pageTableManager( std::move( pageTableManager ) )
    , m_requestProcessor( requestProcessor )
    , m_pinnedMemoryPool( new PinnedAllocator(), new RingSuballocator( PINNED_ALLOC_SIZE ), PINNED_ALLOC_SIZE, options.maxPinnedMemory )
{
    // Determine which devices to use.  Look for devices supporting sparse textures first
    for( unsigned int deviceIndex = 0; deviceIndex < m_numDevices; ++deviceIndex )
    {
        if( m_options.useSparseTextures && supportsSparseTextures( deviceIndex ) )
            m_devices.push_back( deviceIndex );
    }

    // Fall back to dense textures if no devices supporting sparse textures were found
    if( m_devices.empty() )
    {
        // FIXME: log a warning here that we are falling back to dense textures if m_options.useSparseTextures is true.
        //throw Exception( "No devices that support CUDA sparse textures were found (sm_60+ required)." );
        m_options.useSparseTextures = false;
        for( unsigned int deviceIndex = 0; deviceIndex < m_numDevices; ++deviceIndex )
            m_devices.push_back( deviceIndex );
    }

    // Create deviceMemoryManagers and pagingSystems for the devices
    for( unsigned int deviceIndex : m_devices )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
        m_deviceMemoryManagers[deviceIndex].reset( new DeviceMemoryManager( deviceIndex, m_options ) );
        m_pagingSystems[deviceIndex].reset( new PagingSystem(
            deviceIndex, m_options, m_deviceMemoryManagers[deviceIndex].get(), &m_pinnedMemoryPool, m_requestProcessor ) );
    }
}


unsigned int DemandPageLoaderImpl::createResource( unsigned int numPages, ResourceCallback callback, void* callbackContext )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    // Create a request handler that wraps the callback.  These are individually allocated to avoid
    // dangling pointers in the PageTableManager when the request handler vector is resized.
    m_resourceRequestHandlers.emplace_back( new ResourceRequestHandler( callback, callbackContext, this ) );

    // Reserve virtual address space for the resource, which is associated with the request handler.
    m_pageTableManager->reserveBackedPages( numPages, m_resourceRequestHandlers.back().get() );

    // Return the start page.
    return m_resourceRequestHandlers.back()->getStartPage();

    SCOPED_NVTX_RANGE_FUNCTION_NAME();
}

// Returns false if the device doesn't support sparse textures.
bool DemandPageLoaderImpl::launchPrepare( unsigned int deviceIndex, CUstream stream, DeviceContext& context )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    PagingSystem* pagingSystem = m_pagingSystems.at( deviceIndex ).get();
    if( pagingSystem == nullptr )
        return false;

    // Get DeviceContext from pool and copy it to output parameter.
    {
        // allocate() is not thread safe
        std::unique_lock<std::mutex> lock( m_mutex );
        context = *m_deviceMemoryManagers[deviceIndex]->allocateDeviceContext();
        invalidatePages( deviceIndex, stream, context );
    }
    context.requestIfResident = m_options.evictionActive;

    pagingSystem->pushMappings( context, stream );
    return true;
}

void DemandPageLoaderImpl::invalidatePages( unsigned int deviceIndex, CUstream stream, DeviceContext& context )
{
    // Mutex acquired in caller

    for( InvalidationRange& ir : m_pagesToInvalidate[deviceIndex] )
    {
        m_pagingSystems[deviceIndex]->invalidatePages( ir.startPage, ir.endPage, ir.predicate, context, stream );
        delete ir.predicate;
    }
    m_pagesToInvalidate[deviceIndex].clear();
}


// Process page requests.
Ticket DemandPageLoaderImpl::processRequests( unsigned int deviceIndex, CUstream stream, const DeviceContext& context )
{
    Stopwatch stopwatch;
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Create a Ticket that the caller can use to track request processing.
    Ticket ticket( TicketImpl::create( deviceIndex, stream ) );

    // Pull requests from the device.  This launches a kernel on the given stream to scan the
    // request bits copies the requested page ids to host memory (asynchronously).
    PagingSystem* pagingSystem = m_pagingSystems[deviceIndex].get();
    unsigned int  startPage    = 0;
    unsigned int  endPage      = m_pageTableManager->getHighestUsedPage();
    pagingSystem->pullRequests( context, stream, startPage, endPage, ticket );

    std::unique_lock<std::mutex> lock( m_mutex );
    m_totalProcessingTime += stopwatch.elapsed();
    return ticket;
}

Ticket DemandPageLoaderImpl::replayRequests( unsigned int deviceIndex, CUstream stream, unsigned int* requestedPages, unsigned int numRequestedPages )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    // Flush any page mappings that have accumulated for the specified device.
    m_pagingSystems.at( deviceIndex )->flushMappings();

    // Create a Ticket that the caller can use to track request processing.
    Ticket ticket( TicketImpl::create( deviceIndex, stream ) );

    m_requestProcessor->addRequests( deviceIndex, stream, requestedPages, numRequestedPages, ticket );

    return ticket;
}

// Predicate that returns pages to a tile pool if the arenaId is high enough
class ResizeTilePoolPredicate : public PageInvalidatorPredicate
{
  public:
    ResizeTilePoolPredicate( unsigned int maxArenas )
        : m_maxArenas( maxArenas )
    {
    }
    bool operator()( unsigned int pageId, unsigned long long pageVal ) override
    {
        TileBlockDesc tileBlock( pageVal );
        // Note: no need to free the tile block in the deviceMemoryManager because the
        // arena associated with the block will be discarded.
        return ( tileBlock.arenaId >= m_maxArenas );
    }
    ~ResizeTilePoolPredicate() override {}
  private:
    unsigned int m_maxArenas;
};

void DemandPageLoaderImpl::setMaxTextureMemory( size_t maxMem )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    for( unsigned int deviceIndex = 0; deviceIndex < m_deviceMemoryManagers.size(); ++deviceIndex )
    {
        unsigned int tilesStartPage = m_options.numPageTableEntries;
        unsigned int tilesEndPage   = m_options.numPages;
        size_t maxArenas            = maxMem / m_deviceMemoryManagers[0]->getTilePoolArenaSize();

        // Resize, deleting tile arenas as needed
        m_deviceMemoryManagers[deviceIndex]->setMaxTextureTileMemory( maxMem );

        // Schedule tiles from deleted arenas to be discarded
        ResizeTilePoolPredicate* predicate = new ResizeTilePoolPredicate( static_cast<unsigned int>( maxArenas ) );
        m_pagesToInvalidate[deviceIndex].push_back( InvalidationRange{tilesStartPage, tilesEndPage, predicate} );
    }

    m_options.maxTexMemPerDevice = maxMem;
}

void DemandPageLoaderImpl::invalidatePageRange( unsigned int deviceIndex, unsigned int startPage, unsigned int endPage, PageInvalidatorPredicate* predicate )
{
    m_pagesToInvalidate[deviceIndex].push_back( InvalidationRange{startPage, endPage, predicate} );
}

DemandPageLoader* createDemandPageLoader( RequestProcessor* requestProcessor, const Options& options )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    return new DemandPageLoaderImpl( requestProcessor, options );
}

void destroyDemandPageLoader( DemandPageLoader* manager )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    delete manager;
}

}  // namespace demandLoading
