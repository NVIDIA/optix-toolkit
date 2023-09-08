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

#include "Util/Exception.h"
#include "Util/NVTXProfiling.h"
#include "Util/Stopwatch.h"
#include "Util/TraceFile.h"
#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/LRU.h>
#include <OptiXToolkit/DemandLoading/RequestProcessor.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>

#include <cuda.h>

#include <algorithm>
#include <memory>
#include <set>
#include <utility>

using namespace otk;

namespace {

demandLoading::Options configure( demandLoading::Options options )
{
    // If maxTexMemPerDevice is 0, consider it to be unlimited
    if( options.maxTexMemPerDevice == 0 )
        options.maxTexMemPerDevice = 0xfffffffffffffffful;

    // PagingSystem::pushMappings requires enough capacity to handle all the requested pages.
    if( options.maxFilledPages < options.maxRequestedPages )
        options.maxFilledPages = options.maxRequestedPages;

    // Anticipate at least one active stream per device.
    options.maxActiveStreams = std::max( 1U, options.maxActiveStreams );

    return options;
}

}  // anonymous namespace

namespace demandLoading {

bool DemandPageLoaderImpl::supportsSparseTextures( CUdevice device )
{
    int sparseSupport = 0;
    DEMAND_CUDA_CHECK( cuDeviceGetAttribute( &sparseSupport, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, device ) );

    // Skip devices in TCC mode.  This guards against an "operation not supported" error when
    // querying the recommended allocation granularity via cuMemGetAllocationGranularity.
    int inTccMode = 0;
    DEMAND_CUDA_CHECK( cuDeviceGetAttribute( &inTccMode, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device ) );

    return sparseSupport && !inTccMode;
}

DemandPageLoaderImpl::DemandPageLoaderImpl( RequestProcessor* requestProcessor, const Options& options )
    : DemandPageLoaderImpl( std::make_shared<PageTableManager>( configure( options ).numPages, configure( options ).numPageTableEntries ), requestProcessor, options )
{
}

DemandPageLoaderImpl::DemandPageLoaderImpl( std::shared_ptr<PageTableManager> pageTableManager,
                                            RequestProcessor*                 requestProcessor,
                                            const Options&                    options )
    : m_options( configure( options ) )
    , m_deviceMemoryManager( options )
    , m_pinnedMemoryPool( new PinnedAllocator(), new RingSuballocator( DEFAULT_ALLOC_SIZE ), DEFAULT_ALLOC_SIZE, options.maxPinnedMemory )
    , m_pageTableManager( std::move( pageTableManager ) )
    , m_requestProcessor( requestProcessor )
    , m_pagingSystem( m_options, &m_deviceMemoryManager, &m_pinnedMemoryPool, m_requestProcessor )
{
    CUdevice device;
    DEMAND_CUDA_CHECK( cuCtxGetDevice( &device ) );
    if( !supportsSparseTextures( device ) )
        m_options.useSparseTextures = false;
}

unsigned int DemandPageLoaderImpl::allocatePages( unsigned int numPages, bool backed )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    return backed ? m_pageTableManager->reserveBackedPages( numPages, nullptr ) :
                    m_pageTableManager->reserveUnbackedPages( numPages, nullptr );
}

void DemandPageLoaderImpl::setPageTableEntry( unsigned int pageId, bool evictable, void* pageTableEntry )
{
    return m_pagingSystem.addMapping( pageId, evictable ? 0U : NON_EVICTABLE_LRU_VAL,
                                      reinterpret_cast<unsigned long long>( pageTableEntry ) );
}

// Returns false if the device doesn't support sparse textures.
bool DemandPageLoaderImpl::pushMappings( CUstream stream, DeviceContext& context )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    OTK_STREAM_CUDA_CHECK( stream );

    // Get DeviceContext from pool and copy it to output parameter.
    {
        // allocate() is not thread safe
        std::unique_lock<std::mutex> lock( m_mutex );
        context = *m_deviceMemoryManager.allocateDeviceContext();
        invalidatePages( stream, context );
    }
    context.requestIfResident = m_options.evictionActive;

    m_pagingSystem.pushMappings( context, stream );
    return true;
}

void DemandPageLoaderImpl::invalidatePages( CUstream stream, DeviceContext& context )
{
    // Mutex acquired in caller
    for( InvalidationRange& ir : m_pagesToInvalidate )
    {
        m_pagingSystem.invalidatePages( ir.startPage, ir.endPage, ir.predicate, context, stream );
        delete ir.predicate;
    }
    m_pagesToInvalidate.clear();
}


// Process page requests.
void DemandPageLoaderImpl::pullRequests( CUstream stream, const DeviceContext& context, unsigned int id )
{
    Stopwatch stopwatch;
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Pull requests from the device.  This launches a kernel on the given stream to scan the
    // request bits copies the requested page ids to host memory (asynchronously).
    unsigned int  startPage = 0;
    unsigned int  endPage   = m_pageTableManager->getEndPage();
    m_pagingSystem.pullRequests( context, stream, id, startPage, endPage);

    std::unique_lock<std::mutex> lock( m_mutex );
    m_totalProcessingTime += stopwatch.elapsed();
}

void DemandPageLoaderImpl::replayRequests( CUstream stream, unsigned int id, const unsigned int* pageIds, unsigned int numPageIds )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    std::unique_lock<std::mutex> lock( m_mutex );

    // Flush any page mappings that have accumulated for the current CUDA context.
    m_pagingSystem.flushMappings();

    m_requestProcessor->addRequests( stream, id, pageIds, numPageIds);
}

// Predicate that returns pages to a tile pool if the arenaId is high enough
class ResizeTilePoolPredicate : public PageInvalidatorPredicate
{
  public:
    ResizeTilePoolPredicate( unsigned int maxArenas )
        : m_maxArenas( maxArenas )
    {
    }
    bool operator()( unsigned int /*pageId*/, unsigned long long pageVal ) override
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
    
    unsigned int tilesStartPage = m_options.numPageTableEntries;
    unsigned int tilesEndPage   = m_options.numPages;
    size_t       maxArenas      = maxMem / m_deviceMemoryManager.getTilePoolArenaSize();

    // Resize, deleting tile arenas as needed
    m_deviceMemoryManager.setMaxTextureTileMemory( maxMem );

    // Schedule tiles from deleted arenas to be discarded
    ResizeTilePoolPredicate* predicate = new ResizeTilePoolPredicate( static_cast<unsigned int>( maxArenas ) );
    m_pagesToInvalidate.push_back( InvalidationRange{tilesStartPage, tilesEndPage, predicate} );

    m_options.maxTexMemPerDevice = maxMem;
}

void DemandPageLoaderImpl::invalidatePageRange( unsigned int startPage, unsigned int endPage, PageInvalidatorPredicate* predicate )
{
    m_pagesToInvalidate.push_back( InvalidationRange{startPage, endPage, predicate} );
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
