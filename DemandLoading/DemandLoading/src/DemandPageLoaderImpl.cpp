// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPageLoaderImpl.h"

#include "Util/NVTXProfiling.h"
#include "Util/Stopwatch.h"
#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/LRU.h>
#include <OptiXToolkit/DemandLoading/RequestProcessor.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/ErrorCheck.h>

#include <cuda.h>

#include <algorithm>
#include <memory>
#include <set>
#include <utility>

using namespace otk;

namespace demandLoading {

static std::shared_ptr<demandLoading::Options> configureOptions( std::shared_ptr<demandLoading::Options> options )
{
    CUdevice device;
    OTK_ERROR_CHECK( cuCtxGetDevice( &device ) );
    if( !deviceSupportsSparseTextures( device ) )
        options->useSparseTextures = false;
    return options;
}

DemandPageLoaderImpl::DemandPageLoaderImpl( RequestProcessor* requestProcessor, std::shared_ptr<Options> options )
    : DemandPageLoaderImpl( std::make_shared<PageTableManager>( options->numPages, options->numPageTableEntries ), requestProcessor, options )
{
}

DemandPageLoaderImpl::DemandPageLoaderImpl( std::shared_ptr<PageTableManager> pageTableManager,
                                            RequestProcessor*                 requestProcessor,
                                            std::shared_ptr<Options>          options )
    : m_options( configureOptions( options ) )
    , m_deviceMemoryManager( m_options )
    , m_pinnedMemoryPool( new PinnedAllocator(), new RingSuballocator( DEFAULT_ALLOC_SIZE ), DEFAULT_ALLOC_SIZE, m_options->maxPinnedMemory )
    , m_pageTableManager( std::move( pageTableManager ) )
    , m_requestProcessor( requestProcessor )
    , m_pagingSystem( m_options, &m_deviceMemoryManager, &m_pinnedMemoryPool, m_requestProcessor )
{
}

unsigned int DemandPageLoaderImpl::allocatePages( unsigned int numPages, bool backed )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    return backed ? m_pageTableManager->reserveBackedPages( numPages, nullptr ) :
                    m_pageTableManager->reserveUnbackedPages( numPages, nullptr );
}

void DemandPageLoaderImpl::setPageTableEntry( unsigned int pageId, bool evictable, unsigned long long pageTableEntry )
{
    unsigned int lruVal = evictable ? 0U : NON_EVICTABLE_LRU_VAL;
    return m_pagingSystem.addMapping( pageId, lruVal, pageTableEntry );
}

bool DemandPageLoaderImpl::pushMappings( CUstream stream, DeviceContext& context )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );

    // Get DeviceContext from pool and copy it to output parameter.
    {
        // allocate() is not thread safe
        std::unique_lock<std::mutex> lock( m_mutex );
        context = *m_deviceMemoryManager.allocateDeviceContext();
        invalidatePages( stream, context );
    }
    context.requestIfResident = m_options->evictionActive;

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

void DemandPageLoaderImpl::pullRequests( CUstream stream, const DeviceContext& context, unsigned int id )
{
    Stopwatch stopwatch;
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Pull requests from the device.  This launches a kernel on the given stream to scan the
    // request bits, and copies the requested page ids to host memory asynchronously.
    unsigned int  startPage = 0;
    unsigned int  endPage   = m_pageTableManager->getEndPage();
    m_pagingSystem.pullRequests( context, stream, id, startPage, endPage);

    std::unique_lock<std::mutex> lock( m_mutex );
    m_totalProcessingTime += stopwatch.elapsed();
}

// Predicate that returns tile pages to a tile pool if the arenaId is high enough, allowing
// the arenas to be deleted.
class ResizeTilePoolPredicate : public PageInvalidatorPredicate
{
  public:
    ResizeTilePoolPredicate( DeviceMemoryManager* deviceMemoryManager, unsigned int maxArenas )
        : m_deviceMemoryManager( deviceMemoryManager ) 
        , m_maxArenas( maxArenas )
    {
    }
    bool operator()( unsigned int /*pageId*/, unsigned long long pageVal, CUstream /*stream*/ ) override
    {
        TileBlockDesc tileBlock( pageVal );
        // TODO: verify that we don't have to free the tile block in the deviceMemoryManager
        // becasue the arena associated with the block will be discarded.
        // if( tileBlock.arenaId >= m_maxArenas )
        //    m_deviceMemoryManager->freeTileBlock( pageVal );
        return ( tileBlock.arenaId >= m_maxArenas );
    }
    ~ResizeTilePoolPredicate() override {}
  private:
    DeviceMemoryManager* m_deviceMemoryManager;
    unsigned int m_maxArenas;
};

void DemandPageLoaderImpl::setMaxTextureMemory( size_t maxMem )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    unsigned int arenaSize = m_deviceMemoryManager.getTilePoolArenaSize();
    if( maxMem % arenaSize != 0 )
        maxMem = maxMem + (arenaSize - maxMem % arenaSize);
    
    unsigned int tilesStartPage = m_options->numPageTableEntries;
    unsigned int tilesEndPage   = m_options->numPages;
    size_t       maxArenas      = maxMem / m_deviceMemoryManager.getTilePoolArenaSize();

    // Discard tiles from arenas that will be deleted
    if( m_deviceMemoryManager.getTextureTileMemory() > maxMem )
    {
        CUstream stream{0};
        DeviceContext context = *m_deviceMemoryManager.allocateDeviceContext();

        ResizeTilePoolPredicate* predicate =
            new ResizeTilePoolPredicate( &m_deviceMemoryManager, static_cast<unsigned int>( maxArenas ) );
        m_pagesToInvalidate.push_back( InvalidationRange{tilesStartPage, tilesEndPage, predicate} );
        invalidatePages( stream, context );
        cuStreamSynchronize( stream );

        m_deviceMemoryManager.freeDeviceContext( &context );
    }

    // Resize tile pool, deleting tile arenas as needed
    m_deviceMemoryManager.setMaxTextureTileMemory( maxMem );
    m_options->maxTexMemPerDevice = maxMem;
}

void DemandPageLoaderImpl::invalidatePageRange( unsigned int startPage, unsigned int endPage, PageInvalidatorPredicate* predicate )
{
    m_pagesToInvalidate.push_back( InvalidationRange{startPage, endPage, predicate} );
}

DemandPageLoader* createDemandPageLoader( RequestProcessor* requestProcessor, const Options& options )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    return new DemandPageLoaderImpl( requestProcessor, std::make_shared<Options>( options ) );
}

void destroyDemandPageLoader( DemandPageLoader* manager )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    delete manager;
}

}  // namespace demandLoading
