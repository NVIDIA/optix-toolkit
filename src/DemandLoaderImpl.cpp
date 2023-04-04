//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "DemandPageLoaderImpl.h"
#include "Util/CudaContext.h"
#include "Util/ContextSaver.h"
#include "Util/Exception.h"
#include "Util/NVTXProfiling.h"
#include "Util/Stopwatch.h"
#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/RequestProcessor.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>

#include <cuda.h>

#include <algorithm>
#include <memory>
#include <set>

namespace demandLoading {

// Predicate that returns pages (assumed to represent texture tiles) to a tile pool
class TilePoolReturnPredicate : public PageInvalidatorPredicate
{
  public:
    TilePoolReturnPredicate( DeviceMemoryManager* deviceMemoryManager )
        : m_deviceMemoryManager( deviceMemoryManager )
    {
    }
    bool operator()( unsigned int pageId, unsigned long long pageVal ) override
    {
        m_deviceMemoryManager->freeTileBlock( TileBlockDesc( pageVal ) );
        return true;
    }
    ~TilePoolReturnPredicate() override {}
  private:
    DeviceMemoryManager* m_deviceMemoryManager;
};

// Predicate that returns TextureSamplers to texture sampler pool
class TextureSamplerReturnPredicate : public PageInvalidatorPredicate
{
  public:
    TextureSamplerReturnPredicate( DeviceMemoryManager* deviceMemoryManager )
        : m_deviceMemoryManager( deviceMemoryManager )
    {
    }
    bool operator()( unsigned int pageId, unsigned long long pageVal ) override
    {
        m_deviceMemoryManager->freeSampler( reinterpret_cast<TextureSampler*>( pageVal ) );
        return true;
    }
    ~TextureSamplerReturnPredicate() override {}
  private:
    DeviceMemoryManager* m_deviceMemoryManager;
};

DemandLoaderImpl::DemandLoaderImpl( const Options& options )
    : m_pageTableManager( std::make_shared<PageTableManager>( options.numPages, options.numPageTableEntries ) )
    , m_requestProcessor( m_pageTableManager, options )
    , m_pageLoader( new DemandPageLoaderImpl( m_pageTableManager, &m_requestProcessor, options ) )
    , m_samplerRequestHandler( this )
{
    // Reserve bits in the sampler request handler for all possible textures.
    m_samplerRequestHandler.setPageRange( 0, options.numPageTableEntries );

    m_requestProcessor.start( options.maxThreads );
}

DemandLoaderImpl::~DemandLoaderImpl()
{
    m_requestProcessor.stop();
}

// Create a demand-loaded texture.  The image is not opened until the texture sampler is requested
// by device code (via pagingMapOrRequest in Tex2D).
const DemandTexture& DemandLoaderImpl::createTexture( std::shared_ptr<imageSource::ImageSource> imageSource,
                                                      const TextureDescriptor&                  textureDesc )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    // Add new texture to the end of the list of textures.  The texture holds a pointer to the
    // image, from which tile data is obtained on demand.
    unsigned int textureId = allocateTexturePages( 1 );

    DemandTextureImpl* tex = makeTextureOrVariant( textureId, textureDesc, imageSource );
    m_textures.emplace( textureId, tex );

    // Record the image reader and texture descriptor.
    m_requestProcessor.recordTexture( imageSource, textureDesc );

    return *m_textures[textureId];
}

// Create a demand-loaded UDIM texture.  The images are not opened until the texture samplers are requested
// by device code (via pagingMapOrRequest in Tex2DGradUdim, or other Tex2D functions).
const DemandTexture& DemandLoaderImpl::createUdimTexture( std::vector<std::shared_ptr<imageSource::ImageSource>>& imageSources,
                                                          std::vector<TextureDescriptor>& textureDescs,
                                                          unsigned int                    udim,
                                                          unsigned int                    vdim,
                                                          int                             baseTextureId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    // Allocate demand loader pages for the udim grid
    DEMAND_ASSERT_MSG( udim * vdim > 0, "Udim and vdim must both be positive." );
    unsigned int startTextureId = allocateTexturePages( udim * vdim );

    // Fill the textures in
    unsigned int entryPointId = 0xFFFFFFFF;
    for( unsigned int v=0; v<vdim; ++v )
    {
        for( unsigned int u=0; u<udim; ++u )
        {
            unsigned int imageIndex = v*udim + u;
            unsigned int textureId = startTextureId + imageIndex * PAGES_PER_TEXTURE;
            if(imageIndex < imageSources.size() && imageSources[imageIndex].get() != nullptr )
            {
                // Create the texture and put it in the list of textures
                entryPointId = std::min( textureId, entryPointId );
                DemandTextureImpl* tex = makeTextureOrVariant( textureId, textureDescs[imageIndex], imageSources[imageIndex] );
                m_textures.emplace( textureId, tex );
                
                // Record the image reader and texture descriptor.
                m_requestProcessor.recordTexture( imageSources[imageIndex], textureDescs[imageIndex] );
            }
            else 
            {
                m_textures.emplace( textureId, nullptr );
            }
        }
    }

    m_textures[entryPointId]->setUdimTexture( startTextureId, udim, vdim, false );
    if( baseTextureId >= 0 )
    {
        m_textures[baseTextureId]->setUdimTexture( startTextureId, udim, vdim, true );
        return *m_textures[baseTextureId];
    }
    return *m_textures[entryPointId];
}

DemandTextureImpl* DemandLoaderImpl::makeTextureOrVariant( unsigned int textureId, 
                                                           const TextureDescriptor& textureDesc, 
                                                           std::shared_ptr<imageSource::ImageSource>& imageSource )
{
    auto imageIt = m_imageToTextureId.find( imageSource.get() );
    if( imageIt == m_imageToTextureId.end() )
    {
        // image was not found. Make a new texture.
        m_imageToTextureId[imageSource.get()] = textureId;
        return new DemandTextureImpl( textureId, textureDesc, imageSource, this );
    }
    else
    {
        // image was found. Make a variant texture.
        DemandTextureImpl* masterTexture = m_textures[imageIt->second].get();
        return new DemandTextureImpl( textureId, masterTexture, textureDesc, this );
    }
}

unsigned int DemandLoaderImpl::createResource( unsigned int numPages, ResourceCallback callback, void* callbackContext )
{
    m_resourceRequestHandlers.emplace_back( new ResourceRequestHandler( callback, callbackContext, this ) );
    const unsigned int startPage = m_pageTableManager->reserveBackedPages( numPages, m_resourceRequestHandlers.back().get() );
    return startPage;
}

void DemandLoaderImpl::unloadTextureTiles( unsigned int textureId )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Enqueue page ranges to invalidate when launchPrepare is called
    DemandTextureImpl* texture = m_textures.at( textureId ).get();
    if( texture->isOpen() )
    {
        texture->init();
        TextureSampler sampler   = texture->getSampler();
        unsigned int   startPage = sampler.startPage;
        unsigned int   endPage   = sampler.startPage + sampler.numPages;

        for( unsigned int deviceIndex : m_pageLoader->getDevices() )
        {
            // Unload texture tiles
            TilePoolReturnPredicate* predicate = new TilePoolReturnPredicate( getDeviceMemoryManager() );
            m_pageLoader->invalidatePageRange( startPage, endPage, predicate );

            // Unload base color
            unsigned int baseColorId = textureId + BASE_COLOR_OFFSET;
            m_pageLoader->invalidatePageRange( baseColorId, baseColorId + 1, nullptr );
        }
    }
}

void DemandLoaderImpl::replaceTexture( unsigned int textureId, std::shared_ptr<imageSource::ImageSource> image, const TextureDescriptor& textureDesc )
{
    unloadTextureTiles( textureId );
    std::unique_lock<std::mutex> lock( m_mutex );
    bool samplerNeedsReset = m_textures.at( textureId )->setImage( textureDesc, image );

    // Invalidate the texture sampler 
    if( samplerNeedsReset )
    {
        for( unsigned int deviceIndex : m_pageLoader->getDevices() )
        {
            TextureSamplerReturnPredicate* predicate = new TextureSamplerReturnPredicate( getDeviceMemoryManager() );
            m_pageLoader->invalidatePageRange( textureId, textureId + 1, predicate );
        } 
    }

    // Record the image reader and texture descriptor.
    m_requestProcessor.recordTexture( image, textureDesc );
}

void DemandLoaderImpl::initTexture( CUstream stream, unsigned int textureId )
{
    checkCudaContext( stream );
    m_samplerRequestHandler.fillRequest( stream, textureId );
    m_samplerRequestHandler.fillRequest( stream, textureId + BASE_COLOR_OFFSET );
}

unsigned int DemandLoaderImpl::getTextureTilePageId( unsigned int textureId, unsigned int mipLevel, unsigned int tileX, unsigned int tileY )
{
    return m_textures[textureId]->getRequestHandler()->getTextureTilePageId( mipLevel, tileX, tileY );
}

unsigned int DemandLoaderImpl::getMipTailFirstLevel( unsigned int textureId )
{
    return m_textures[textureId]->getMipTailFirstLevel();
}

void DemandLoaderImpl::loadTextureTile( CUstream stream, unsigned int textureId, unsigned int mipLevel, unsigned int tileX, unsigned int tileY )
{
    checkCudaContext( stream );
    unsigned int pageId = m_textures[textureId]->getRequestHandler()->getTextureTilePageId( mipLevel, tileX, tileY );
    m_textures[textureId]->getRequestHandler()->loadPage( stream, pageId, true );
}

bool DemandLoaderImpl::pageResident( unsigned int pageId )
{
    PagingSystem* pagingSystem = m_pageLoader->getPagingSystem();
    return pagingSystem->isResident( pageId );
}

// Returns false if the device doesn't support sparse textures.
bool DemandLoaderImpl::launchPrepare( CUstream stream, DeviceContext& context )
{
    return m_pageLoader->pushMappings( stream, context );
}

namespace { // anonymous

// Check that the current CUDA context matches the one associated with the given stream
// and return the associated device index.
unsigned int getDeviceIndex( CUstream stream )
{
    // Get the current CUDA context.
    CUcontext cudaContext, streamContext;
    DEMAND_CUDA_CHECK( cuCtxGetCurrent( &cudaContext ) );
    DEMAND_CUDA_CHECK( cuCtxGetCurrent( &streamContext ) );
    DEMAND_ASSERT_MSG( cudaContext == streamContext,
                       "The current CUDA context must match the one associated with the given stream" );

    // Get the device index from the CUDA context.
    CUdevice device;
    DEMAND_CUDA_CHECK( cuCtxGetDevice( &device ) );
    return static_cast<unsigned int>( device );
}

} // anonymous namespace

// Process page requests.
Ticket DemandLoaderImpl::processRequests( CUstream stream, const DeviceContext& context )

{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    unsigned int deviceIndex = getDeviceIndex( stream );

    // Create a Ticket that the caller can use to track request processing.
    Ticket ticket = TicketImpl::create( stream );
    const unsigned int id = m_ticketId++;
    m_requestProcessor.setTicket( id, ticket);

    m_pageLoader->pullRequests( stream, context, id );

    return ticket;
}

Ticket DemandLoaderImpl::replayRequests( CUstream stream, unsigned int* requestedPages, unsigned int numRequestedPages )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    unsigned int deviceIndex = getDeviceIndex( stream );

    // Create a Ticket that the caller can use to track request processing.
    Ticket ticket = TicketImpl::create( stream );
    const unsigned int id = m_ticketId++;
    m_requestProcessor.setTicket( id, ticket );

    m_pageLoader->replayRequests( stream, id, requestedPages, numRequestedPages );

    return ticket;
}

void DemandLoaderImpl::unmapTileResource( CUstream stream, unsigned int pageId )
{
    // Ask the PageTableManager for the RequestHandler associated with the given page index.
    TextureRequestHandler* handler = dynamic_cast<TextureRequestHandler*>( m_pageTableManager->getRequestHandler( pageId ) );
    DEMAND_ASSERT_MSG( handler != nullptr, "Page request does not correspond to a known resource" );
    handler->unmapTileResource( stream, pageId );
}

void DemandLoaderImpl::setPageTableEntry( unsigned pageId, bool evictable, void* pageTableEntry )
{
    m_pageLoader->setPageTableEntry( pageId, evictable, pageTableEntry);
}

PagingSystem* DemandLoaderImpl::getPagingSystem() const
{
    return m_pageLoader->getPagingSystem();
}


#if CUDA_VERSION >= 11020
MemoryPool<DeviceAsyncAllocator, RingSuballocator>* DemandLoaderImpl::getDeviceTransferPool()
{
    std::unique_lock<std::mutex> lock( m_deviceTransferPoolsMutex );
    MemoryPool<DeviceAsyncAllocator, RingSuballocator>* deviceTransferPool = m_deviceTransferPools.find();
    if( deviceTransferPool )
        return deviceTransferPool;
    DeviceAsyncAllocator* allocator = new DeviceAsyncAllocator();
    std::unique_ptr<MemoryPool<DeviceAsyncAllocator, RingSuballocator>> ptr(
        new MemoryPool<DeviceAsyncAllocator, RingSuballocator>( allocator, nullptr, 8 * ( 1 << 20 ), 64 * ( 1 << 20 ) ) );
    return m_deviceTransferPools.insert( std::move( ptr ) );
}
#else
MemoryPool<DeviceAllocator, RingSuballocator>* DemandLoaderImpl::getDeviceTransferPool()
{
    std::unique_lock<std::mutex> lock( m_deviceTransferPoolsMutex );
    MemoryPool<DeviceAllocator, RingSuballocator>* deviceTransferPool = m_deviceTransferPools.find();
    if( deviceTransferPool )
        return deviceTransferPool;
    DeviceAllocator*  allocator    = new DeviceAllocator();
    RingSuballocator* suballocator = new RingSuballocator( 4 << 20 );
    std::unique_ptr<MemoryPool<DeviceAllocator, RingSuballocator>> ptr(
        new MemoryPool<DeviceAllocator, RingSuballocator>( allocator, suballocator, 8 * ( 1 << 20 ), 64 * ( 1 << 20 ) ) );
    return m_deviceTransferPools.insert( std::move( ptr ) );
}
#endif

PageTableManager* DemandLoaderImpl::getPageTableManager()
{
    return m_pageTableManager.get();
}

void DemandLoaderImpl::freeStagedTiles( CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    PagingSystem* pagingSystem = getPagingSystem();
    PageMapping   mapping;

    while( getDeviceMemoryManager()->needTileBlocksFreed() )
    {
        pagingSystem->activateEviction( true );
        if( pagingSystem->freeStagedPage( &mapping ) )
        {
            unmapTileResource( stream, mapping.id );
            getDeviceMemoryManager()->freeTileBlock( mapping.page );
        }
        else 
        {
            break;
        }
    }
}

const TransferBufferDesc DemandLoaderImpl::allocateTransferBuffer( CUmemorytype memoryType, size_t size, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    const unsigned int alignment = 4096;
    
    MemoryBlockDesc memoryBlock{};
    if( memoryType == CU_MEMORYTYPE_HOST )
        memoryBlock = m_pageLoader->getPinnedMemoryPool()->alloc( size, alignment );
    else if( memoryType == CU_MEMORYTYPE_DEVICE )
        memoryBlock = getDeviceTransferPool()->alloc( size, alignment );

    DEMAND_ASSERT_MSG( memoryBlock.isGood(), "Transfer buffer allocation failed." );
    return TransferBufferDesc{ memoryType, memoryBlock };
}

 
void DemandLoaderImpl::freeTransferBuffer( const TransferBufferDesc& transferBuffer, CUstream stream )
{
    // Free the transfer buffer after the stream clears

    if( transferBuffer.memoryType == CU_MEMORYTYPE_HOST )
        m_pageLoader->getPinnedMemoryPool()->freeAsync( transferBuffer.memoryBlock, stream );
    else if( transferBuffer.memoryType == CU_MEMORYTYPE_DEVICE )
        getDeviceTransferPool()->freeAsync( transferBuffer.memoryBlock, stream );
    else 
        DEMAND_ASSERT_MSG( false, "Unknown memory type." );
}


Statistics DemandLoaderImpl::getStatistics() const
{
    std::unique_lock<std::mutex> lock( m_mutex );
    Statistics                   stats{};
    stats.requestProcessingTime = m_pageLoader->getTotalProcessingTime();
    stats.numTextures           = m_textures.size();

    // Multiple textures might share the same ImageSource, so we create a set as we go to avoid
    // duplicate counting.
    std::set<imageSource::ImageSource*> images;
    for( auto texIt = m_textures.begin(); texIt != m_textures.end(); ++texIt )
    {
        // Skip null textures
        DemandTextureImpl* tex = texIt->second.get();
        if( tex == nullptr ) 
            continue; 

        tex->accumulateStatistics( stats, images );
    }

    m_pageLoader->accumulateStatistics( stats );

    return stats;
}

const Options& DemandLoaderImpl::getOptions() const
{
    return m_pageLoader->getOptions();
}

std::vector<unsigned> DemandLoaderImpl::getDevices() const
{
    return m_pageLoader->getDevices();
}

void DemandLoaderImpl::enableEviction( bool evictionActive )
{
    m_pageLoader->enableEviction( evictionActive );
}

DeviceMemoryManager* DemandLoaderImpl::getDeviceMemoryManager() const
{
    return m_pageLoader->getDeviceMemoryManager();
}

MemoryPool<PinnedAllocator, RingSuballocator>* DemandLoaderImpl::getPinnedMemoryPool()
{
    return m_pageLoader->getPinnedMemoryPool();
}

void DemandLoaderImpl::setMaxTextureMemory( size_t maxMem )
{
    m_pageLoader->setMaxTextureMemory( maxMem );
}

unsigned int DemandLoaderImpl::allocateTexturePages( unsigned int numTextures )
{
    // Allocate enough pages per texture, aligned to PAGES_PER_TEXTURE
    unsigned int textureId = m_pageTableManager->reserveBackedPages( PAGES_PER_TEXTURE * numTextures, &m_samplerRequestHandler );
    if( textureId % PAGES_PER_TEXTURE != 0 )
    {
        unsigned int alignmentPages = PAGES_PER_TEXTURE - (textureId % PAGES_PER_TEXTURE);
        m_pageTableManager->reserveBackedPages( alignmentPages, &m_samplerRequestHandler );
        textureId = textureId + alignmentPages;
    }
    return textureId;
}

DemandLoader* createDemandLoader( const Options& options )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Initialize CUDA if necessary
    DEMAND_CUDA_CHECK( cuInit( 0 ) );

    ContextSaver contextSaver;
    return new DemandLoaderImpl( options );
}

void destroyDemandLoader( DemandLoader* manager )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    ContextSaver contextSaver;
    delete manager;
}

}  // namespace demandLoading
