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

#include "CascadeRequestFilter.h"
#include "DemandPageLoaderImpl.h"
#include "Util/ContextSaver.h"
#include "Util/NVTXProfiling.h"
#include "Util/Stopwatch.h"
#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/RequestProcessor.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/CascadeImage.h>

#include <cuda.h>

#include <algorithm>
#include <memory>
#include <set>

using namespace otk;

namespace demandLoading {

// Predicate that returns pages (assumed to represent texture tiles) to a tile pool
class TilePoolReturnPredicate : public PageInvalidatorPredicate
{
  public:
    TilePoolReturnPredicate( DeviceMemoryManager* deviceMemoryManager )
        : m_deviceMemoryManager( deviceMemoryManager )
    {
    }
    bool operator()( unsigned int /*pageId*/, unsigned long long pageVal, CUstream /*stream*/ ) override
    {
        m_deviceMemoryManager->freeTileBlock( TileBlockDesc( pageVal ) );
        return true;
    }
    ~TilePoolReturnPredicate() override {}
  private:
    DeviceMemoryManager* m_deviceMemoryManager;
};

// Predicate that migrates texture tiles from an old texture to a new larger texture.
class MigrateTextureTilesPredicate : public PageInvalidatorPredicate
{
  public:
    MigrateTextureTilesPredicate( const TextureSampler& oldSampler,
                                  DemandTextureImpl* newTexture,
                                  DemandPageLoaderImpl* demandPageLoader,
                                  DeviceMemoryManager* deviceMemoryManager )
        : m_oldSampler{oldSampler}
        , m_newTexture{newTexture}
        , m_demandPageLoader{demandPageLoader}
        , m_deviceMemoryManager{deviceMemoryManager}
    {
    }
    bool operator()( unsigned int pageId, unsigned long long pageVal, CUstream stream ) override
    {
        OTK_ASSERT_MSG( pageId >= m_oldSampler.startPage && pageId < m_oldSampler.startPage + m_oldSampler.numPages,
            "pageId outside texture tile range." );

        // FIXME: What if the tiles are non-evictable?

        unsigned int mipLevel;
        unsigned int tileX;
        unsigned int tileY;
        TileBlockDesc tileBlock( pageVal );

        CUmemGenericAllocationHandle tileHandle = m_deviceMemoryManager->getTileBlockHandle( tileBlock );
        unpackTileIndex( m_oldSampler, pageId - m_oldSampler.startPage, mipLevel, tileX, tileY );

        unsigned int newMipLevel = mipLevel;
        unsigned int w = m_newTexture->getSampler().width;
        while( w > m_oldSampler.width )
        {
            w = w >> 1;
            newMipLevel++;
        }

        if( mipLevel >= m_oldSampler.mipTailFirstLevel )
            m_newTexture->mapMipTail( stream, tileHandle, tileBlock.offset() );
        else
            m_newTexture->mapTile( stream, newMipLevel, tileX, tileY, tileHandle, tileBlock.offset() );

        unsigned int newPageId = pageId - m_oldSampler.startPage + m_newTexture->getSampler().startPage;

        // Call addMappingBody instead of addMapping since mutex already acquired in PagingSystem::invalidatePages
        m_demandPageLoader->getPagingSystem()->addMappingBody( newPageId, true, pageVal );

        return true;
    }
    ~MigrateTextureTilesPredicate() override {}
  private:
    TextureSampler m_oldSampler;
    DemandTextureImpl* m_newTexture;
    DemandPageLoaderImpl* m_demandPageLoader;
    DeviceMemoryManager* m_deviceMemoryManager;
};

namespace {

std::shared_ptr<demandLoading::Options> configure( demandLoading::Options options )
{
    // If maxTexMemPerDevice is 0, consider it to be unlimited
    if( options.maxTexMemPerDevice == 0 )
        options.maxTexMemPerDevice = 0xfffffffffffffffful;

    // PagingSystem::pushMappings requires enough capacity to handle all the requested pages.
    if( options.maxFilledPages < options.maxRequestedPages )
        options.maxFilledPages = options.maxRequestedPages;

    return std::shared_ptr<Options>( new Options( options ) );
}

}  // anonymous namespace

DemandLoaderImpl::DemandLoaderImpl( const Options& options )
    : m_options( configure( options ) )
    , m_pageTableManager( std::make_shared<PageTableManager>( m_options->numPages, m_options->numPageTableEntries ) )
    , m_requestProcessor( m_pageTableManager, options )
    , m_pageLoader( new DemandPageLoaderImpl( m_pageTableManager, &m_requestProcessor, m_options ) )
    , m_samplerRequestHandler( this )
    , m_cascadeRequestHandler( this )
    , m_deviceTransferPool( new otk::DeviceAsyncAllocator(), new RingSuballocator(), DEFAULT_ALLOC_SIZE, options.maxPinnedMemory )
{
    // The demand loader is for the current cuda context
    OTK_ERROR_CHECK( cuCtxGetCurrent( &m_cudaContext ) );

    // Reserve pages in the sampler request handler for all possible textures.
    m_samplerRequestHandler.setPageRange( 0, m_options->numPageTableEntries );

    unsigned int samplerStartPage = m_pageTableManager->reserveBackedPages( options.maxTextures * NUM_PAGES_PER_TEXTURE, &m_samplerRequestHandler );
    m_samplerRequestHandler.setPageRange( samplerStartPage, options.maxTextures * NUM_PAGES_PER_TEXTURE );

    // Reserve pages for the cascade request handler if supported
    if( options.useCascadingTextureSizes )
    {
        unsigned int numCascadePages = NUM_CASCADES * options.maxTextures;
        unsigned int cascadeStartPage = m_pageTableManager->reserveUnbackedPages( numCascadePages, &m_cascadeRequestHandler );
        CascadeRequestFilter* requestFilter = new CascadeRequestFilter( cascadeStartPage, cascadeStartPage + numCascadePages, this );
        m_requestProcessor.setRequestFilter( std::shared_ptr<RequestFilter>( requestFilter ) );
    }
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
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    std::unique_lock<std::mutex> lock( m_mutex );

    // Add new texture to the end of the list of textures.  The texture holds a pointer to the
    // image, from which tile data is obtained on demand.
    unsigned int textureId = allocateTexturePages( 1 );

    DemandTextureImpl* tex = makeTextureOrVariant( textureId, textureDesc, imageSource );
    m_textures.emplace( textureId, tex );

    return *m_textures[textureId];
}

// Create a demand-loaded UDIM texture.  The images are not opened until the texture samplers are requested
// by device code (via pagingMapOrRequest in Tex2DGradUdim, or other Tex2D functions).
const DemandTexture& DemandLoaderImpl::createUdimTexture( std::vector<std::shared_ptr<imageSource::ImageSource>>& imageSources,
                                                          std::vector<TextureDescriptor>& textureDescs,
                                                          unsigned int                    udim,
                                                          unsigned int                    vdim,
                                                          int                             baseTextureId,
                                                          unsigned int                    numChannelTextures )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    std::unique_lock<std::mutex> lock( m_mutex );

    // Allocate demand loader pages for the udim grid
    OTK_ASSERT_MSG( udim * vdim > 0, "Udim and vdim must both be positive." );
    unsigned int startTextureId = allocateTexturePages( udim * vdim );

    // Fill the textures in
    unsigned int entryPointId = 0xFFFFFFFF;
    for( unsigned int v = 0; v < vdim; ++v )
    {
        for( unsigned int u = 0; u < udim; ++u )
        {
            for( unsigned int channelId = 0; channelId < numChannelTextures; ++channelId )
            {
                unsigned int imageIndex = (v*udim + u) * numChannelTextures + channelId;
                unsigned int textureId = startTextureId + imageIndex;
                if(imageIndex < imageSources.size() && imageSources[imageIndex].get() != nullptr )
                {
                    // Create the texture and put it in the list of textures
                    entryPointId = std::min( textureId, entryPointId );
                    DemandTextureImpl* tex = makeTextureOrVariant( textureId, textureDescs[imageIndex], imageSources[imageIndex] );
                    m_textures.emplace( textureId, tex );
                    tex->setUdimTexture( startTextureId, udim, vdim, numChannelTextures, false );
                }
                else
                {
                    m_textures.emplace( textureId, nullptr );
                }
            }
        }
    }

    if( baseTextureId >= 0 )
    {
        m_textures[baseTextureId]->setUdimTexture( startTextureId, udim, vdim, numChannelTextures, true );
        return *m_textures[baseTextureId];
    }

    return *m_textures[entryPointId];
}

DemandTextureImpl* DemandLoaderImpl::makeTextureOrVariant( unsigned int textureId, 
                                                           const TextureDescriptor& textureDesc, 
                                                           std::shared_ptr<imageSource::ImageSource>& imageSource )
{
    auto imageIt = m_imageToTextureId.find( imageSource.get() );
    if( imageIt == m_imageToTextureId.end() ) // image was not found. Make a new texture.
    {
        if( getOptions().useCascadingTextureSizes )
        {
            imageSource::CascadeImage* cascadeImg = new imageSource::CascadeImage( imageSource, CASCADE_BASE );
            std::shared_ptr<imageSource::ImageSource> cascadeImage( cascadeImg );
            m_imageToTextureId[imageSource.get()] = textureId;
            return new DemandTextureImpl( textureId, textureDesc, cascadeImage, this );
        }

        m_imageToTextureId[imageSource.get()] = textureId;
        return new DemandTextureImpl( textureId, textureDesc, imageSource, this );
    }
    else // image was found. Make a variant texture.
    {
        DemandTextureImpl* masterTexture = m_textures[imageIt->second].get();
        return new DemandTextureImpl( textureId, masterTexture, textureDesc, this );
    }
}

unsigned int DemandLoaderImpl::createResource( unsigned int numPages, ResourceCallback callback, void* callbackContext )
{
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    m_resourceRequestHandlers.emplace_back( new ResourceRequestHandler( callback, callbackContext, this ) );
    const unsigned int startPage = m_pageTableManager->reserveBackedPages( numPages, m_resourceRequestHandlers.back().get() );
    return startPage;
}

void DemandLoaderImpl::invalidatePage( unsigned int pageId )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    m_pageLoader->invalidatePageRange( pageId, pageId + 1, nullptr );
}

void DemandLoaderImpl::loadTextureTiles( CUstream stream, unsigned int textureId, bool reloadIfResident )
{
    initTexture( stream, textureId );
    TextureRequestHandler *requestHandler = m_textures[textureId]->getRequestHandler();
    unsigned int startPage = requestHandler->getStartPage();
    unsigned int endPage = startPage + requestHandler->getNumPages();

    for( unsigned int pageId = startPage; pageId < endPage; ++pageId )
    {
        requestHandler->loadPage( stream, pageId, reloadIfResident );
    }
}

void DemandLoaderImpl::unloadTextureTiles( unsigned int textureId )
{
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    std::unique_lock<std::mutex> lock( m_mutex );

    // Enqueue page ranges to invalidate when launchPrepare is called
    DemandTextureImpl* texture = m_textures.at( textureId ).get();
    if( texture->isOpen() && texture->getSampler().desc.isSparseTexture )
    {
        texture->init();
        TextureSampler sampler   = texture->getSampler();
        unsigned int   startPage = sampler.startPage;
        unsigned int   endPage   = sampler.startPage + sampler.numPages;

        // Unload texture tiles
        TilePoolReturnPredicate* predicate = new TilePoolReturnPredicate( getDeviceMemoryManager() );
        m_pageLoader->invalidatePageRange( startPage, endPage, predicate );

        // Unload base color
        unsigned int baseColorId = samplerIdToBaseColorId( textureId, getOptions().maxTextures );
        m_pageLoader->invalidatePageRange( baseColorId, baseColorId + 1, nullptr );
    }
}

void DemandLoaderImpl::migrateTextureTiles( const TextureSampler& oldSampler, DemandTextureImpl* newTexture )
{
    // Mutex acquired in caller
    if( !newTexture->getSampler().desc.isSparseTexture )
        return;

    unsigned int startPage = oldSampler.startPage;
    unsigned int endPage   = oldSampler.startPage + oldSampler.numPages;
    MigrateTextureTilesPredicate* predicate = new MigrateTextureTilesPredicate( oldSampler, newTexture, m_pageLoader.get(), getDeviceMemoryManager() );
    m_pageLoader->invalidatePageRange( startPage, endPage, predicate );
}

void DemandLoaderImpl::replaceTexture( CUstream                                  stream,
                                       unsigned int                              textureId,
                                       std::shared_ptr<imageSource::ImageSource> image,
                                       const TextureDescriptor&                  textureDesc,
                                       bool                                      migrateTiles )
{
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );

    // Unload all the texture tiles if they are not being migrated
    if( !migrateTiles )
        unloadTextureTiles( textureId );

    std::unique_lock<std::mutex> lock( m_mutex );

    // Copy the old sampler (for migrating tiles), and replace the texture
    bool textureOpen = m_textures.at( textureId )->isOpen();
    TextureSampler oldSampler = ( textureOpen ) ? m_textures.at( textureId )->getSampler() : TextureSampler{};
    m_textures.at( textureId )->setImage( textureDesc, image );

    if( textureOpen )
    {
        m_samplerRequestHandler.loadPage( stream, textureId, true );
        m_samplerRequestHandler.loadPage( stream, samplerIdToBaseColorId( textureId, getOptions().maxTextures ), true );

        // Reload base color if not migrating the tiles
        if( !migrateTiles )
            m_samplerRequestHandler.loadPage( stream, samplerIdToBaseColorId( textureId, getOptions().maxTextures ), true );

        // Migrate the texture tiles to the new texture
        if( migrateTiles )
            migrateTextureTiles( oldSampler, m_textures.at( textureId ).get() );

        // Take care of texture variants
        DemandTextureImpl* masterTexture = getTexture( textureId );
        const std::vector<unsigned int>& variantIds = masterTexture->getVariantsIds();
        for( unsigned int variantId : variantIds )
        {
            const TextureDescriptor& variantDesc = m_textures.at( variantId )->getDescriptor();
            m_textures.at( variantId )->setImage( variantDesc, image );
            m_samplerRequestHandler.loadPage( stream, variantId, true );
            if( !migrateTiles )
                m_samplerRequestHandler.loadPage( stream, samplerIdToBaseColorId( variantId, getOptions().maxTextures ), true );
        }
    }
}

void DemandLoaderImpl::initTexture( CUstream stream, unsigned int textureId )
{
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
    m_samplerRequestHandler.fillRequest( stream, textureId );
    m_samplerRequestHandler.fillRequest( stream, samplerIdToBaseColorId( textureId, getOptions().maxTextures ) );
}

void DemandLoaderImpl::initUdimTexture( CUstream stream, unsigned int baseTextureId )
{
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
    m_samplerRequestHandler.loadPage( stream, baseTextureId, true ); // make sure the sampler is reloaded to get udim params.
    m_samplerRequestHandler.fillRequest( stream, samplerIdToBaseColorId( baseTextureId, getOptions().maxTextures ) );

    const DemandTextureImpl* baseTexture = m_textures.at( baseTextureId ).get();
    const TextureSampler& baseSampler = baseTexture->getSampler();

    for( unsigned int v = 0; v < baseSampler.vdim; ++v )
    {
        for( unsigned int u = 0; u < baseSampler.udim; ++u)
        {
            for( unsigned int channelId = 0; channelId < baseSampler.numChannelTextures; ++channelId )
            {
                unsigned int subTextureId = baseSampler.udimStartPage +
                                            (v * baseSampler.udim + u) * baseSampler.numChannelTextures + channelId;
                if( subTextureId != baseTextureId )
                    initTexture( stream, subTextureId );
            }
        }
    }
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
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
    unsigned int pageId = m_textures[textureId]->getRequestHandler()->getTextureTilePageId( mipLevel, tileX, tileY );
    m_textures[textureId]->getRequestHandler()->loadPage( stream, pageId, true );
}

bool DemandLoaderImpl::pageResident( unsigned int pageId )
{
    PagingSystem* pagingSystem = m_pageLoader->getPagingSystem();
    return pagingSystem->isResident( pageId );
}

bool DemandLoaderImpl::launchPrepare( CUstream stream, DeviceContext& context )
{
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
    return m_pageLoader->pushMappings( stream, context );
}

Ticket DemandLoaderImpl::processRequests( CUstream stream, const DeviceContext& context )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
    std::unique_lock<std::mutex> lock( m_mutex );

    // Create a Ticket that the caller can use to track request processing.
    Ticket ticket = TicketImpl::create( stream );
    const unsigned int id = m_ticketId++;
    m_requestProcessor.setTicket( id, ticket);

    m_pageLoader->pullRequests( stream, context, id );

    return ticket;
}

void DemandLoaderImpl::abort()
{
    m_requestProcessor.stop();
}

void DemandLoaderImpl::unmapTileResource( CUstream stream, unsigned int pageId )
{
    // Ask the PageTableManager for the RequestHandler associated with the given page index.
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
    RequestHandler* handler = m_pageTableManager->getRequestHandler( pageId );
    OTK_ASSERT_MSG( handler, "Page request does not correspond to a known resource." );

    // Make sure that the handler is a TextureRequestHandler instead of a null request handler
    TextureRequestHandler* textureRequestHandler = dynamic_cast<TextureRequestHandler*>( handler );
    if( textureRequestHandler ) 
        textureRequestHandler->unmapTileResource( stream, pageId );
}

void DemandLoaderImpl::setPageTableEntry( unsigned pageId, bool evictable, unsigned long long pageTableEntry )
{
    m_pageLoader->setPageTableEntry( pageId, evictable, pageTableEntry);
}

PagingSystem* DemandLoaderImpl::getPagingSystem() const
{
    return m_pageLoader->getPagingSystem();
}

PageTableManager* DemandLoaderImpl::getPageTableManager()
{
    return m_pageTableManager.get();
}

void DemandLoaderImpl::freeStagedTiles( CUstream stream )
{
    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
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

const TransferBufferDesc DemandLoaderImpl::allocateTransferBuffer( CUmemorytype memoryType, size_t size, CUstream /*stream*/ )
{
    const unsigned int alignment = 4096;

    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    MemoryBlockDesc memoryBlock{};
    if( memoryType == CU_MEMORYTYPE_HOST )
        memoryBlock = m_pageLoader->getPinnedMemoryPool()->alloc( size, alignment );
    else if( memoryType == CU_MEMORYTYPE_DEVICE )
        memoryBlock = m_deviceTransferPool.alloc( size, alignment );

    return TransferBufferDesc{ memoryType, memoryBlock };
}

void DemandLoaderImpl::freeTransferBuffer( const TransferBufferDesc& transferBuffer, CUstream stream )
{
    // Free the transfer buffer after the stream clears

    OTK_ASSERT_CONTEXT_IS( m_cudaContext );
    OTK_ASSERT_CONTEXT_MATCHES_STREAM( stream );
    if( transferBuffer.memoryType == CU_MEMORYTYPE_HOST )
        m_pageLoader->getPinnedMemoryPool()->freeAsync( transferBuffer.memoryBlock, stream );
    else if( transferBuffer.memoryType == CU_MEMORYTYPE_DEVICE )
        m_deviceTransferPool.freeAsync( transferBuffer.memoryBlock, stream );
    else 
        OTK_ASSERT_MSG( false, "Unknown memory type." );
}

Statistics DemandLoaderImpl::getStatistics() const
{
    std::unique_lock<std::mutex> lock( m_mutex );

    Statistics stats{};
    stats.numTextures           = m_textures.size();
    stats.requestProcessingTime = m_pageLoader->getTotalProcessingTime();
    stats.deviceMemoryUsed      = getDeviceMemoryManager()->getTotalDeviceMemory();

    // Multiple textures can share the same ImageSource. Use a set to avoid duplicate counting.
    std::set<imageSource::ImageSource*> images;
    for( auto texIt = m_textures.begin(); texIt != m_textures.end(); ++texIt )
    {
        DemandTextureImpl* tex = texIt->second.get();
        // If the texture has a new image, add its stats
        if( tex && ( images.find( tex->getImage().get() ) == images.end() ) ) 
        {
            tex->accumulateStatistics( stats );
            images.insert( tex->getImage().get() );
        }
    }
    return stats;
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
    // Allocate pages for numTextures. Note: pages for all textures were reserved in the constructor of DemandLoaderImpl.
    unsigned int textureId = static_cast<unsigned int>( m_textures.size() );
    OTK_ASSERT_MSG( textureId + numTextures - 1 < getOptions().maxTextures, "Too many textures defined.\n" );
    (void)numTextures;  // silence unused variable warning
    return textureId;
}

DemandLoader* createDemandLoader( const Options& options )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Initialize CUDA if necessary
    OTK_ERROR_CHECK( cuInit( 0 ) );

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
