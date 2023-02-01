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
#include "RequestProcessor.h"
#include "Util/Exception.h"
#include "Util/NVTXProfiling.h"
#include "Util/Stopwatch.h"
#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
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
    : m_pageTableManager( std::make_shared<PageTableManager>( options.numPages ) )
    , m_requestProcessor( m_pageTableManager, options )
    , m_pageLoader( new DemandPageLoaderImpl( m_pageTableManager, &m_requestProcessor, options ) )
    , m_baseColorRequestHandler( this )
    , m_samplerRequestHandler( this )
{

    // Create transfer buffer pools
    for( unsigned int deviceIndex = 0; deviceIndex < m_pageLoader->getNumDevices(); ++deviceIndex )
    {
#if CUDA_VERSION >= 11020
        DeviceAsyncAllocator* allocator = new DeviceAsyncAllocator( deviceIndex );
        m_deviceTransferPools.emplace_back( allocator, nullptr, 8 * (1<<20), 64 * (1<<20) );
#else
        DeviceAllocator* allocator = new DeviceAllocator( deviceIndex );
        RingSuballocator* suballocator = new RingSuballocator( 4 << 20 );
        m_deviceTransferPools.emplace_back( allocator, suballocator, 8 * (1<<20), 64 * (1<<20) );
#endif
    }

    // Reserve virtual address space for texture samplers, which is associated with the sampler request handler.
    // Note that the max number of samplers/textures is half the number of page table entries.
    m_pageTableManager->reserve( m_pageLoader->getOptions().numPageTableEntries / 2, &m_samplerRequestHandler );
    m_pageTableManager->reserve( m_pageLoader->getOptions().numPageTableEntries / 2, &m_baseColorRequestHandler );

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

    // The texture id will be the next index in the texture array.
    unsigned int textureId = static_cast<unsigned int>( m_textures.size() );

    // Add new texture to the end of the list of textures.  The texture holds a pointer to the
    // image, from which tile data is obtained on demand.
    DemandTextureImpl* tex = makeTextureOrVariant( textureId, textureDesc, imageSource );
    m_textures.emplace_back( tex );

    // Record the image reader and texture descriptor.
    m_requestProcessor.recordTexture( imageSource, textureDesc );

    return *m_textures.back();
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

    // Create all the slots we need in textures array
    unsigned int startIndex = 0;
    {
        startIndex = static_cast<unsigned int>( m_textures.size() );
        m_textures.resize( startIndex + udim*vdim );
    }

    // Fill the slots in the textures array
    unsigned int entryPointIndex = static_cast<unsigned int>( m_textures.size() );
    for( unsigned int v=0; v<vdim; ++v )
    {
        for( unsigned int u=0; u<udim; ++u )
        {
            unsigned int imageIndex = v*udim + u;
            unsigned int textureId = startIndex + imageIndex;
            if(imageIndex < imageSources.size() && imageSources[imageIndex].get() != nullptr )
            {
                if( textureId < entryPointIndex )
                    entryPointIndex = textureId;
                
                // Create the texture and put it in the list of textures
                DemandTextureImpl* tex = makeTextureOrVariant( textureId, textureDescs[imageIndex], imageSources[imageIndex] );
                m_textures[textureId].reset( tex );
                
                // Record the image reader and texture descriptor.
                m_requestProcessor.recordTexture( imageSources[imageIndex], textureDescs[imageIndex] );
            }
            else 
            {
                m_textures[textureId].reset( nullptr );
            }
        }
    }

    m_textures[entryPointIndex]->setUdimTexture( startIndex, udim, vdim, false );
    if( baseTextureId >= 0 )
        m_textures[baseTextureId]->setUdimTexture( startIndex, udim, vdim, true );

    return (baseTextureId >= 0) ? *m_textures[baseTextureId] : *m_textures[entryPointIndex];
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
        return new DemandTextureImpl( textureId, m_pageLoader->getNumDevices(), textureDesc, imageSource, this );
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
    return m_pageLoader->createResource( numPages, callback, callbackContext );
}

void DemandLoaderImpl::unloadTextureTiles( unsigned int textureId )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_ASSERT_MSG( textureId < m_textures.size(), "Unknown texture id" );

    // Enqueue page ranges to invalidate when launchPrepare is called
    if( m_textures[textureId]->isOpen() )
    {
        m_textures[textureId]->init( m_pageLoader->getDevices()[0] );
        TextureSampler sampler   = m_textures[textureId]->getSampler();
        unsigned int   startPage = sampler.startPage;
        unsigned int   endPage   = sampler.startPage + sampler.numPages;

        for( unsigned int deviceIndex : m_pageLoader->getDevices() )
        {
            // Unload texture tiles
            TilePoolReturnPredicate* predicate = new TilePoolReturnPredicate( getDeviceMemoryManager( deviceIndex ) );
            m_pageLoader->invalidatePageRange( deviceIndex, startPage, endPage, predicate );

            // Unload base color
            unsigned int baseColorId = getPagingSystem( deviceIndex )->getBaseColorPageId( textureId );
            m_pageLoader->invalidatePageRange( deviceIndex, baseColorId, baseColorId + 1, nullptr );
        }
    }
}

void DemandLoaderImpl::replaceTexture( unsigned int textureId, std::shared_ptr<imageSource::ImageSource> image, const TextureDescriptor& textureDesc )
{
    unloadTextureTiles( textureId );
    std::unique_lock<std::mutex> lock( m_mutex );
    bool                         samplerNeedsReset = m_textures[textureId]->setImage( textureDesc, image );

    // Invalidate the texture sampler
    if( samplerNeedsReset )
    {
        for( unsigned int deviceIndex : m_pageLoader->getDevices() )
        {
            TextureSamplerReturnPredicate* predicate =
                new TextureSamplerReturnPredicate( getDeviceMemoryManager( deviceIndex ) );
            m_pageLoader->invalidatePageRange( deviceIndex, textureId, textureId + 1, predicate );
        }
    }

    // Record the image reader and texture descriptor.
    m_requestProcessor.recordTexture( image, textureDesc );
}

// Returns false if the device doesn't support sparse textures.
bool DemandLoaderImpl::launchPrepare( unsigned int deviceIndex, CUstream stream, DeviceContext& context )
{
    return m_pageLoader->launchPrepare( deviceIndex, stream, context );
}

// Process page requests.
Ticket DemandLoaderImpl::processRequests( unsigned int deviceIndex, CUstream stream, const DeviceContext& context )
{
    return m_pageLoader->processRequests( deviceIndex, stream, context );
}

Ticket DemandLoaderImpl::replayRequests( unsigned int deviceIndex, CUstream stream, unsigned int* requestedPages, unsigned int numRequestedPages )
{
    return m_pageLoader->replayRequests( deviceIndex, stream, requestedPages, numRequestedPages );
}


void DemandLoaderImpl::unmapTileResource( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    // Ask the PageTableManager for the RequestHandler associated with the given page index.
    TextureRequestHandler* handler = dynamic_cast<TextureRequestHandler*>( m_pageTableManager->getRequestHandler( pageId ) );
    DEMAND_ASSERT_MSG( handler != nullptr, "Page request does not correspond to a known resource" );
    handler->unmapTileResource( deviceIndex, stream, pageId );
}

PagingSystem* DemandLoaderImpl::getPagingSystem( unsigned int deviceIndex ) const
{
    return m_pageLoader->getPagingSystem( deviceIndex );
}

PageTableManager* DemandLoaderImpl::getPageTableManager()
{
    return m_pageTableManager.get();
}

void DemandLoaderImpl::freeStagedTiles( unsigned int deviceIndex, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    PagingSystem* pagingSystem = getPagingSystem( deviceIndex );
    PageMapping   mapping;

    while( getDeviceMemoryManager( deviceIndex )->needTileBlocksFreed()  )
    {
        pagingSystem->activateEviction( true );
        if( pagingSystem->freeStagedPage( &mapping ) )
        {
            unmapTileResource( deviceIndex, stream, mapping.id );
            getDeviceMemoryManager( deviceIndex )->freeTileBlock( mapping.page );
        }
        else 
        {
            break;
        }
    }
}


const TransferBufferDesc DemandLoaderImpl::allocateTransferBuffer( unsigned int deviceIndex, CUmemorytype memoryType, size_t size, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    const unsigned int alignment = 4096;
    
    MemoryBlockDesc memoryBlock{};
    if( memoryType == CU_MEMORYTYPE_HOST )
        memoryBlock = m_pageLoader->getPinnedMemoryPool()->alloc( size, alignment );
    else if( memoryType == CU_MEMORYTYPE_DEVICE )
        memoryBlock = m_deviceTransferPools[deviceIndex].alloc( size, alignment );

    DEMAND_ASSERT_MSG( memoryBlock.isGood(), "Transfer buffer allocation failed." );
    return TransferBufferDesc{ deviceIndex, memoryType, memoryBlock };
}

 
void DemandLoaderImpl::freeTransferBuffer( const TransferBufferDesc& transferBuffer, CUstream stream )
{
    // Free the transfer buffer after the stream clears

    if( transferBuffer.memoryType == CU_MEMORYTYPE_HOST )
        m_pageLoader->getPinnedMemoryPool()->freeAsync( transferBuffer.memoryBlock, transferBuffer.deviceIndex, stream );
    else if( transferBuffer.memoryType == CU_MEMORYTYPE_DEVICE )
        m_deviceTransferPools[ transferBuffer.deviceIndex ].freeAsync( transferBuffer.memoryBlock, transferBuffer.deviceIndex, stream );
    else 
        DEMAND_ASSERT_MSG( false, "Unknown memory type." );
}


Statistics DemandLoaderImpl::getStatistics() const
{
    std::unique_lock<std::mutex> lock( m_mutex );
    Statistics                   stats{};
    stats.requestProcessingTime = m_totalProcessingTime;
    stats.numTextures           = m_textures.size();

    // Multiple textures might share the same ImageSource, so we create a set as we go to avoid
    // duplicate counting.
    std::set<imageSource::ImageSource*> images;
    for( const std::unique_ptr<DemandTextureImpl>& tex : m_textures )
    {
        // Skip null textures
        if( tex == nullptr ) 
            continue; 

        tex->accumulateStatistics( stats, images );
    }

    for( unsigned int i = 0; i < m_pageLoader->getNumDevices() && i < Statistics::NUM_DEVICES; ++i )
    {
        if( DeviceMemoryManager* manager = getDeviceMemoryManager( i ) )
            manager->accumulateStatistics( stats.perDevice[i] );
    }

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

bool DemandLoaderImpl::isActiveDevice( unsigned int deviceIndex ) const
{
    return m_pageLoader->isActiveDevice( deviceIndex );
}

DeviceMemoryManager* DemandLoaderImpl::getDeviceMemoryManager( unsigned int deviceIndex ) const
{
    return m_pageLoader->getDeviceMemoryManager( deviceIndex );
}

MemoryPool<PinnedAllocator, RingSuballocator>* DemandLoaderImpl::getPinnedMemoryPool()
{
    return m_pageLoader->getPinnedMemoryPool();
}

void DemandLoaderImpl::setMaxTextureMemory( size_t maxMem )
{
    m_pageLoader->setMaxTextureMemory( maxMem );
}

DemandLoader* createDemandLoader( const Options& options )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    return new DemandLoaderImpl( options );
}

void destroyDemandLoader( DemandLoader* manager )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    delete manager;
}

}  // namespace demandLoading
