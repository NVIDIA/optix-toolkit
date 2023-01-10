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
#include "Util/TraceFile.h"
#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>

#include <cuda.h>

#include <algorithm>
#include <set>

namespace demandLoading {

DemandLoaderImpl::DemandLoaderImpl( const Options& options )
    : m_pageTableManager( std::make_shared<PageTableManager>( options.numPages ) )
    , m_requestProcessor( m_pageTableManager, options )
    , m_pageLoader( new DemandPageLoaderImpl( m_pageTableManager, &m_requestProcessor, options ) )
    , m_baseColorRequestHandler( this )
    , m_samplerRequestHandler( this )
    , m_pinnedMemoryManager( options )
{
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
    m_textures.emplace_back( new DemandTextureImpl( textureId, m_pageLoader->getNumDevices(), textureDesc, imageSource, this ) );

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
                DemandTextureImpl* tex = new DemandTextureImpl( textureId, m_pageLoader->getNumDevices(), textureDescs[imageIndex], imageSources[imageIndex], this );
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

unsigned int DemandLoaderImpl::createResource( unsigned int numPages, ResourceCallback callback )
{
    return m_pageLoader->createResource( numPages, callback );
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
    TilePool*     tilePool     = getDeviceMemoryManager( deviceIndex )->getTilePool();
    PageMapping   mapping;

    while( tilePool->getTotalFreeTiles() < tilePool->getDesiredFreeTiles() )
    {
        pagingSystem->activateEviction( true );
        if( pagingSystem->freeStagedPage( &mapping ) )
        {
            unmapTileResource( deviceIndex, stream, mapping.id );
            tilePool->freeBlock( mapping.page );
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

    if( memoryType == CU_MEMORYTYPE_HOST )
    {
        DEMAND_ASSERT_MSG( size <= sizeof( MipTailBuffer ), "Requested buffer size too large." );
        if( size <= sizeof(TileBuffer) )
        {
            PinnedItemPool<TileBuffer>* pinnedTilePool = m_pinnedMemoryManager.getPinnedTilePool();
            char* buffer = reinterpret_cast<char*>( pinnedTilePool->allocate() );
            return TransferBufferDesc{ deviceIndex, memoryType, buffer, sizeof(TileBuffer) };
        }
        if( size < sizeof( MipTailBuffer ) )
        {
            PinnedItemPool<MipTailBuffer>* pinnedMipTailPool = m_pinnedMemoryManager.getPinnedMipTailPool();
            char* buffer = reinterpret_cast<char*>( pinnedMipTailPool->allocate() );
            return TransferBufferDesc{ deviceIndex, memoryType, buffer, sizeof(MipTailBuffer) };
        }
    }
    else if( memoryType == CU_MEMORYTYPE_DEVICE )
    {
        char* ptr;
#if OTK_USE_CUDA_MEMORY_POOLS
        DEMAND_CUDA_CHECK( cuMemAllocAsync( reinterpret_cast<CUdeviceptr*>( &ptr ), size, stream ) );
#else
        DEMAND_CUDA_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &ptr ), size ) );
#endif
        return TransferBufferDesc{ deviceIndex, memoryType, ptr, size };
    }
    return TransferBufferDesc{};
}

 
void DemandLoaderImpl::freeTransferBuffer( const TransferBufferDesc& transferBuffer, CUstream stream )
{
    // Note: This doesn't immediately reclaim the buffer. An event is recorded on
    // the stream, and the buffer isn't reused until all preceding operations are complete.

    if( transferBuffer.memoryType == CU_MEMORYTYPE_HOST )
    {
        DEMAND_ASSERT_MSG( transferBuffer.size <= sizeof( MipTailBuffer ), "Buffer size too large." );
        if( transferBuffer.size <= sizeof(TileBuffer) )
        {
            PinnedItemPool<TileBuffer>* pinnedTilePool = m_pinnedMemoryManager.getPinnedTilePool();
            pinnedTilePool->free( reinterpret_cast<TileBuffer*>( transferBuffer.buffer ), transferBuffer.deviceIndex, stream );
        }
        else if( transferBuffer.size <= sizeof( MipTailBuffer ) )
        {
            PinnedItemPool<MipTailBuffer>* pinnedMipTailPool = m_pinnedMemoryManager.getPinnedMipTailPool();
            pinnedMipTailPool->free( reinterpret_cast<MipTailBuffer*>( transferBuffer.buffer ), transferBuffer.deviceIndex, stream );
        }
    }
    else if( transferBuffer.memoryType == CU_MEMORYTYPE_DEVICE )
    {
#if OTK_USE_CUDA_MEMORY_POOLS
        DEMAND_CUDA_CHECK( cuMemFreeAsync( reinterpret_cast<CUdeviceptr>(transferBuffer.buffer), stream ) );
#else 
        DEMAND_CUDA_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>(transferBuffer.buffer) ) );
#endif
    }
    else 
    {
        DEMAND_ASSERT_MSG( false, "Unknown memory type." );
    }
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
            
        // Get the size of the texture, and number of bytes read
        imageSource::ImageSource* image = tex->getImageSource();
        if( images.find( image ) == images.end() )
        {
            images.insert( image );
            stats.numTilesRead += image->getNumTilesRead();
            stats.numBytesRead += image->getNumBytesRead();
            stats.readTime += image->getTotalReadTime();

            // Calculate size (total number of bytes) in virtual texture
            imageSource::TextureInfo info = image->getInfo();
            if( info.isValid )  // texture initialized
            {
                stats.virtualTextureBytes += getTextureSizeInBytes( info );
            }
        }

        // Get the number of bytes filled (transferred) per device
        const std::vector<SparseTexture>& sparseTextures = tex->getSparseTextures();
        for(unsigned int i=0; i<sparseTextures.size(); ++i)
        {
            stats.bytesTransferredPerDevice[i] += sparseTextures[i].getNumBytesFilled();
            stats.numEvictionsPerDevice[i] += sparseTextures[i].getNumUnmappings();
        }

        const std::vector<DenseTexture>& denseTextures = tex->getDenseTextures();
        for(unsigned int i=0; i<denseTextures.size(); ++i)
        {
            stats.bytesTransferredPerDevice[i] += denseTextures[i].getNumBytesFilled();
            // Count memory used per device for dense texture data
            if( denseTextures[i].isInitialized() && denseTextures[i].getTextureObject() != 0 )
            {
                imageSource::TextureInfo info = image->getInfo();
                stats.memoryUsedPerDevice[i] += getTextureSizeInBytes( info );
            }
        }
    }

    size_t maxNumDevices = sizeof( Statistics::memoryUsedPerDevice ) / sizeof( size_t );
    for( unsigned int i = 0; i < m_pageLoader->getNumDevices() && i < maxNumDevices; ++i )
    {
        if( DeviceMemoryManager* manager = getDeviceMemoryManager( i ) )
            stats.memoryUsedPerDevice[i] += manager->getTotalDeviceMemory();
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
