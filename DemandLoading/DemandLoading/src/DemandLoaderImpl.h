// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DemandLoader.h>

#include "DemandPageLoaderImpl.h"
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/MemoryPool.h>
#include <OptiXToolkit/Memory/RingSuballocator.h>
#include "PageTableManager.h"
#include "PagingSystem.h"
#include "ThreadPoolRequestProcessor.h"
#include "ResourceRequestHandler.h"
#include "Textures/DemandTextureImpl.h"
#include "Textures/SamplerRequestHandler.h"
#include "Textures/CascadeRequestHandler.h"
#include <OptiXToolkit/DemandLoading/TextureCascade.h>
#include "TransferBufferDesc.h"

#include <cuda.h>

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace imageSource {
class ImageSource;
}

namespace demandLoading {

struct DeviceContext;
class DeviceMemoryManager;
class DemandTexture;
class RequestProcessor;
struct TextureDescriptor;

/// DemandLoader demonstrates how to implement demand-loaded textures using the OptiX paging library.
class DemandLoaderImpl : public DemandLoader
{
  public:
    /// Construct demand loading sytem.
    DemandLoaderImpl( const Options& options );

    /// Destroy demand loading system.
    ~DemandLoaderImpl() override;

    /// Create a demand-loaded texture for the given image.  The texture initially has no backing
    /// storage.  The readTile() method is invoked on the image to fill each required tile.  The
    /// ImageSource pointer is retained indefinitely.
    const DemandTexture& createTexture( std::shared_ptr<imageSource::ImageSource> image,
                                        const TextureDescriptor&                  textureDesc ) override;

    /// Create a demand-loaded UDIM texture for a given set of images.  If a baseTexture is used,
    /// it should be created first by calling createTexture.  The id of the returned texture should be
    /// used when calling tex2DGradUdim.  This will create demand-loaded textures for each image
    /// supplied, and all of the image readers are retained for the lifetime of the DemandLoader.
    const DemandTexture& createUdimTexture( std::vector<std::shared_ptr<imageSource::ImageSource>>& imageSources,
                                            std::vector<TextureDescriptor>&                         textureDescs,
                                            unsigned int                                            udim,
                                            unsigned int                                            vdim,
                                            int                                                     baseTextureId,
                                            unsigned int                                            numChannelTextures = 1 ) override;

    /// Create an arbitrary resource with the specified number of pages.  \see ResourceCallback.
    unsigned int createResource( unsigned int numPages, ResourceCallback callback, void* callbackContext ) override;

    /// Invalidate a page in an arbitrary resource.
    void invalidatePage( unsigned int pageId ) override;
    
    /// Load  or reload all texture tiles in a texture.
    void loadTextureTiles( CUstream stream, unsigned int textureId, bool reloadIfResident ) override;

    /// Schedule a list of textures to be unloaded when launchPrepare is called next.
    void unloadTextureTiles( unsigned int textureId ) override;

    void migrateTextureTiles( const TextureSampler& oldSampler, DemandTextureImpl* newTexture );

    /// Replace the indicated texture, clearing out the old texture as needed
    void replaceTexture( CUstream                                  stream,
                         unsigned int                              textureId,
                         std::shared_ptr<imageSource::ImageSource> image,
                         const TextureDescriptor&                  textureDesc,
                         bool                                      migrateTiles ) override;

    /// Pre-initialize the texture.  The caller must ensure that the current CUDA context matches the given stream.
    void initTexture( CUstream stream, unsigned int textureId ) override;

    /// Pre-initialize all of the subtextures in the udim grid, as well as the base texture.
    void initUdimTexture( CUstream stream, unsigned int baseTextureId ) override;

    /// Get the page id associated with with the given texture tile. Return MAX_INT if the texture is not initialized.
    unsigned int getTextureTilePageId( unsigned int textureId, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) override;

    /// Get the starting mip level of the mip tail
    unsigned int getMipTailFirstLevel( unsigned int textureId ) override;

    /// Load or reload a texture tile.  The caller must ensure that the current CUDA context matches
    /// the given stream.
    void loadTextureTile( CUstream stream, unsigned int textureId, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) override;

    /// Return true if the requested page is resident on the device corresponding to the current
    /// CUDA context.
    bool pageResident( unsigned int pageId ) override;

    /// Prepare for launch.  The caller must ensure that the current CUDA context matches the given
    /// stream.  Returns false if the corresponding device does not support sparse textures.  If
    /// successful, returns a DeviceContext via result parameter, which should be copied to device
    /// memory (typically along with OptiX kernel launch parameters), so that it can be passed to
    /// Tex2D().
    bool launchPrepare( CUstream stream, DeviceContext& demandTextureContext ) override;

    /// Fetch page requests from the given device context and enqueue them for background
    /// processing.  The caller must ensure that the current CUDA context matches the given stream.
    /// The given DeviceContext must reside in host memory.  The given stream is used when copying
    /// tile data to the device.  Returns a ticket that is notified when the requests have been
    /// filled on the host side.
    Ticket processRequests( CUstream stream, const DeviceContext& deviceContext ) override;

    /// Abort demand loading, with minimal cleanup and no CUDA calls.  Halts asynchronous request
    /// processing.  Useful in case of catastrophic CUDA error or corruption.
    void abort() override;

    /// Get time/space stats for the DemandLoader.
    Statistics getStatistics() const override;

    /// Get the demand loading configuration options.
    const Options& getOptions() const override { return *m_options; }

    /// Turn on or off eviction
    void enableEviction( bool evictionActive ) override;

    /// Set the max memory per device to be used for texture tiles, deleting memory arenas if needed
    void setMaxTextureMemory( size_t maxMem ) override;

    /// Get the DeviceMemoryManager for the current CUDA context.
    DeviceMemoryManager* getDeviceMemoryManager() const;

    /// Get the pinned memory manager.
    otk::MemoryPool<otk::PinnedAllocator, otk::RingSuballocator>* getPinnedMemoryPool();
    
    /// Get the specified texture.
    DemandTextureImpl* getTexture( unsigned int textureId ) { return m_textures.at( textureId ).get(); }

    /// Get the PagingSystem for the current CUDA context.
    PagingSystem* getPagingSystem() const;
    
    /// Get the PageTableManager.
    PageTableManager* getPageTableManager();

    /// Free some staged tiles if there are some that are ready
    void freeStagedTiles( CUstream stream );

    /// Allocate a temporary buffer of the given memory type, used as a staging point for an asset such as a texture tile.
    const TransferBufferDesc allocateTransferBuffer( CUmemorytype memoryType, size_t size, CUstream stream );

    /// Free a temporary buffer after current work in the stream finishes 
    void freeTransferBuffer( const TransferBufferDesc& transferBuffer, CUstream stream );

    /// Set the value of a page table entry (does not take effect until launchPrepare is called).
    /// It's usually not necessary to call this.  It is helpful for asynchronous resource request
    /// handling, in which a ResourceCallback enqueues a request and returns false, indicating that
    /// the request has not yet been satisfied.  Later, when the request has been processed,
    /// setPageTableEntry is called to update the page table.
    void setPageTableEntry( unsigned int pageId, bool evictable, unsigned long long pageTableEntry ) override;

    /// Get the CUDA context associated with this demand loader
    CUcontext getCudaContext() override { return m_cudaContext; }

  private:
    mutable std::mutex       m_mutex;
    std::shared_ptr<Options> m_options;
    CUcontext                m_cudaContext;  // The demand loader is for this context

    std::shared_ptr<PageTableManager>     m_pageTableManager;  // Allocates ranges of virtual pages.
    ThreadPoolRequestProcessor            m_requestProcessor;  // Asynchronously processes page requests.
    std::unique_ptr<DemandPageLoaderImpl> m_pageLoader;

    std::map<unsigned int, std::unique_ptr<DemandTextureImpl>> m_textures; // demand-loaded textures, indexed by textureId
    std::map<imageSource::ImageSource*, unsigned int> m_imageToTextureId;  // lookup from image* to textureId

    SamplerRequestHandler m_samplerRequestHandler;  // Handles requests for texture samplers.
    CascadeRequestHandler m_cascadeRequestHandler;  // Handles cascading texture sizes.

    otk::MemoryPool<otk::DeviceAsyncAllocator, otk::RingSuballocator> m_deviceTransferPool;

    std::vector<std::unique_ptr<ResourceRequestHandler>> m_resourceRequestHandlers;  // Request handlers for arbitrary resources.

    unsigned int m_ticketId{};

    // Unmap the backing storage associated with a texture tile or mip tail
    void unmapTileResource( CUstream stream, unsigned int pageId );

    // Create a normal or variant version of a demand texture, based on the imageSource 
    DemandTextureImpl* makeTextureOrVariant( unsigned int textureId, const TextureDescriptor& textureDesc, std::shared_ptr<imageSource::ImageSource>& imageSource );

    // Allocate pages for a number of textures (samplers and base colors)
    unsigned int allocateTexturePages( unsigned int numTextures );
};

}  // namespace demandLoading
