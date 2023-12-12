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
#include "Util/TraceFile.h"

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

#if CUDA_VERSION >= 11020
#define DEVICE_MEMORY_POOL_ALLOCATOR otk::DeviceAsyncAllocator
#else
#define DEVICE_MEMORY_POOL_ALLOCATOR otk::DeviceAllocator
#endif

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
    const DemandTexture& createTexture( std::shared_ptr<imageSource::ImageSource> image, const TextureDescriptor& textureDesc ) override;

    /// Create a demand-loaded UDIM texture for a given set of images.  If a baseTexture is used,
    /// it should be created first by calling createTexture.  The id of the returned texture should be
    /// used when calling tex2DGradUdim.  This will create demand-loaded textures for each image
    /// supplied, and all of the image readers are retained for the lifetime of the DemandLoader.
    const DemandTexture& createUdimTexture( std::vector<std::shared_ptr<imageSource::ImageSource>>& imageSources,
                                            std::vector<TextureDescriptor>&                         textureDescs,
                                            unsigned int                                            udim,
                                            unsigned int                                            vdim,
                                            int baseTextureId ) override;

    /// Create an arbitrary resource with the specified number of pages.  \see ResourceCallback.
    unsigned int createResource( unsigned int numPages, ResourceCallback callback, void* callbackContext ) override;

    /// Schedule a list of textures to be unloaded when launchPrepare is called next.
    void unloadTextureTiles( unsigned int textureId ) override;

    /// Replace the indicated texture, clearing out the old texture as needed
    void replaceTexture( CUstream stream, unsigned int textureId, std::shared_ptr<imageSource::ImageSource> image, const TextureDescriptor& textureDesc ) override;

    /// Pre-initialize the texture.  The caller must ensure that the current CUDA context matches
    /// the given stream.
    void initTexture( CUstream stream, unsigned int textureId ) override;

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

    /// Replay the given page requests (from a trace file), adding them to the page requeuest queue
    /// for asynchronous processing.  The caller must ensure that the current CUDA context matches
    /// the given stream.  Returns a ticket that is notified when the requests have been filled.
    Ticket replayRequests( CUstream stream, unsigned int* requestedPages, unsigned int numRequestedPages );

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

    void setPageTableEntry( unsigned int pageId, bool evictable, void* pageTableEntry );

    /// Get the CUDA context associated with this demand loader
    virtual CUcontext getCudaContext() { return m_cudaContext; }

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

    otk::MemoryPool<DEVICE_MEMORY_POOL_ALLOCATOR, otk::RingSuballocator> m_deviceTransferPool;

    std::vector<std::unique_ptr<ResourceRequestHandler>> m_resourceRequestHandlers;  // Request handlers for arbitrary resources.

    unsigned int m_ticketId{};

    // Unmap the backing storage associated with a texture tile or mip tail
    void unmapTileResource( CUstream stream, unsigned int pageId );

    // Create a normal or variant version of a demand texture, based on the imageSource 
    DemandTextureImpl* makeTextureOrVariant( unsigned int textureId, const TextureDescriptor& textureDesc, std::shared_ptr<imageSource::ImageSource>& imageSource );

    unsigned int allocateTexturePages( unsigned int numTextures );
};

}  // namespace demandLoading
