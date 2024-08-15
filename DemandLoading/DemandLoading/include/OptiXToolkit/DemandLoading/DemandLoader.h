// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file DemandLoader.h 
/// Primary interface of the Demand Loading library.

#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/Resource.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/Statistics.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/DemandLoading/Ticket.h>

#include <cuda.h>

#include <memory>
#include <vector>

namespace imageSource {
class ImageSource;
}

namespace demandLoading {

/// DemandLoader loads sparse textures on demand.
class DemandLoader
{
  public:
    /// Base class destructor.
    virtual ~DemandLoader() = default;

    /// Create a demand-loaded texture for the given image.  The texture initially has no backing
    /// storage.  The readTile() method is invoked on the image to fill each required tile.  The
    /// ImageSource pointer is retained for the lifetime of the DemandLoader.
    virtual const DemandTexture& createTexture( std::shared_ptr<imageSource::ImageSource> image,
                                                const TextureDescriptor&                  textureDesc ) = 0;

    /// Create a demand-loaded UDIM texture for a given set of images.  If a baseTexture is used,
    /// it should be created first by calling createTexture.  The id of the returned texture should be used
    /// when calling tex2DGradUdim.  All of the image readers are retained for the lifetime of the DemandLoader.
    virtual const DemandTexture& createUdimTexture( std::vector<std::shared_ptr<imageSource::ImageSource>>& imageSources,
                                                    std::vector<TextureDescriptor>&                         textureDescs,
                                                    unsigned int                                            udim,
                                                    unsigned int                                            vdim,
                                                    int                                                     baseTextureId,
                                                    unsigned int                                            numChannelTextures = 1 ) = 0;

    /// Create an arbitrary resource with the specified number of pages.  \see ResourceCallback.
    /// Returns the starting index of the resource in the page table.  The user-supplied callbackContext
    /// value is forwarded to the callback during request processing.
    virtual unsigned int createResource( unsigned int numPages, ResourceCallback callback, void* callbackContext ) = 0;

    /// Invalidate a page in an arbitrary resource.
    virtual void invalidatePage( unsigned int pageId ) = 0;
    
    // Load or reload all texture tiles in a texture.
    virtual void loadTextureTiles( CUstream stream, unsigned int textureId, bool reloadIfResident ) = 0;

    /// Schedule a list of textures to be unloaded when launchPrepare is called next.
    virtual void unloadTextureTiles( unsigned int textureId ) = 0;

    /// Set the value of a page table entry (does not take effect until launchPrepare is called).
    /// It's usually not necessary to call this.  It is helpful for asynchronous resource request
    /// handling, in which a ResourceCallback enqueues a request and returns false, indicating that
    /// the request has not yet been satisfied.  Later, when the request has been processed,
    /// setPageTableEntry is called to update the page table.
    virtual void setPageTableEntry( unsigned int pageId, bool evictable, unsigned long long pageTableEntry ) = 0;

    /// Replace the indicated texture, clearing out the old texture as needed
    virtual void replaceTexture( CUstream                                  stream,
                                 unsigned int                              textureId,
                                 std::shared_ptr<imageSource::ImageSource> image,
                                 const TextureDescriptor&                  textureDesc,
                                 bool                                      migrateTiles ) = 0;

    /// Pre-initialize the texture on the device corresponding to the given stream.  The caller must
    /// ensure that the current CUDA context matches the given stream.
    virtual void initTexture( CUstream stream, unsigned int textureId ) = 0;

    /// Pre-initialize all of the subtextures in the udim grid, as well as the base texture.
    virtual void initUdimTexture( CUstream stream, unsigned int baseTextureId ) = 0;

    /// Get the page id associated with with the given texture tile. Return MAX_INT if the texture is not initialized.
    virtual unsigned int getTextureTilePageId( unsigned int textureId, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) = 0;

    /// Get the starting mip level of the mip tail
    virtual unsigned int getMipTailFirstLevel( unsigned int textureId ) = 0;

    /// Load or reload a texture tile.  The caller must ensure that the current CUDA context matches
    /// the given stream.
    virtual void loadTextureTile( CUstream stream, unsigned int textureId, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) = 0;

    /// Return true if the requested page is resident on the device corresponding to the current
    /// CUDA context.
    virtual bool pageResident( unsigned int pageId ) = 0;

    /// Prepare for launch.  The caller must ensure that the current CUDA context matches the given
    /// stream.  The stream and its context are retained until the DemandLoader is destroyed.
    /// Returns false if the corresponding device does not support sparse textures.  If
    /// successful, returns a DeviceContext via result parameter, which should be copied to device
    /// memory (typically along with OptiX kernel launch parameters), so that it can be passed to
    /// Tex2D().
    virtual bool launchPrepare( CUstream stream, DeviceContext& context ) = 0;

    /// Fetch page requests from the given device context and enqueue them for background
    /// processing.  The caller must ensure that the current CUDA context matches the given stream.
    /// The stream and its context are retained until the DemandLoader is destroyed.
    /// The given DeviceContext must reside in host memory.  The given stream is used when copying
    /// tile data to the device.  Returns a ticket that is notified when the requests have been
    /// filled on the host side.
    virtual Ticket processRequests( CUstream stream, const DeviceContext& deviceContext ) = 0;

    /// Abort demand loading, with minimal cleanup and no CUDA calls.  Halts asynchronous request
    /// processing.  Useful in case of catastrophic CUDA error or corruption.
    virtual void abort() = 0;
    
    /// Get time/space stats for the DemandLoader.
    virtual Statistics getStatistics() const = 0;

    /// Get the current options
    virtual const Options& getOptions() const = 0;

    /// Turn on or off eviction
    virtual void enableEviction( bool evictionActive ) = 0;

    /// Set the max memory per device to be used for texture tiles, deleting memory arenas if needed
    virtual void setMaxTextureMemory( size_t maxMem ) = 0;

    /// Get the CUDA context associated with this demand loader
    virtual CUcontext getCudaContext() = 0;
};

/// Create a DemandLoader with the given options.  
DemandLoader* createDemandLoader( const Options& options );

/// Function to destroy a DemandLoader.
void destroyDemandLoader( DemandLoader* manager );

/// Get a bitmap of devices to use for demand loading
inline std::vector<unsigned int> getDemandLoadDevices( bool sparseOnly )
{
    return sparseOnly ? getSparseTextureDevices() : getCudaDevices();
}

}  // namespace demandLoading
