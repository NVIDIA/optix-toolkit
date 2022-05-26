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

#pragma once

/// \file DemandLoader.h 
/// Primary interface of the Demand Loading library.

#include <DemandLoading/DemandTexture.h>
#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/Options.h>
#include <DemandLoading/Resource.h>
#include <DemandLoading/Statistics.h>
#include <DemandLoading/TextureDescriptor.h>
#include <DemandLoading/Ticket.h>

#include <cuda.h>

#include <memory>
#include <vector>

namespace imageSource {
class ImageSource;
}

namespace demandLoading {

class DemandLoaderImpl;

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
                                                    int baseTextureId ) = 0;

    /// Create an arbitrary resource with the specified number of pages.  \see ResourceCallback.
    /// Returns the starting index of the resource in the page table.
    virtual unsigned int createResource( unsigned int numPages, ResourceCallback callback ) = 0;

    /// Prepare for launch.  Returns false if the specified device does not support sparse textures.
    /// If successful, returns a DeviceContext via result parameter, which should be copied to
    /// device memory (typically along with OptiX kernel launch parameters), so that it can be
    /// passed to Tex2D().
    virtual bool launchPrepare( unsigned int deviceIndex, CUstream stream, DeviceContext& context ) = 0;

    /// Fetch page requests from the given device context and enqueue them for background
    /// processing.  The given DeviceContext must reside in host memory.  The given stream is used
    /// when copying tile data to the device.  Returns a ticket that is notified when the requests
    /// have been filled on the host side.
    virtual Ticket processRequests( unsigned int deviceIndex, CUstream stream, const DeviceContext& deviceContext ) = 0;

    /// Get current statistics.
    virtual Statistics getStatistics() const = 0;

    /// Get indices of the devices that can be employed by the DemandLoader (i.e. those that support sparse textures).
    virtual const std::vector<unsigned int> getDevices() const = 0;

    /// Turn on or off eviction
    virtual void enableEviction( bool evictionActive ) = 0;
};

/// Create a DemandLoader with the given options.  
DemandLoader* createDemandLoader( const Options& options );

/// Function to destroy a DemandLoader.
void destroyDemandLoader( DemandLoader* manager );

}  // namespace demandLoading
