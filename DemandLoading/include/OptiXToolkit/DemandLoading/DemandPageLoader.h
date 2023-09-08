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

/// \file DemandLoader.h 
/// Primary interface of the Demand Loading library.

#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/Resource.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/DemandLoading/Ticket.h>

#include <cuda.h>

#include <memory>
#include <vector>

namespace imageSource {
class ImageSource;
}

namespace demandLoading {

class RequestProcessor;

/// DemandPageLoader loads pages on demand.
class DemandPageLoader
{
  public:
    /// Base class destructor.
    virtual ~DemandPageLoader() = default;

    /// Allocate a contiguous range of page ids.  Returns the first page id in the the allocated range.
    virtual unsigned int allocatePages( unsigned int numPages, bool backed ) = 0;

    /// Set the page table entry for the given page.  Sets the associated page as resident.
    virtual void setPageTableEntry( unsigned int pageId, bool evictable, void* pageTableEntry ) = 0;

    /// Prepare for launch by pushing mapped pages to the device.  The caller must ensure that the
    /// current CUDA context matches the given stream.  Returns false if the specified device does
    /// not support sparse textures.  If successful, returns a DeviceContext via result parameter,
    /// which should be copied to device memory (typically along with OptiX kernel launch
    /// parameters), so that it can be passed to Tex2D().
    virtual bool pushMappings( CUstream stream, DeviceContext& context ) = 0;

    /// Fetch page requests from the given device context and enqueue them for background
    /// processing.  The caller must ensure that the current CUDA context matches the given stream.
    /// The given DeviceContext must reside in host memory.  The given stream is used to launch a
    /// kernel to obtain requested page ids and asynchronously copy them to host memory.
    virtual void pullRequests( CUstream stream, const DeviceContext& deviceContext, unsigned int id ) = 0;

    /// Turn on or off eviction
    virtual void enableEviction( bool evictionActive ) = 0;
};

/// Create a DemandLoader with the given options.  
DemandPageLoader* createDemandPageLoader( RequestProcessor* requestProcessor, const Options& options );

/// Function to destroy a DemandLoader.
void destroyDemandPageLoader( DemandPageLoader* manager );

}  // namespace demandLoading
