// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    virtual void setPageTableEntry( unsigned int pageId, bool evictable, unsigned long long pageTableEntry ) = 0;

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
