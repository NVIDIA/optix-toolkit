// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>

#include <optix.h>

#include <cuda.h>

#include <vector>

namespace demandGeometry {

class GeometryLoader
{
  public:
      virtual ~GeometryLoader() = default;

    /// Register a proxy for the given bounds.
    ///
    /// This method is multi-thread safe and can be called from inside the resource callback.
    ///
    /// @param  bounds      The bounding box of the proxy for the real geometry.
    ///
    /// @returns The pageId that will be requested when this proxy is intersected.
    ///
    virtual uint_t add( const OptixAabb& bounds ) = 0;

    /// Unregister a proxy for the page id.
    ///
    /// This method is multi-thread safe and can be called from inside the resource callback.
    ///
    /// @param  pageId      The page id of the proxy to remove.  This is the value returned by add().
    ///
    virtual void remove( uint_t pageId ) = 0;

    /// Copy proxy data to the device synchronously.
    virtual void copyToDevice() = 0;

    /// Copy proxy data to the device asynchronously.
    ///
    /// @param  stream      The stream on which to enqueue the copy.
    ///
    virtual void copyToDeviceAsync( CUstream stream ) = 0;

    /// Returns the requested proxy ids.
    ///
    /// After DemandLoader::processRequests and Ticket::wait has been called, all
    /// the requested proxy ids are known and can be returned by this method.
    ///
    virtual std::vector<uint_t> requestedProxyIds() const = 0;

    /// Clear the requested proxy ids, unloading their associated resources.
    ///
    /// Subsequent launches will result in a fresh set of requested proxy ids, including any
    /// previously requested proxies that were not resolved.
    ///
    virtual void clearRequestedProxyIds() = 0;
    
    /// Set the shader binding table index to be used by the proxy traversable.
    ///
    /// @param  index       The hit group index to use for the proxies.
    ///
    virtual void setSbtIndex( uint_t index )  = 0;

    /// Create the traversable for the proxies.
    ///
    /// @param  dc          The OptiX device context to use for the AS build.
    /// @param  stream      The stream used to build the traversable.
    ///
    /// @returns            The traversable used to intersect proxies.
    ///
    virtual OptixTraversableHandle createTraversable( OptixDeviceContext dc, CUstream stream ) = 0;

    /// Get the proxy instance context for the device.
    ///
    /// This is typically held as a member in your launch parameters and
    /// made available via the implementation of ::demandGeometry::app::getContext.
    ///
    /// @returns    Context structure.
    ///
    virtual Context getContext() const = 0;

    /// Get the name of the closest hit program.
    virtual const char* getCHFunctionName() const = 0;

    /// Get the name of the intersection program.
    virtual const char* getISFunctionName() const = 0;

    /// Return the maximum number of attributes used by IS and CH programs.
    virtual int getNumAttributes() const = 0;

    /// Return whether or not proxy ids are recycled as they are removed.
    virtual bool getRecycleProxyIds() const = 0;

    /// Enable or disable whether or not proxy ids are recycled as they are removed.
    /// The default is to not recycle proxy ids.
    virtual void setRecycleProxyIds( bool enable ) = 0;
};

}  // namespace demandGeometry
