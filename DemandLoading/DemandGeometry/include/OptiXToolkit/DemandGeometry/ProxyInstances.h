// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/Resource.h>

#include <optix.h>

#include <cuda.h>

#include <functional>
#include <mutex>
#include <vector>

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>
#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>

namespace demandLoading {

class DemandLoader;

}  // namespace demandLoading

namespace demandGeometry {

class ProxyInstances : public GeometryLoader
{
  public:
    static constexpr uint_t PAGE_CHUNK_SIZE = 16U;

    ProxyInstances( demandLoading::DemandLoader* loader );
    ~ProxyInstances() override = default;

    /// Register a proxy for the given bounds.
    ///
    /// This method is multi-thread safe and can be called from inside the resource callback.
    ///
    /// @param  bounds      The bounding box of the proxy for the real geometry.
    ///
    /// @returns The pageId that will be requested when this proxy is intersected.
    ///
    uint_t add( const OptixAabb& bounds ) override;

    /// Unregister a proxy for the page id.
    ///
    /// This method is multi-thread safe and can be called from inside the resource callback.
    ///
    /// @param  pageId      The page id of the proxy to remove.  This is the value returned by add().
    ///
    void remove( uint_t pageId ) override;

    /// Copy proxy data to the device synchronously.
    void copyToDevice() override;

    /// Copy proxy data to the device asynchronously.
    ///
    /// @param  stream      The stream on which to enqueue the copy.
    ///
    void copyToDeviceAsync( CUstream stream ) override;

    /// Returns the requested proxy ids.
    ///
    /// After DemandLoader::processRequests and Ticket::wait has been called, all
    /// the requested proxy ids are known and can be returned by this method.
    ///
    std::vector<uint_t> requestedProxyIds() const override;

    /// Clear the requested proxy ids, unloading their associated resources.
    ///
    /// Subsequent launches will result in a fresh set of requested proxy ids, including any
    /// previously requested proxies that were not resolved.
    ///
    void clearRequestedProxyIds() override;

    /// Set the shader binding table index to be used by the proxy traversable.
    ///
    /// @param  index       The hit group index to use for the proxies.
    ///
    void setSbtIndex( uint_t index )  override { m_sbtIndex = index; }

    /// Create the traversable for the proxies.
    ///
    /// @param  dc          The OptiX device context to use for the AS build.
    /// @param  stream      The stream used to build the traversable.
    ///
    /// @returns            The traversable used to intersect proxies.
    ///
    OptixTraversableHandle createTraversable( OptixDeviceContext dc, CUstream stream ) override;

    /// Get the proxy instance context for the device.
    ///
    /// This is typically held as a member in your launch parameters and
    /// made available via the implementation of ::demandGeometry::app::getContext.
    ///
    /// @returns    Context structure.
    ///
    Context getContext() const override { return { m_proxyData.typedDevicePtr() }; }

    /// Get the name of the closest hit program.
    const char* getCHFunctionName() const override;

    /// Get the name of the intersection program.
    const char* getISFunctionName() const override;

    /// Return the maximum number of attributes used by IS and CH programs.
    int getNumAttributes() const override;

    /// Return whether or not proxy ids are recycled as they are removed.
    bool getRecycleProxyIds() const override { return m_recycleProxyIds; }

    /// Enable or disable whether or not proxy ids are recycled as they are removed.
    /// The default is to not recycle proxy ids.
    void setRecycleProxyIds( bool enable ) override { m_recycleProxyIds = enable; }

  private:
    struct PageIdRange
    {
        uint_t m_size;
        uint_t m_start;
        uint_t m_used;
    };

    static bool s_callback( CUstream stream, unsigned int pageId, void* context, void** pageTableEntry )
    {
        return static_cast<ProxyInstances*>( context )->callback( stream, pageId, pageTableEntry );
    }

    bool callback( CUstream stream, uint_t pageId, void** pageTableEntry );

    uint_t allocateResource();
    void   deallocateResource( uint_t pageId );

    uint_t insertResource( uint_t pageId );

    void                   createProxyGeomAS( OptixDeviceContext dc, CUstream stream );
    OptixTraversableHandle createProxyInstanceAS( OptixDeviceContext dc, CUstream stream );

    mutable std::mutex m_proxyDataMutex;  // protects the CPU proxy data structures.

    demandLoading::DemandLoader* m_loader;
    std::vector<PageIdRange>     m_pageRanges;
    std::vector<uint_t>          m_freePages;

    otk::SyncVector<OptixAabb> m_primitiveBounds;
    otk::SyncVector<OptixAabb> m_proxyData;
    std::vector<uint_t>        m_proxyPageIds;
    otk::DeviceBuffer          m_devTempAccelBuffer;
    otk::DeviceBuffer          m_devProxyGeomAccelBuffer;
    OptixTraversableHandle     m_proxyGeomTraversable{};

    otk::SyncVector<OptixInstance> m_proxyInstances;
    otk::DeviceBuffer              m_devProxyInstanceAccelBuffer;
    OptixTraversableHandle         m_proxyInstanceTraversable{};
    otk::SyncVector<uint32_t>      m_sbtIndices;

    uint_t m_sbtIndex{};

    std::vector<uint_t> m_requestedResources;

    bool m_recycleProxyIds{};
};

}  // namespace demandGeometry
