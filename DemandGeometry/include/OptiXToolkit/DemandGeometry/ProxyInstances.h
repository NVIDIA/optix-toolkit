//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/DemandLoading/Resource.h>

#include <optix.h>

#include <cuda.h>

#include <functional>
#include <mutex>
#include <vector>

#include "DemandGeometry.h"

#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>

namespace demandLoading {

class DemandLoader;

}  // namespace demandLoading

namespace demandGeometry {

class ProxyInstances
{
  public:
    ProxyInstances( demandLoading::DemandLoader* loader );

    std::vector<uint_t> requestedResources() const;

    /// Set the shader binding table index to be used by the proxies.
    ///
    /// @param  index       The hit group index to use for the proxies.
    ///
    void setSbtIndex( uint_t index ) { m_sbtIndex = index; }

    /// Register a proxy for the given bounds.
    ///
    /// This method is multi-thread safe and can be called from inside the resource callback.
    ///
    /// @param  bounds      The bounding box of the proxy for the real geometry.
    ///
    /// @returns The pageId that will be requested when this proxy is intersected.
    ///
    uint_t add( const OptixAabb& bounds );

    /// Unregister a proxy for the page id.
    ///
    /// This method is multi-thread safe and can be called from inside the resource callback.
    ///
    /// @param  pageId      The page id of the proxy to remove.  This is the value returned by add().
    ///
    void remove( uint_t pageId );

    /// Copy proxy data to the device synchronously.
    void copyToDevice();

    /// Copy proxy data to the device asynchronously.
    ///
    /// @param  stream      The stream on which to enqueue the copy.
    ///
    void copyToDeviceAsync( CUstream stream );

    /// Create the traversable for the proxies.
    ///
    /// @param  dc          The OptiX device context to use for the AS build.
    /// @param  stream      The stream used to build the traversable.
    ///
    /// @returns            The traversable used to intersect proxies.
    ///
    OptixTraversableHandle createTraversable( OptixDeviceContext dc, CUstream stream );

    /// Get the typed pointer to the proxy bounding box array on the device.
    ///
    /// @returns    Device pointer to the proxy bounding boxes.
    ///
    const OptixAabb* getTypedProxyDataDevicePtr() const { return m_proxyData.typedDevicePtr(); }

    /// Get the name of the closest hit program.
    const char* getCHFunctionName() const;
    /// Get the name of the intersection program.
    const char* getISFunctionName() const;
    /// Return the maximum number of attributes used by IS and CH programs.
    int getNumAttributes() const;

  private:
    struct PageIdRange
    {
        uint_t m_size;
        uint_t m_start;
        uint_t m_used;
    };
    struct Resource
    {
        uint_t pageId;
        size_t index;
    };

    static bool s_callback( CUstream stream, unsigned int pageId, void* context, void** pageTableEntry )
    {
        return static_cast<ProxyInstances*>( context )->callback( stream, pageId, pageTableEntry );
    }

    bool callback( CUstream stream, uint_t pageId, void** pageTableEntry );

    uint_t allocateResource( size_t index );

    void insertResource( uint_t pageId, size_t index );

    void                   createProxyGeomAS( OptixDeviceContext dc, CUstream stream );
    OptixTraversableHandle createProxyInstanceAS( OptixDeviceContext dc, CUstream stream );

    mutable std::mutex m_proxyDataMutex;  // protects the CPU proxy data structures.

    demandLoading::DemandLoader* m_loader;
    std::vector<PageIdRange>     m_pageRanges;
    std::vector<Resource>        m_resources;  // sorted by pageId and suitable for binary search.
    const uint_t                 PAGE_CHUNK_SIZE = 16U;

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
};

}  // namespace demandGeometry
