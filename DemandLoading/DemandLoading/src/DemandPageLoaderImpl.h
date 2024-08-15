// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DemandPageLoader.h>

#include <OptiXToolkit/Memory/Allocators.h>
#include "Memory/DeviceMemoryManager.h"
#include <OptiXToolkit/Memory/MemoryPool.h>
#include <OptiXToolkit/Memory/RingSuballocator.h>
#include "PageTableManager.h"
#include "PagingSystem.h"
#include "ResourceRequestHandler.h"
#include "Textures/DemandTextureImpl.h"
#include "Textures/SamplerRequestHandler.h"
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
class DemandTexture;
class RequestProcessor;
struct TextureDescriptor;

/// DemandLoader demonstrates how to implement demand-loaded textures using the OptiX paging library.
class DemandPageLoaderImpl : public DemandPageLoader
{
  public:
    /// Construct demand loading sytem.
    DemandPageLoaderImpl( RequestProcessor* requestProcessor, std::shared_ptr<Options> options );

    DemandPageLoaderImpl( std::shared_ptr<PageTableManager> pageTableManager, RequestProcessor *requestProcessor, std::shared_ptr<Options> options );

    /// Destroy demand page loader.
    ~DemandPageLoaderImpl() override = default;

    /// Allocate backed or unbacked pages
    unsigned int allocatePages( unsigned int numPages, bool backed ) override;

    /// Set the value of a single page table entry
    void setPageTableEntry( unsigned int pageId, bool evictable, unsigned long long pageTableEntry ) override;

    /// Prepare for launch.  The caller must ensure that the current CUDA context matches the given
    /// stream.  Returns false if the current device does not support sparse textures.  If
    /// successful, returns a DeviceContext via result parameter, which should be copied to device
    /// memory (typically along with OptiX kernel launch parameters), so that it can be passed to
    /// Tex2D().
    bool pushMappings( CUstream stream, DeviceContext& demandTextureContext ) override;

    /// Fetch page requests from the given device context and enqueue them for background
    /// processing.  The caller must ensure that the current CUDA context matches the given stream.
    /// The given stream is used when copying tile data to the device.  Returns a ticket that is
    /// notified when the requests have been filled.
    void pullRequests( CUstream stream, const DeviceContext& deviceContext, unsigned int id ) override;

    /// Turn on or off eviction
    void enableEviction( bool evictionActive ) override { m_options->evictionActive = evictionActive; }

    /// Get the DeviceMemoryManager for the current CUDA context.
    DeviceMemoryManager* getDeviceMemoryManager() { return &m_deviceMemoryManager; }

    otk::MemoryPool<otk::PinnedAllocator, otk::RingSuballocator> *getPinnedMemoryPool() { return &m_pinnedMemoryPool; }

    /// Get the PagingSystem for the current CUDA context.
    PagingSystem* getPagingSystem() { return &m_pagingSystem; };

    void setMaxTextureMemory( size_t maxMem );

    void invalidatePageRange( unsigned int startPage, unsigned int endPage, PageInvalidatorPredicate* predicate );

    double getTotalProcessingTime() const { return m_totalProcessingTime; }

  private:
    mutable std::mutex       m_mutex;
    std::shared_ptr<Options> m_options;
    DeviceMemoryManager      m_deviceMemoryManager;

    otk::MemoryPool<otk::PinnedAllocator, otk::RingSuballocator> m_pinnedMemoryPool;

    struct InvalidationRange
    {
        unsigned int startPage;
        unsigned int endPage;
        PageInvalidatorPredicate* predicate;
    };
    std::vector<InvalidationRange> m_pagesToInvalidate;

    std::shared_ptr<PageTableManager> m_pageTableManager;  // Allocates ranges of virtual pages.
    RequestProcessor*   m_requestProcessor;  // Processes page requests.

    PagingSystem m_pagingSystem;

    std::vector<std::unique_ptr<RequestHandler>> m_requestHandlers;

    double m_totalProcessingTime{};


    // Invalidate the pages for current device in m_pagesToInvalidate
    void invalidatePages( CUstream stream, DeviceContext& context );
};

}  // namespace demandLoading
