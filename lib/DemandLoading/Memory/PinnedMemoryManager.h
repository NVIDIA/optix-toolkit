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

#include "Memory/Buffers.h"
#include "Memory/BulkPinnedItemPool.h"
#include "Memory/PinnedItemPool.h"
#include "Memory/PinnedRequestContextPool.h"
#include "PageMappingsContext.h"

#include <DemandLoading/Options.h>
#include <DemandLoading/TextureSampler.h>

#include <algorithm>

namespace demandLoading {

using PinnedPageMappingsContextPool = BulkPinnedItemPool<PageMappingsContext, Options>;

/// PinnedMemoryManager contains separate PinnedItemPools for texture tiles, mip tails, and
/// samplers.  We use separate fixed-sized pools instead of a single ring buffer allocator to avoid
/// blocking on allocation when some fill operations take orders of magnitude longer than others.
class PinnedMemoryManager
{
  public:
    /// Construct PinnedMemoryManager, which is sized according to Options::maxPinnedMemory.
    PinnedMemoryManager( const Options& options )
        // We need two contexts per stream to push page mappings.  See PagingSystem::pushMappings.
        : m_pageMappingsContextPool( 2 * options.maxActiveStreams, options )
        , m_requestContextPool( options.maxActiveStreams, options )
        , m_mipTailPool( maxMipTailBytes( options ) / sizeof( MipTailBuffer ) )
        , m_samplerPool( maxSamplerBytes( options ) / sizeof( TextureSampler ) )
        , m_tilePool( maxTileBytes( options ) / sizeof( TileBuffer ) )
    {
    }

    /// Get the tile pool.
    PinnedItemPool<TileBuffer>* getPinnedTilePool() { return &m_tilePool; }

    /// Get the mip tail pool.
    PinnedItemPool<MipTailBuffer>* getPinnedMipTailPool() { return &m_mipTailPool; }

    /// Get the sampler pool
    PinnedItemPool<TextureSampler>* getPinnedSamplerPool() { return &m_samplerPool; }

    /// Get the page mappings context pool.
    PinnedPageMappingsContextPool* getPageMappingsContextPool() { return &m_pageMappingsContextPool; }

    /// Get the RequestContext pool.
    PinnedRequestContextPool* getRequestContextPool() { return &m_requestContextPool; }

    /// Get the total amount of pinned memory allocated.
    size_t getTotalPinnedMemory() const
    {
        return m_requestContextPool.getTotalPinnedMemory() + m_tilePool.getTotalPinnedMemory()
               + m_mipTailPool.getTotalPinnedMemory() + m_samplerPool.getTotalPinnedMemory();
    }

  private:
    PinnedPageMappingsContextPool  m_pageMappingsContextPool;
    PinnedRequestContextPool       m_requestContextPool;
    PinnedItemPool<MipTailBuffer>  m_mipTailPool;
    PinnedItemPool<TextureSampler> m_samplerPool;
    PinnedItemPool<TileBuffer>     m_tilePool;

    size_t maxMipTailBytes( const Options& options )
    {
        return std::min( static_cast<size_t>( 0.10f * options.maxPinnedMemory ), static_cast<size_t>( 256ULL * 1024 * 1024 ) );
    }

    size_t maxSamplerBytes( const Options& options )
    {
        return std::min( static_cast<size_t>( 0.01f * options.maxPinnedMemory ), static_cast<size_t>( 2ULL * 1024 * 1024 ) );
    }

    size_t maxTileBytes( const Options& options )
    {
        return static_cast<size_t>( 0.89f * options.maxPinnedMemory );
    }
};

}  // namespace demandLoading
