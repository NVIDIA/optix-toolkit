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

#include <cstddef>
#include <vector>

#include "Memory/Allocators.h"
#include "Memory/FixedSuballocator.h"
#include "Memory/HeapSuballocator.h"
#include "Memory/MemoryPool.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/Statistics.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>

namespace demandLoading {

class DeviceMemoryManager
{
  public:
    DeviceMemoryManager( const Options& options );
    ~DeviceMemoryManager();

    /// Allocate a DeviceContext for this device.
    DeviceContext* allocateDeviceContext();
    /// Free a DeviceContext for this device.
    void freeDeviceContext( DeviceContext* context );

    /// Allocate a Sampler for this device.
    TextureSampler* allocateSampler() { return reinterpret_cast<TextureSampler*>( m_samplerPool.allocItem() ); }
    /// Free a Sampler for this device.
    void freeSampler( TextureSampler* sampler ) { m_samplerPool.freeItem( reinterpret_cast<uint64_t>( sampler ) ); }

    /// Allocate a TileBlock for this device.
    TileBlockHandle allocateTileBlock( size_t numBytes ) { return m_tilePool.allocTextureTiles( numBytes ); }
    /// Free a TileBlock for this device.
    void freeTileBlock( const TileBlockDesc& blockDesc ) { m_tilePool.freeTextureTiles( blockDesc ); }
    /// Get the memory handle associated with the tileBlock.
    CUmemGenericAllocationHandle getTileBlockHandle( const TileBlockDesc& bh )
    {
        return m_tilePool.getAllocationHandle( bh.arenaId );
    }

    /// Returns true if TileBlocks need to be freed.
    bool needTileBlocksFreed() const { return m_tilePool.allocatableSpace() < m_tilePool.allocationGranularity(); };
    /// Returns the arena size for m_tilePool.
    size_t getTilePoolArenaSize() const { return static_cast<size_t>( m_tilePool.allocationGranularity() ); }
    /// Set the max texture memory
    void setMaxTextureTileMemory( size_t maxMemory ) { m_tilePool.setMaxSize( static_cast<uint64_t>( maxMemory ) ); }

    /// Returns the amount of device memory allocated.
    size_t getTotalDeviceMemory() const
    {
        return m_samplerPool.trackedSize() + m_deviceContextMemory.trackedSize() + m_tilePool.trackedSize();
    }

    void accumulateStatistics( DeviceStatistics& stats ) const { stats.memoryUsed += getTotalDeviceMemory(); }

  private:
    Options      m_options;

    MemoryPool<DeviceAllocator, FixedSuballocator>     m_samplerPool;
    MemoryPool<DeviceAllocator, HeapSuballocator>      m_deviceContextMemory;
    MemoryPool<TextureTileAllocator, HeapSuballocator> m_tilePool;

    std::vector<DeviceContext*> m_deviceContextPool;
    std::vector<DeviceContext*> m_deviceContextFreeList;
};

}  // namespace demandLoading
