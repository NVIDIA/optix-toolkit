// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/Statistics.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Memory/Allocators.h>
#include <OptiXToolkit/Memory/FixedSuballocator.h>
#include <OptiXToolkit/Memory/HeapSuballocator.h>
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>
#include <OptiXToolkit/Memory/MemoryPool.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace demandLoading {

class DeviceMemoryManager
{
  public:
    DeviceMemoryManager( std::shared_ptr<Options> options );
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
    otk::TileBlockHandle allocateTileBlock( size_t numBytes )
    {
        OTK_ASSERT( m_tilePool );
        return m_tilePool->allocTextureTiles( numBytes );
    }

    /// Free a TileBlock for this device.
    void freeTileBlock( const otk::TileBlockDesc& blockDesc )
    {
        OTK_ASSERT( m_tilePool );
        m_tilePool->freeTextureTiles( blockDesc );
    }

    /// Get the memory handle associated with the tileBlock.
    CUmemGenericAllocationHandle getTileBlockHandle( const otk::TileBlockDesc& bh )
    {
        OTK_ASSERT( m_tilePool );
        return m_tilePool->getAllocationHandle( bh.arenaId );
    }

    /// Returns true if TileBlocks need to be freed.
    bool needTileBlocksFreed() const 
    { 
        if( !m_tilePool || m_tilePool->trackedSize() < m_tilePool->maxSize() )
            return false;
        return m_tilePool->currentFreeSpace() < ( m_options->maxStagedPages * otk::TILE_SIZE_IN_BYTES );
    }

    /// Returns the arena size for tile pool.
    size_t getTilePoolArenaSize() const { return m_tilePool ? static_cast<size_t>( m_tilePool->allocationGranularity() ) : 2 * 1024 * 1024; }

    /// Set the max texture memory
    void setMaxTextureTileMemory( size_t maxMemory )
    {
        if( m_tilePool )
            m_tilePool->setMaxSize( static_cast<uint64_t>( maxMemory ) );
    }

    /// Return the amount of device memory allocated in different pools.
    size_t getSamplerMemory() const { return m_samplerPool.trackedSize(); }
    size_t getDeviceContextMemory() const { return m_deviceContextMemory.trackedSize(); }
    size_t getTextureTileMemory() const { return m_tilePool ? m_tilePool->trackedSize() : 0; }
    size_t getTotalDeviceMemory() const { return getSamplerMemory() + getDeviceContextMemory() + getTextureTileMemory(); }

  private:
    std::shared_ptr<Options> m_options;

    using SamplerPool = otk::MemoryPool<otk::DeviceAllocator, otk::FixedSuballocator>;
    using DeviceContextPool = otk::MemoryPool<otk::DeviceAllocator, otk::HeapSuballocator>;
    using TilePool          = otk::MemoryPool<otk::TextureTileAllocator, otk::HeapSuballocator>;

    SamplerPool               m_samplerPool;
    DeviceContextPool         m_deviceContextMemory;
    std::unique_ptr<TilePool> m_tilePool; // null if sparse textures disabled.

    std::vector<DeviceContext*> m_deviceContextPool;
    std::vector<DeviceContext*> m_deviceContextFreeList;
};

}  // namespace demandLoading
