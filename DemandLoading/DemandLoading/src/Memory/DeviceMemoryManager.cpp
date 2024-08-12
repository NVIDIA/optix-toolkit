// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DeviceMemoryManager.h"
#include "DeviceContextImpl.h"

#include <OptiXToolkit/Error/ErrorCheck.h>

using namespace otk;

namespace demandLoading {

static const unsigned int SAMPLER_POOL_ALLOC_SIZE = 65536;

DeviceMemoryManager::DeviceMemoryManager( std::shared_ptr<Options> options )
    : m_options( options )
    , m_samplerPool( new DeviceAllocator(), new FixedSuballocator( sizeof( TextureSampler ), alignof( TextureSampler ) ), SAMPLER_POOL_ALLOC_SIZE )
    , m_deviceContextMemory( new DeviceAllocator(), nullptr )
    , m_whiteBlackTiles( static_cast<int>( WB_NONE ), otk::TileBlockHandle{0, 0} )
{
    if( m_options->useSparseTextures )
    {
        m_tilePool.reset( new TilePool( new TextureTileAllocator(), new HeapSuballocator(),
                                        TextureTileAllocator::getRecommendedAllocationSize(), m_options->maxTexMemPerDevice ) );
    }
}

DeviceMemoryManager::~DeviceMemoryManager()
{
    for( DeviceContext* context : m_deviceContextPool )
        delete context;
    
    // No need to delete the members of the contexts, since they are pool allocated.
}

DeviceContext* DeviceMemoryManager::allocateDeviceContext()
{
    if( !m_deviceContextFreeList.empty() )
    {
        DeviceContext* context = m_deviceContextFreeList.back();
        m_deviceContextFreeList.pop_back();
        return context;
    }

    DeviceContextImpl* context = new DeviceContextImpl();
    context->poolIndex         = static_cast<unsigned int>( m_deviceContextPool.size() );
    m_deviceContextPool.push_back( context );

    // Per device data is shared by all contexts for the device.
    if( m_deviceContextPool.size() > 1 )
        context->setPerDeviceData( *m_deviceContextPool[0] );
    else
        context->allocatePerDeviceData( &m_deviceContextMemory, *m_options );

    // Each context gets its own copy of per stream data.
    context->allocatePerStreamData( &m_deviceContextMemory, *m_options );

    return context;
}

void DeviceMemoryManager::freeDeviceContext( DeviceContext* context )
{
    // The pool index is recorded to permit a copied DeviceContext to be returned to the pool.
    // Compare the page table pointer as a sanity check.
    OTK_ASSERT_MSG( context, "Null context in DeviceContextPool::free" );
    OTK_ASSERT_MSG( context->pageTable.data == m_deviceContextPool.at( context->poolIndex )->pageTable.data,
                       "Invalid context in DeviceContextPool::free" );
    m_deviceContextFreeList.push_back( m_deviceContextPool[context->poolIndex] );
}

}  // namespace demandLoading
