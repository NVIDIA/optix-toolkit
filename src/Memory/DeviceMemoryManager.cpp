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

#include "DeviceMemoryManager.h"
#include "DeviceContextImpl.h"

namespace demandLoading {

static const unsigned int SAMPLER_POOL_ALLOC_SIZE = 65536;

DeviceMemoryManager::DeviceMemoryManager( const Options& options )
    : m_options( options )
    , m_samplerPool( new DeviceAllocator(), new FixedSuballocator( sizeof( TextureSampler ), alignof( TextureSampler ) ), SAMPLER_POOL_ALLOC_SIZE )
    , m_deviceContextMemory( new DeviceAllocator(), nullptr )
    , m_tilePool( new TextureTileAllocator(),
                  new HeapSuballocator(),
                  TextureTileAllocator::getRecommendedAllocationSize(),
                  m_options.maxTexMemPerDevice )
{
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
        context->allocatePerDeviceData( &m_deviceContextMemory, m_options );

    // Each context gets its own copy of per stream data.
    context->allocatePerStreamData( &m_deviceContextMemory, m_options );

    return context;
}

void DeviceMemoryManager::freeDeviceContext( DeviceContext* context )
{
    // The pool index is recorded to permit a copied DeviceContext to be returned to the pool.
    // Compare the page table pointer as a sanity check.
    DEMAND_ASSERT_MSG( context, "Null context in DeviceContextPool::free" );
    DEMAND_ASSERT_MSG( context->pageTable.data == m_deviceContextPool.at( context->poolIndex )->pageTable.data,
                       "Invalid context in DeviceContextPool::free" );
    m_deviceContextFreeList.push_back( m_deviceContextPool[context->poolIndex] );
}

}  // namespace demandLoading
