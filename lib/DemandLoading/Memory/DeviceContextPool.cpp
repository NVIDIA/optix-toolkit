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

#include "Memory/DeviceContextPool.h"

#include <DemandLoading/Options.h>

namespace demandLoading {

DeviceContextPool::DeviceContextPool( unsigned int deviceIndex, const Options& options )
    : m_memory( deviceIndex )
    , m_contexts( options.maxActiveStreams )
{
    // Reserve storage.  Per-device data is reserved for only one context.
    DEMAND_ASSERT( !m_contexts.empty() );
    m_contexts[0].reservePerDeviceData( &m_memory, options );
    m_contexts[0].reservePerStreamData( &m_memory, options );
    for( size_t i = 1; i < m_contexts.size(); ++i )
    {
        m_contexts[i].reservePerStreamData( &m_memory, options );
    }

    // Allocate per-device storage for a single context and share it with the others.
    m_contexts[0].allocatePerDeviceData( &m_memory, options );
    for( size_t i = 1; i < m_contexts.size(); ++i )
    {
        m_contexts[i].setPerDeviceData( m_contexts[0] );
    }

    // Allocate per-stream storage for each context.
    for( size_t i = 0; i < m_contexts.size(); ++i )
    {
        m_contexts[i].allocatePerStreamData( &m_memory, options );

        // The pool index is recorded to permit a copied DeviceContext to be returned to the pool.
        m_contexts[i].poolIndex = static_cast<unsigned int>( i );
    }
    m_memory.setToZero();
}

DeviceContext* DeviceContextPool::allocate()
{
    // Use most recently freed context, if any.
    if( !m_freeList.empty() )
    {
        DeviceContext* context = m_freeList.back();
        m_freeList.pop_back();
        return context;
    }

    // Otherwise use the next available context.
    if( m_nextAvailable < m_contexts.size() )
    {
        return &m_contexts[m_nextAvailable++];
    }

    DEMAND_ASSERT_MSG(
        false, "Maximum number of DeviceContexts exceeded (increase demandLoading::Options::maxActiveStreams)" );
    return nullptr;
}

void DeviceContextPool::free( DeviceContext* context )
{
    DEMAND_ASSERT_MSG( context, "Null context in DeviceContextPool::free" );

    // The pool index is recorded to permit a copied DeviceContext to be returned to the pool.
    // Compare the page table pointer as a sanity check.
    DEMAND_ASSERT_MSG( context->pageTable.data == m_contexts.at(context->poolIndex).pageTable.data,
                       "Invalid context in DeviceContextPool::free" );

    // Since the given context might have been copied, the free list entry is a pointer to the
    // original context, as indicated by the pool index.
    m_freeList.push_back( &m_contexts[context->poolIndex] );
}

}  // namespace demandLoading
