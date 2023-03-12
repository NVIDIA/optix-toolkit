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

#include "Util/ContextSaver.h"
#include "Util/Exception.h"

#include <cuda.h>

#include <map>
#include <memory>

namespace demandLoading {

/// PerContextData is a std::map that associates data with CUDA contexts.    
template <typename T>
class PerContextData
{
  public:
    /// Get non-const pointer to the data associated with the current CUDA context, if any.  Returns
    /// a null pointer if no associated data was found.
    T* find() 
    {
        CUcontext context;
        DEMAND_CUDA_CHECK( cuCtxGetCurrent( &context ) );
        typename MapType::iterator it = m_map.find( context );
        return it == m_map.end() ? nullptr : it->second.get();
    }

    /// Get const pointer to the data associated with the current CUDA context, if any.  Returns
    /// a null pointer if no associated data was found.
    const T* find() const { return const_cast<PerContextData*>( this )->find(); }

    /// Store the given data (taking ownership), associating it with the current CUDA context.
    /// Any previously associated data is destroyed.  Returns pointer to the stored data.
    T* insert( std::unique_ptr<T> data )
    {
        CUcontext context;
        DEMAND_CUDA_CHECK( cuCtxGetCurrent( &context ) );
        std::pair<typename MapType::iterator, bool> pair = m_map.insert( typename MapType::value_type( context, std::move( data ) ) );
        return pair.first->second.get();
    }

    /// Apply the given functor to each data item, setting the corresponding CUDA context beforehand.
    template <typename Functor>
    void for_each( Functor functor ) const
    {
        ContextSaver contextSaver;
        for( auto& it : m_map )
        {
            DEMAND_CUDA_CHECK( cuCtxSetCurrent( it.first ) );
            functor( *it.second );
        }
    }

    /// Destroy all the per-context data.
    void clear()
    {
        // Save/restore current CUDA context.
        ContextSaver contextSaver;
        for( auto& it : m_map )
        {
            // Set CUDA context.
            DEMAND_CUDA_CHECK( cuCtxSetCurrent( it.first ) );

            // Destroy per-context data.
            it.second.reset();
        }
    }

    /// Destroy all the per-context data.
    ~PerContextData()
    {
        clear();
    }

  private:
    using MapType = std::map<CUcontext, std::unique_ptr<T>>;
    MapType m_map;
};

}  // namespace demandLoading
