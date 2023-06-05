//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "DemandMaterial.h"

#include "OptiXToolkit/DemandLoading/DemandLoader.h"

namespace demandGeometryViewer {

DemandMaterial::DemandMaterial( demandLoading::DemandLoader* loader )
    : m_loader( loader )
{
}

uint_t DemandMaterial::add()
{
    m_materialIds.push_back( m_loader->createResource( 1, callback, this ) );
    return m_materialIds.back();
}

void DemandMaterial::remove( uint_t pageId )
{
    std::lock_guard<std::mutex> lock( m_requestedMaterialsMutex );

    {
        auto pos = std::lower_bound( m_materialIds.begin(), m_materialIds.end(), pageId,
                                     []( uint_t lhs, uint_t pageId ) { return lhs < pageId; } );
        if( pos == m_materialIds.end() || *pos != pageId )
            throw std::runtime_error( "Resource not found for page " + std::to_string( pageId ) );

        m_materialIds.erase( pos );
        // TODO: reuse material id?
    }

    {
        auto pos = std::lower_bound( m_requestedMaterials.begin(), m_requestedMaterials.end(), pageId );
        if( pos != m_requestedMaterials.end() )
            m_requestedMaterials.erase( pos );
    }
}

bool DemandMaterial::loadMaterial( CUstream stream, uint_t pageId, void** pageTableEntry )
{
    std::lock_guard<std::mutex> lock( m_requestedMaterialsMutex );

    {
        auto pos = std::lower_bound( m_materialIds.begin(), m_materialIds.end(), pageId,
                                     []( uint_t lhs, uint_t pageId ) { return lhs < pageId; } );
        if( pos == m_materialIds.end() || *pos != pageId )
            throw std::runtime_error( "Callback invoked for resource " + std::to_string( pageId )
                                      + " not associated with a proxy material." );
    }

    // Deduplicate the requested resource page id.
    auto pos = std::lower_bound( m_requestedMaterials.begin(), m_requestedMaterials.end(), pageId );
    if( pos == m_requestedMaterials.end() || *pos != pageId )
        m_requestedMaterials.insert( pos, pageId );

    // The value stored in the page table doesn't matter.
    *pageTableEntry = nullptr;

    return true;
}

}  // namespace demandGeometryViewer
