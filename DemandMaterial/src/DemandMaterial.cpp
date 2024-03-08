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

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/Memory/SyncVector.h>

#include <mutex>
#include <vector>

namespace demandMaterial {

using uint_t = unsigned int;

class DemandMaterial : public MaterialLoader
{
  public:
    DemandMaterial( demandLoading::DemandLoader* loader )
        : m_loader( loader )
    {
    }
    ~DemandMaterial() override = default;

    const char* getCHFunctionName() const override { return "__closesthit__proxyMaterial"; }

    uint_t add() override;
    void   remove( uint_t id ) override;

    std::vector<uint_t> requestedMaterialIds() const override { return m_requestedMaterials; }

    bool getRecycleProxyIds() const override { return m_recycleProxyIds; }
    void setRecycleProxyIds( bool enable ) override { m_recycleProxyIds = enable; }

  private:
    demandLoading::DemandLoader* m_loader;
    std::vector<uint_t>          m_materialIds;
    std::vector<uint_t>          m_freeMaterialIds;
    std::vector<uint_t>          m_requestedMaterials;
    std::mutex                   m_requestedMaterialsMutex;
    bool                         m_recycleProxyIds{};

    uint_t allocateMaterialId();

    bool loadMaterial( CUstream stream, uint_t pageId, void** pageTableEntry );

    static bool callback( CUstream stream, uint_t pageIndex, void* context, void** pageTableEntry )
    {
        return static_cast<DemandMaterial*>( context )->loadMaterial( stream, pageIndex, pageTableEntry );
    }
};

uint_t DemandMaterial::allocateMaterialId()
{
    if( m_recycleProxyIds && !m_freeMaterialIds.empty() )
    {
        const uint_t materialId = m_freeMaterialIds.back();
        m_freeMaterialIds.pop_back();
        return materialId;
    }

    return m_loader->createResource( 1, callback, this );
}

uint_t DemandMaterial::add()
{
    m_materialIds.push_back( allocateMaterialId() );
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
    }

    {
        auto pos = std::lower_bound( m_requestedMaterials.begin(), m_requestedMaterials.end(), pageId );
        if( pos != m_requestedMaterials.end() )
            m_requestedMaterials.erase( pos );
    }

    if( m_recycleProxyIds )
    {
        m_freeMaterialIds.push_back( pageId );
        m_loader->unloadResource( pageId );
    }
}

bool DemandMaterial::loadMaterial( CUstream /*stream*/, uint_t pageId, void** pageTableEntry )
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

std::shared_ptr<MaterialLoader> createMaterialLoader( demandLoading::DemandLoader* loader )
{
    return std::make_shared<DemandMaterial>( loader );
}

}  // namespace demandMaterial
