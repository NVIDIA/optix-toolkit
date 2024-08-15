// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    void                clearRequestedMaterialIds() override;

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

void DemandMaterial::clearRequestedMaterialIds()
{
    std::lock_guard<std::mutex> lock( m_requestedMaterialsMutex );

    if( m_recycleProxyIds )
    {
        m_freeMaterialIds.insert( m_freeMaterialIds.end(), m_requestedMaterials.begin(), m_requestedMaterials.end() );
    }
    m_requestedMaterials.clear();
}

void DemandMaterial::remove( uint_t pageId )
{
    // Set page table entry for the requested page, ensuring that it won't be requested again.
    m_loader->setPageTableEntry( pageId, /*evictable=*/true, 0LL /* value doesn't matter */ );
    
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
        m_loader->invalidatePage( pageId );
    }
}

bool DemandMaterial::loadMaterial( CUstream /*stream*/, uint_t pageId, void** /*pageTableEntry*/ )
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

    // The callback returns false, indicating that the request has not yet been satisfied.  Later,
    // when the material has been loaded, setPageTableEntry is called to update the page table.
    return false;
}

std::shared_ptr<MaterialLoader> createMaterialLoader( demandLoading::DemandLoader* loader )
{
    return std::make_shared<DemandMaterial>( loader );
}

}  // namespace demandMaterial
