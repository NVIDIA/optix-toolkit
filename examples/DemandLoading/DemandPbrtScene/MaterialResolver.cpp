// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "MaterialResolver.h"

#include "DemandTextureCache.h"
#include "FrameStopwatch.h"
#include "Options.h"
#include "ProgramGroups.h"
#include "SceneGeometry.h"
#include "SceneProxy.h"
#include "SceneSyncState.h"

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <map>
#include <optional>
#include <stdexcept>

namespace demandPbrtScene {

template <typename Container>
void grow( Container& container, size_t size )
{
    if( container.size() < size )
    {
        container.resize( size );
    }
}

namespace {

class PbrtMaterialResolver : public MaterialResolver
{
  public:
    PbrtMaterialResolver( const Options& options, MaterialLoaderPtr materialLoader, DemandTextureCachePtr demandTextureCache, ProgramGroupsPtr programGroups )
        : m_options( options )
        , m_materialLoader( std::move( materialLoader ) )
        , m_demandTextureCache( std::move( demandTextureCache ) )
        , m_programGroups( std::move( programGroups ) )
    {
    }
    ~PbrtMaterialResolver() override = default;

    void resolveOneMaterial() override { m_resolveOneMaterial = true; }

    bool resolveMaterialForGeometry( uint_t proxyGeomId, const GeometryInstance& sceneGeom, SceneSyncState& syncState ) override;

    MaterialResolution resolveRequestedProxyMaterials( CUstream stream, const FrameStopwatch& frameTime, SceneSyncState& syncState ) override;

    MaterialResolverStats getStatistics() const override { return m_stats; }

  private:
    MaterialResolution    resolveMaterial( uint_t proxyMaterialId, SceneSyncState& m_sync );
    std::optional<uint_t> findResolvedMaterial( const MaterialGroup& group, SceneSyncState& syncState ) const;

    // Dependencies
    const Options&        m_options;
    MaterialLoaderPtr     m_materialLoader;
    DemandTextureCachePtr m_demandTextureCache;
    ProgramGroupsPtr      m_programGroups;

    bool                            m_resolveOneMaterial{};
    MaterialResolverStats           m_stats{};
    std::map<uint_t, SceneGeometry> m_proxyMaterialGeometries;  // indexed by proxy material id
};

MaterialResolution PbrtMaterialResolver::resolveMaterial( uint_t proxyMaterialId, SceneSyncState& m_sync )
{
    const auto it{ m_proxyMaterialGeometries.find( proxyMaterialId ) };
    if( it == m_proxyMaterialGeometries.end() )
    {
        throw std::runtime_error( "Unknown material id " + std::to_string( proxyMaterialId ) );
    }

    SceneGeometry& geom{ it->second };
    if( geom.materialId != proxyMaterialId )
    {
        throw std::runtime_error( "Mismatched material id; expected " + std::to_string( geom.materialId ) + ", got "
                                  + std::to_string( proxyMaterialId ) );
    }

    // Only triangle meshes support alpha maps currently.
    // TODO: support alpha maps on spheres
    if( geom.instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        // phase 1 alpha map resolution
        if( flagSet( geom.instance.groups.material.flags, MaterialFlags::ALPHA_MAP )
            && !flagSet( geom.instance.groups.material.flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
        {
            const uint_t alphaTextureId{ m_demandTextureCache->createAlphaTextureFromFile( geom.instance.groups.alphaMapFileName ) };
            m_sync.minAlphaTextureId                     = std::min( alphaTextureId, m_sync.minAlphaTextureId );
            m_sync.maxAlphaTextureId                     = std::max( alphaTextureId, m_sync.maxAlphaTextureId );
            geom.instance.groups.material.alphaTextureId = alphaTextureId;
            geom.instance.groups.material.flags |= MaterialFlags::ALPHA_MAP_ALLOCATED;
            const size_t numProxyMaterials{ proxyMaterialId + 1 };  // ids are zero based
            grow( m_sync.partialMaterials, numProxyMaterials );
            grow( m_sync.partialUVs, numProxyMaterials );
            m_sync.partialMaterials[proxyMaterialId].alphaTextureId = geom.instance.groups.material.alphaTextureId;
            m_sync.partialUVs[proxyMaterialId]                      = geom.instance.devUVs;
            m_sync.topLevelInstances[geom.instanceIndex].sbtOffset  = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA;
            if( m_options.verboseProxyMaterialResolution )
            {
                std::cout << "Resolved proxy material id " << proxyMaterialId << " to partial alpha texture id "
                          << geom.instance.groups.material.alphaTextureId << '\n';
            }
            return MaterialResolution::PARTIAL;
        }

        // phase 2 alpha map resolution
        if( flagSet( geom.instance.groups.material.flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
        {
            // not strictly necessary, but indicates this partial material has been resolved completely
            m_sync.partialMaterials[proxyMaterialId].alphaTextureId = 0;
            m_sync.partialUVs[proxyMaterialId]                      = nullptr;
        }

        // diffuse map resolution
        if( flagSet( geom.instance.groups.material.flags, MaterialFlags::DIFFUSE_MAP )
            && !flagSet( geom.instance.groups.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
        {
            const uint_t diffuseTextureId =
                m_demandTextureCache->createDiffuseTextureFromFile( geom.instance.groups.diffuseMapFileName );
            m_sync.minDiffuseTextureId                     = std::min( diffuseTextureId, m_sync.minDiffuseTextureId );
            m_sync.maxDiffuseTextureId                     = std::max( diffuseTextureId, m_sync.maxDiffuseTextureId );
            geom.instance.groups.material.diffuseTextureId = diffuseTextureId;
            geom.instance.groups.material.flags |= MaterialFlags::DIFFUSE_MAP_ALLOCATED;
        }
    }

    geom.instance.instance.sbtOffset  = m_programGroups->getRealizedMaterialSbtOffset( geom.instance );
    geom.instance.instance.instanceId = m_sync.instanceMaterialIds.size();
    const uint_t materialId           = m_sync.realizedMaterials.size();
    m_sync.instanceMaterialIds.push_back( materialId );
    m_sync.realizedMaterials.push_back( geom.instance.groups.material );
    m_sync.realizedNormals.push_back( geom.instance.devNormals );
    m_sync.realizedUVs.push_back( geom.instance.devUVs );
    m_sync.topLevelInstances[geom.instanceIndex] = geom.instance.instance;
    if( m_options.verboseProxyMaterialResolution )
    {
        std::cout << "Resolved proxy material id " << proxyMaterialId << " for instance id " << geom.instance.instance.instanceId
                  << ( flagSet( geom.instance.groups.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) ?
                           " with diffuse map" :
                           "" )
                  << '\n';
    }
    m_materialLoader->remove( proxyMaterialId );
    m_proxyMaterialGeometries.erase( proxyMaterialId );
    return MaterialResolution::FULL;
}

MaterialResolution PbrtMaterialResolver::resolveRequestedProxyMaterials( CUstream stream, const FrameStopwatch& frameTime, SceneSyncState& syncState )
{
    if( m_options.oneShotMaterial && !m_resolveOneMaterial )
    {
        return MaterialResolution::NONE;
    }

    MaterialResolution resolution{ MaterialResolution::NONE };
    const unsigned int MIN_REALIZED{ 512 };
    unsigned int       realizedCount{};
    for( uint_t id : m_materialLoader->requestedMaterialIds() )
    {
        if( frameTime.expired() && realizedCount > MIN_REALIZED )
        {
            break;
        }

        resolution = std::max( resolution, resolveMaterial( id, syncState ) );

        if( m_resolveOneMaterial )
        {
            m_resolveOneMaterial = false;
            break;
        }
    }
    m_materialLoader->clearRequestedMaterialIds();

    switch( resolution )
    {
        case MaterialResolution::NONE:
            break;
        case MaterialResolution::PARTIAL:
            syncState.partialMaterials.copyToDeviceAsync( stream );
            syncState.partialUVs.copyToDeviceAsync( stream );
            ++m_stats.numPartialMaterialsRealized;
            break;
        case MaterialResolution::FULL:
            syncState.partialMaterials.copyToDeviceAsync( stream );
            syncState.partialUVs.copyToDeviceAsync( stream );
            syncState.realizedNormals.copyToDeviceAsync( stream );
            syncState.realizedUVs.copyToDeviceAsync( stream );
            syncState.realizedMaterials.copyToDeviceAsync( stream );
            syncState.instanceMaterialIds.copyToDeviceAsync( stream );
            ++m_stats.numMaterialsRealized;
            break;
    }
    return resolution;
}

std::optional<uint_t> PbrtMaterialResolver::findResolvedMaterial( const MaterialGroup& group, SceneSyncState& syncState ) const
{
    // Check for loaded diffuse map
    if( flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP )
        && !m_demandTextureCache->hasDiffuseTextureForFile( group.diffuseMapFileName ) )
    {
        return {};
    }

    // Check for loaded alpha map
    if( flagSet( group.material.flags, MaterialFlags::ALPHA_MAP )
        && !m_demandTextureCache->hasAlphaTextureForFile( group.alphaMapFileName ) )
    {
        return {};
    }

    // TODO: consider a sorted container for binary search instead of linear search of m_realizedMaterials
    const auto it =
        std::find_if( syncState.realizedMaterials.cbegin(), syncState.realizedMaterials.cend(), [&]( const PhongMaterial& entry ) {
            return group.material.Ka == entry.Ka                 //
                   && group.material.Kd == entry.Kd              //
                   && group.material.Ks == entry.Ks              //
                   && group.material.Kr == entry.Kr              //
                   && group.material.phongExp == entry.phongExp  //
                   && ( group.material.flags & ( MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP ) )
                          == ( entry.flags & ( MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP ) );
        } );
    if( it != syncState.realizedMaterials.cend() )
    {
        return { static_cast<uint_t>( std::distance( syncState.realizedMaterials.cbegin(), it ) ) };
    }

    return {};
}

bool PbrtMaterialResolver::resolveMaterialForGeometry( uint_t proxyGeomId, const GeometryInstance& geomInstance, SceneSyncState& syncState )
{
    SceneGeometry geom{};
    geom.instance = geomInstance;

    // check for shared materials
    if( const std::optional<uint_t> id = findResolvedMaterial( geom.instance.groups, syncState ); id.has_value() )
    {
        // just for completeness's sake, mark the duplicate material's textures as having
        // been loaded, although we won't use the duplicate material after this.
        const auto markAllocated = [&]( MaterialFlags requested, MaterialFlags allocated ) {
            MaterialFlags& flags{ geom.instance.groups.material.flags };
            if( flagSet( flags, requested ) )
            {
                flags |= allocated;
            }
        };
        markAllocated( MaterialFlags::ALPHA_MAP, MaterialFlags::ALPHA_MAP_ALLOCATED );
        markAllocated( MaterialFlags::DIFFUSE_MAP, MaterialFlags::DIFFUSE_MAP_ALLOCATED );

        // reuse already realized material
        //OTK_ASSERT( m_phongModule != nullptr );  // should have already been realized
        const uint_t materialId{ id.value() };
        const uint_t instanceId{ static_cast<uint_t>( syncState.instanceMaterialIds.size() ) };
        geom.materialId                   = materialId;
        geom.instance.instance.sbtOffset  = m_programGroups->getRealizedMaterialSbtOffset( geom.instance );
        geom.instance.instance.instanceId = instanceId;
        geom.instanceIndex                = syncState.topLevelInstances.size();
        syncState.instanceMaterialIds.push_back( materialId );
        syncState.topLevelInstances.push_back( geom.instance.instance );
        syncState.realizedNormals.push_back( geom.instance.devNormals );
        syncState.realizedUVs.push_back( geom.instance.devUVs );
        m_proxyMaterialGeometries[materialId] = geom;
        ++m_stats.numMaterialsReused;

        if( m_options.verboseProxyGeometryResolution )
        {
            std::cout << "Resolved proxy geometry id " << proxyGeomId << " to geometry instance id "
                      << geom.instanceIndex << " with existing material id " << materialId << '\n';
        }

        return true;
    }

    const uint_t materialId{ m_materialLoader->add() };
    const uint_t instanceId{ materialId };  // use the proxy material id as the instance id
    geom.materialId                   = materialId;
    geom.instance.instance.instanceId = instanceId;
    geom.instanceIndex                = syncState.topLevelInstances.size();
    syncState.topLevelInstances.push_back( geom.instance.instance );
    m_proxyMaterialGeometries[materialId] = geom;
    if( m_options.verboseProxyGeometryResolution )
    {
        std::cout << "Resolved proxy geometry id " << proxyGeomId << " to geometry instance id " << geom.instanceIndex
                  << " with proxy material id " << geom.materialId << '\n';
    }
    ++m_stats.numProxyMaterialsCreated;
    return false;
}

}  // namespace

MaterialResolverPtr createMaterialResolver( const Options&        options,
                                            MaterialLoaderPtr     materialLoader,
                                            DemandTextureCachePtr demandTextureCache,
                                            ProgramGroupsPtr      programGroups )
{
    return std::make_shared<PbrtMaterialResolver>( options, std::move( materialLoader ),
                                                   std::move( demandTextureCache ), std::move( programGroups ) );
}

}  // namespace demandPbrtScene
