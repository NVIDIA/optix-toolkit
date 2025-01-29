// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MaterialResolver.h"

#include "DemandPbrtScene/Conversions.h"
#include "DemandPbrtScene/DemandTextureCache.h"
#include "DemandPbrtScene/FrameStopwatch.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/ProgramGroups.h"
#include "DemandPbrtScene/SceneGeometry.h"
#include "DemandPbrtScene/SceneProxy.h"
#include "DemandPbrtScene/SceneSyncState.h"

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <iterator>
#include <map>
#include <memory>
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

using SceneGeometryPtr = std::shared_ptr<SceneGeometry>;

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

    bool resolveMaterialForGeometry( uint_t proxyGeomId, const GeometryInstance& geomInstance, SceneSyncState& syncState ) override;

    MaterialResolution resolveRequestedProxyMaterials( CUstream stream, const FrameStopwatch& frameTime, SceneSyncState& syncState ) override;

    MaterialResolverStats getStatistics() const override { return m_stats; }

  private:
    MaterialResolution    resolveMaterialGroup( std::vector<uint_t>&    requestedMaterials,
                                                SceneSyncState&         sync,
                                                const SceneGeometryPtr& geom,
                                                size_t                  index,
                                                std::vector<uint_t>&    resolvedMaterialIds );
    MaterialResolution    resolveMaterial( std::vector<uint_t>& requestedMaterials, SceneSyncState& sync );
    std::optional<uint_t> findResolvedMaterial( const MaterialGroup& group, const SceneSyncState& syncState ) const;
    bool                  resolveGeometryToExistingMaterial( uint_t                  proxyGeomId,
                                                             uint_t                  materialIndex,
                                                             const SceneGeometryPtr& geom,
                                                             MaterialGroup&          group,
                                                             SceneSyncState&         syncState );
    void resolveGeometryToProxyMaterial( uint_t proxyGeomId, const SceneGeometryPtr& geom, const MaterialGroup& group, SceneSyncState& syncState );

    // Dependencies
    const Options&        m_options;
    MaterialLoaderPtr     m_materialLoader;
    DemandTextureCachePtr m_demandTextureCache;
    ProgramGroupsPtr      m_programGroups;

    bool                               m_resolveOneMaterial{};
    MaterialResolverStats              m_stats{};
    std::map<uint_t, SceneGeometryPtr> m_proxyMaterialGeometries;  // indexed by proxy material id
};

std::string toString( const std::vector<uint_t>& ids )
{
    std::string result{ "[" };
    bool        first{ true };
    for( uint_t id : ids )
    {
        if( !first )
            result += ", ";
        result += std::to_string( id );
        first = false;
    }
    result += "]";
    return result;
}

MaterialResolution PbrtMaterialResolver::resolveMaterialGroup( std::vector<uint_t>&    requestedMaterials,
                                                               SceneSyncState&         sync,
                                                               const SceneGeometryPtr& geom,
                                                               size_t                  index,
                                                               std::vector<uint_t>&    resolvedMaterialIds )
{
    MaterialGroup& group{ geom->instance.groups[index] };
    const uint_t   groupMaterialId{ geom->materialIds[index] };

    if( auto it = std::find( requestedMaterials.begin(), requestedMaterials.end(), groupMaterialId );
        it != requestedMaterials.end() )
    {
        requestedMaterials.erase( it );
    }

    // Only triangle meshes support alpha maps currently.
    // TODO: support alpha maps on spheres
    if( geom->instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        // phase 1 alpha map resolution
        if( flagSet( group.material.flags, MaterialFlags::ALPHA_MAP )
            && !flagSet( group.material.flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
        {
            const uint_t alphaTextureId{ m_demandTextureCache->createAlphaTextureFromFile( group.alphaMapFileName ) };
            sync.minAlphaTextureId        = std::min( alphaTextureId, sync.minAlphaTextureId );
            sync.maxAlphaTextureId        = std::max( alphaTextureId, sync.maxAlphaTextureId );
            group.material.alphaTextureId = alphaTextureId;
            group.material.flags |= MaterialFlags::ALPHA_MAP_ALLOCATED;
            const size_t numProxyMaterials{ groupMaterialId + 1 };  // ids are zero based
            grow( sync.partialMaterials, numProxyMaterials );
            grow( sync.partialUVs, numProxyMaterials );
            sync.partialMaterials[groupMaterialId].alphaTextureId = group.material.alphaTextureId;
            sync.partialUVs[groupMaterialId]                      = geom->instance.devUVs;
            geom->instance.instance.sbtOffset                     = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA;
            if( m_options.verboseProxyMaterialResolution )
            {
                std::cout << "Resolved proxy material id " << groupMaterialId << " for instance id "
                          << geom->instance.instance.instanceId << ", material group " << index
                          << " to partial alpha texture id " << group.material.alphaTextureId << '\n';
            }
            ++m_stats.numPartialMaterialsRealized;
            return MaterialResolution::PARTIAL;
        }

        // phase 2 alpha map resolution
        if( flagSet( group.material.flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
        {
            // not strictly necessary, but indicates this partial material has been resolved completely
            sync.partialMaterials[groupMaterialId].alphaTextureId = 0;
            sync.partialUVs[groupMaterialId]                      = nullptr;
        }

        // diffuse map resolution
        if( flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP )
            && !flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
        {
            const uint_t diffuseTextureId = m_demandTextureCache->createDiffuseTextureFromFile( group.diffuseMapFileName );
            sync.minDiffuseTextureId        = std::min( diffuseTextureId, sync.minDiffuseTextureId );
            sync.maxDiffuseTextureId        = std::max( diffuseTextureId, sync.maxDiffuseTextureId );
            group.material.diffuseTextureId = diffuseTextureId;
            group.material.flags |= MaterialFlags::DIFFUSE_MAP_ALLOCATED;
        }
    }

    geom->instance.instance.sbtOffset = m_programGroups->getRealizedMaterialSbtOffset( geom->instance );
    const uint_t materialId{ containerSize( sync.realizedMaterials ) };
    sync.realizedMaterials.push_back( group.material );
    const uint_t materialIndex{ geom->instance.instance.instanceId };
    OTK_ASSERT( materialIndex < sync.materialIndices.size() );
    sync.primitiveMaterials[sync.materialIndices[materialIndex].primitiveMaterialBegin + index].materialId = materialId;
    if( m_options.verboseProxyMaterialResolution )
    {
        std::cout << "Resolved proxy material id " << groupMaterialId << " for instance id "
                  << geom->instance.instance.instanceId << ", material group " << index;
        if (flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
        {
            std::cout << " with diffuse texture id " << group.material.diffuseTextureId;
        }
        std::cout << '\n';
    }
    m_materialLoader->remove( groupMaterialId );
    resolvedMaterialIds.push_back( groupMaterialId );
    ++m_stats.numMaterialsRealized;
    return MaterialResolution::FULL;
}

MaterialResolution PbrtMaterialResolver::resolveMaterial( std::vector<uint_t>& requestedMaterials, SceneSyncState& sync )
{
    const uint_t requestedMaterialId{ requestedMaterials.front() };
    const auto   proxyMatGeomIt{ m_proxyMaterialGeometries.find( requestedMaterialId ) };
    if( proxyMatGeomIt == m_proxyMaterialGeometries.end() )
    {
        throw std::runtime_error( "Unknown material id " + std::to_string( requestedMaterialId ) );
    }

    SceneGeometryPtr& geom{ proxyMatGeomIt->second };
    if( const auto pos{ std::find( geom->materialIds.begin(), geom->materialIds.end(), requestedMaterialId ) };
        pos == geom->materialIds.end() )
    {
        throw std::runtime_error( "Mismatched material id; expected one of " + toString( geom->materialIds ) + ", got "
                                  + std::to_string( requestedMaterialId ) );
    }

    if( geom->materialIds.size() != geom->instance.groups.size() )
    {
        throw std::runtime_error( "Mismatched material id count (" + std::to_string( geom->materialIds.size() )
                                  + ") for material group count (" + std::to_string( geom->instance.groups.size() )
                                  + ")" );
    }

    MaterialResolution  result{ MaterialResolution::NONE };
    std::vector<uint_t> resolvedMaterialIds;
    resolvedMaterialIds.reserve( geom->instance.groups.size() );
    for( size_t i = 0; i < geom->instance.groups.size(); ++i )
    {
        result = std::max( result, resolveMaterialGroup( requestedMaterials, sync, geom, i, resolvedMaterialIds ) );
    }
    const uint_t index{ geom->instance.instance.instanceId };
    grow( sync.realizedNormals, index + 1 );
    grow( sync.realizedUVs, index + 1 );
    sync.realizedNormals[index]                 = geom->instance.devNormals;
    sync.realizedUVs[index]                     = geom->instance.devUVs;
    sync.topLevelInstances[geom->instanceIndex] = geom->instance.instance;
    for( uint_t materialId : resolvedMaterialIds )
    {
        if( auto it = m_proxyMaterialGeometries.find( materialId ); it != m_proxyMaterialGeometries.end() )
        {
            m_proxyMaterialGeometries.erase( it );
        }
        else
        {
            throw std::runtime_error( "Resolved material id " + std::to_string( materialId )
                                      + " that was missing from proxy material geometries map" );
        }
    }

    return result;
}

MaterialResolution PbrtMaterialResolver::resolveRequestedProxyMaterials( CUstream stream, const FrameStopwatch& frameTime, SceneSyncState& syncState )
{
    if( m_options.oneShotMaterial && !m_resolveOneMaterial )
    {
        return MaterialResolution::NONE;
    }

    MaterialResolution  resolution{ MaterialResolution::NONE };
    const unsigned int  MIN_REALIZED{ 512 };
    unsigned int        realizedCount{};
    std::vector<uint_t> requestedMaterials{ m_materialLoader->requestedMaterialIds() };
    while( !requestedMaterials.empty() )
    {
        if( frameTime.expired() && realizedCount > MIN_REALIZED )
        {
            break;
        }

        resolution = std::max( resolution, resolveMaterial( requestedMaterials, syncState ) );

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
            break;
        case MaterialResolution::FULL:
            syncState.partialMaterials.copyToDeviceAsync( stream );
            syncState.partialUVs.copyToDeviceAsync( stream );
            syncState.realizedNormals.copyToDeviceAsync( stream );
            syncState.realizedUVs.copyToDeviceAsync( stream );
            syncState.realizedMaterials.copyToDeviceAsync( stream );
            syncState.primitiveMaterials.copyToDeviceAsync( stream );
            break;
    }
    return resolution;
}

std::optional<uint_t> PbrtMaterialResolver::findResolvedMaterial( const MaterialGroup& group, const SceneSyncState& syncState ) const
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
        return { toUInt( std::distance( syncState.realizedMaterials.cbegin(), it ) ) };
    }

    return {};
}

bool PbrtMaterialResolver::resolveGeometryToExistingMaterial( uint_t                  proxyGeomId,
                                                              uint_t                  materialIndex,
                                                              const SceneGeometryPtr& geom,
                                                              MaterialGroup&          group,
                                                              SceneSyncState&         syncState )
{
    // just for completeness's sake, mark the duplicate material's textures as having
    // been loaded, although we won't use the duplicate material after this.
    const auto markAllocated = [&]( MaterialFlags requested, MaterialFlags allocated ) {
        MaterialFlags& flags{ group.material.flags };
        if( flagSet( flags, requested ) )
        {
            flags |= allocated;
        }
    };
    markAllocated( MaterialFlags::ALPHA_MAP, MaterialFlags::ALPHA_MAP_ALLOCATED );
    markAllocated( MaterialFlags::DIFFUSE_MAP, MaterialFlags::DIFFUSE_MAP_ALLOCATED );

    // reuse already realized material
    geom->materialIds.push_back( materialIndex );
    OTK_ASSERT( materialIndex < syncState.materialIndices.size() );
    const uint_t index{ geom->instance.instance.instanceId };
    grow( syncState.realizedNormals, index + 1 );
    grow( syncState.realizedUVs, index + 1 );
    OTK_ASSERT( index < syncState.realizedNormals.size() );
    OTK_ASSERT( index < syncState.realizedUVs.size() );
    syncState.realizedNormals[index] = geom->instance.devNormals;
    syncState.realizedUVs[index] = geom->instance.devUVs;
    syncState.primitiveMaterials.push_back( PrimitiveMaterialRange{ group.primitiveIndexEnd, materialIndex } );
    m_proxyMaterialGeometries[materialIndex] = geom;
    ++m_stats.numMaterialsReused;

    if( m_options.verboseProxyGeometryResolution )
    {
        std::cout << "Resolved proxy geometry id " << proxyGeomId << " to geometry instance id " << geom->instanceIndex
                  << " with existing material id " << materialIndex << '\n';
    }

    return true;
}

void PbrtMaterialResolver::resolveGeometryToProxyMaterial( uint_t                  proxyGeomId,
                                                           const SceneGeometryPtr& geom,
                                                           const MaterialGroup&    group,
                                                           SceneSyncState&         syncState )
{
    const uint_t materialId{ m_materialLoader->add() };
    geom->materialIds.push_back( materialId );
    syncState.primitiveMaterials.push_back( PrimitiveMaterialRange{ group.primitiveIndexEnd, materialId } );
    m_proxyMaterialGeometries[materialId] = geom;
    if( m_options.verboseProxyGeometryResolution )
    {
        std::cout << "Resolved proxy geometry id " << proxyGeomId << " to geometry instance index "
                  << geom->instanceIndex << " with proxy material id " << geom->materialIds[0] << '\n';
    }
    ++m_stats.numProxyMaterialsCreated;
}

bool PbrtMaterialResolver::resolveMaterialForGeometry( uint_t proxyGeomId, const GeometryInstance& geomInstance, SceneSyncState& syncState )
{
    SceneGeometryPtr geom{ std::make_shared<SceneGeometry>() };
    geom->instance = geomInstance;

    // check for shared materials
    bool                               updateNeeded{};
    const uint_t                       primitiveMaterialBegin{ containerSize( syncState.primitiveMaterials ) };
    std::vector<MaterialGroup>&        groups{ geom->instance.groups };
    std::vector<std::optional<uint_t>> resolvedMaterialIds;
    resolvedMaterialIds.resize( groups.size() );
    std::transform( groups.cbegin(), groups.cend(), resolvedMaterialIds.begin(),
                    [&]( const MaterialGroup& group ) { return findResolvedMaterial( group, syncState ); } );
    if( std::any_of( resolvedMaterialIds.cbegin(), resolvedMaterialIds.cend(),
                     []( const std::optional<uint_t>& id ) { return !id.has_value(); } ) )
    {
        geom->instanceIndex                = containerSize( syncState.topLevelInstances );
        geom->instance.instance.instanceId = containerSize( syncState.materialIndices );
        for( MaterialGroup& group : groups )
        {
            resolveGeometryToProxyMaterial( proxyGeomId, geom, group, syncState );
        }
    }
    else
    {
        geom->instanceIndex                = containerSize( syncState.topLevelInstances );
        geom->instance.instance.instanceId = containerSize( syncState.materialIndices );
        geom->instance.instance.sbtOffset  = m_programGroups->getRealizedMaterialSbtOffset( geom->instance );
        auto id                            = resolvedMaterialIds.cbegin();
        for( MaterialGroup& group : geom->instance.groups )
        {
            if( resolveGeometryToExistingMaterial( proxyGeomId, id->value(), geom, group, syncState ) )
            {
                updateNeeded = true;
            }
            ++id;
        }
    }
    syncState.topLevelInstances.push_back( geom->instance.instance );
    const uint_t numGroups{ containerSize( geom->instance.groups ) };
    syncState.materialIndices.push_back( MaterialIndex{ numGroups, primitiveMaterialBegin } );

    return updateNeeded;
}

}  // namespace

MaterialResolverPtr createMaterialResolver( const Options&        options,
                                            MaterialLoaderPtr     materialLoader,
                                            DemandTextureCachePtr demandTextureCache,
                                            ProgramGroupsPtr      programGroups )
{
    return std::make_shared<PbrtMaterialResolver>( options,                          //
                                                   std::move( materialLoader ),      //
                                                   std::move( demandTextureCache ),  //
                                                   std::move( programGroups ) );
}

}  // namespace demandPbrtScene
