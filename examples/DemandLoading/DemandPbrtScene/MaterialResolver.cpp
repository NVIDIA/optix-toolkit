// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MaterialResolver.h"

#include "DemandPbrtScene/Config.h"
#include "DemandPbrtScene/Conversions.h"
#include "DemandPbrtScene/DemandTextureCache.h"
#include "DemandPbrtScene/FrameStopwatch.h"
#ifdef OTK_USE_MDL
#include "DemandPbrtScene/MdlShaderCache.h"
#endif
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/ProgramGroups.h"
#include "DemandPbrtScene/SceneGeometry.h"
#include "DemandPbrtScene/SceneProxy.h"
#include "DemandPbrtScene/SceneSyncState.h"

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

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

#ifdef OTK_USE_MDL
struct PendingMdlMaterial
{
    PendingMdlMaterial() = default;
    PendingMdlMaterial( const GeometryInstance& instance_, uint_t materialId_, uint_t shaderKeyId_ )
        : instance( instance_ )
        , materialId( materialId_ )
        , shaderKeyId( shaderKeyId_ )
    {
    }

    GeometryInstance instance;
    uint_t           materialId{};
    uint_t           shaderKeyId{};
};
#endif

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

    MaterialResolverStats getStatistics() const override
    {
        MaterialResolverStats stats{ m_stats };
#ifdef OTK_USE_MDL
        stats.mdlShaders = m_mdlShaderCompileCache.getStatistics();
#endif
        return stats;
    }

  private:
    MaterialResolution resolveMaterialGroup( std::vector<uint_t>&    requestedMaterials,
                                             SceneSyncState&         sync,
                                             const SceneGeometryPtr& geom,
                                             size_t                  index,
                                             std::vector<uint_t>&    resolvedMaterialIds );
    MaterialResolution resolveMaterial( std::vector<uint_t>& requestedMaterials, SceneSyncState& sync );
    MaterialState resolveMaterialState( SceneSyncState& sync, GeometryInstance& instance, const MaterialGroup& group, uint_t materialId );
#ifdef OTK_USE_MDL
    MaterialResolution resolvePendingMdlMaterial( SceneSyncState& sync );
    MaterialState resolveMdlMaterialState( SceneSyncState&      sync,
                                           GeometryInstance&    instance,
                                           const MaterialGroup& group,
                                           uint_t               materialId );
    void          queuePendingMdlMaterial( const MdlMaterialInstanceKey& key, const PendingMdlMaterial& material );
#endif
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

    bool                  m_resolveOneMaterial{};
    MaterialResolverStats m_stats{};
#ifdef OTK_USE_MDL
    MdlShaderCompileCache                                             m_mdlShaderCompileCache;
    std::map<MdlMaterialInstanceKey, std::vector<PendingMdlMaterial>> m_pendingMdlMaterials;
#endif
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

MaterialState localFallbackState( uint_t materialId )
{
    return makeMaterialState( materialId, MaterialBackend::LOCAL_FALLBACK );
}

#ifdef OTK_USE_MDL
MaterialState unsupportedFallbackState( uint_t materialId )
{
    return makeMaterialState( materialId, MaterialBackend::LOCAL_FALLBACK, 0U, MaterialFallbackReason::UNSUPPORTED );
}

MaterialState mdlReadyState( uint_t materialId, uint_t shaderKeyId )
{
    return makeMaterialState( materialId, MaterialBackend::MDL_READY, shaderKeyId );
}

bool supportsGeneratedMdlMaterial( const std::string& type )
{
    return type == "matte" || type == "plastic" || type == "uber" || type == "mirror" || type == "glass"
           || type == "metal" || type == "substrate" || type == "translucent" || type == "mix";
}

bool supportsGeneratedMdlNamedMaterialType( const std::string& type )
{
    return type == "matte" || type == "plastic" || type == "uber" || type == "mirror" || type == "glass"
           || type == "metal" || type == "substrate" || type == "translucent";
}

std::string generatedMdlNamedMaterialType( const otk::pbrt::PbrtNamedMaterial& material )
{
    if( !material.type.empty() )
    {
        return material.type;
    }
    return material.params.FindOneString( "type", std::string{} );
}

bool hasGeneratedMdlMaterialTextureReference( const ::pbrt::ParamSet& params )
{
    static const char* const textureParams[] = {
        "Kd", "Kr",      "Ks",      "Kt",        "alpha",       "amount", "bumpmap",  "eta",        "index",
        "k",  "opacity", "reflect", "roughness", "shadowalpha", "sigma",  "transmit", "uroughness", "vroughness",
    };

    for( const char* const param : textureParams )
    {
        if( !params.FindTexture( param ).empty() )
        {
            return true;
        }
    }
    return false;
}

bool hasGeneratedMdlConstantFloatTexture( const otk::pbrt::PbrtMaterial& material, const std::string& textureName )
{
    for( otk::pbrt::PbrtTextureMap::const_iterator it = material.graph.textures.begin(); it != material.graph.textures.end(); ++it )
    {
        if( it->second.name != textureName || it->second.valueType != "float" || it->second.type != "constant" )
        {
            continue;
        }
        int          count{};
        const float* values = it->second.params.FindFloat( "value", &count );
        return count > 0 && values != nullptr;
    }
    return false;
}

bool supportsGeneratedMdlNamedMaterialReference( const otk::pbrt::PbrtMaterial& material, const std::string& paramName )
{
    const std::string materialName{ material.params.FindOneString( paramName, std::string{} ) };
    if( materialName.empty() )
    {
        return false;
    }

    const otk::pbrt::PbrtNamedMaterialMap::const_iterator namedMaterial = material.graph.namedMaterials.find( materialName );
    if( namedMaterial == material.graph.namedMaterials.end() )
    {
        return false;
    }

    return supportsGeneratedMdlNamedMaterialType( generatedMdlNamedMaterialType( namedMaterial->second ) )
           && !hasGeneratedMdlMaterialTextureReference( namedMaterial->second.params );
}

bool supportsGeneratedMdlNamedMaterialReferences( const otk::pbrt::PbrtMaterial& material )
{
    if( material.graph.namedMaterials.empty() )
    {
        return true;
    }
    if( material.type != "mix" )
    {
        return false;
    }
    return supportsGeneratedMdlNamedMaterialReference( material, "namedmaterial1" )
           && supportsGeneratedMdlNamedMaterialReference( material, "namedmaterial2" );
}

bool hasGeneratedMdlDemandTexture( MaterialFlags flags, MaterialFlags mapFlag, MaterialFlags allocatedFlag, const std::string& fileName )
{
    return flagSet( flags, mapFlag | allocatedFlag ) && !fileName.empty();
}

bool supportsGeneratedMdlTextureReferences( const otk::pbrt::PbrtMaterial& material, const MaterialGroup& group )
{
    const MaterialFlags flags{ group.material.flags };
    const bool hasDiffuseMap{ hasGeneratedMdlDemandTexture( flags, MaterialFlags::DIFFUSE_MAP,
                                                            MaterialFlags::DIFFUSE_MAP_ALLOCATED, group.diffuseMapFileName ) };
    const bool hasAlphaMap{ hasGeneratedMdlDemandTexture( flags, MaterialFlags::ALPHA_MAP,
                                                          MaterialFlags::ALPHA_MAP_ALLOCATED, group.alphaMapFileName ) };

    static const char* const textureParams[] = {
        "Kd", "Kr",      "Ks",      "Kt",        "alpha",       "amount", "bumpmap",  "eta",        "index",
        "k",  "opacity", "reflect", "roughness", "shadowalpha", "sigma",  "transmit", "uroughness", "vroughness",
    };

    for( const char* const param : textureParams )
    {
        const std::string textureName{ material.params.FindTexture( param ) };
        if( textureName.empty() )
        {
            continue;
        }
        const std::string paramName{ param };
        if( paramName == "Kd" )
        {
            if( !hasDiffuseMap )
            {
                return false;
            }
            continue;
        }
        if( paramName == "alpha" || paramName == "shadowalpha" || paramName == "opacity" )
        {
            if( !hasAlphaMap )
            {
                return false;
            }
            continue;
        }
        if( paramName == "amount" )
        {
            if( !hasGeneratedMdlConstantFloatTexture( material, textureName ) )
            {
                return false;
            }
            continue;
        }
        return false;
    }

    const MaterialFlags supportedFlags{ MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED
                                        | MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED };
    if( ( flags & ~supportedFlags ) != MaterialFlags::NONE )
    {
        return false;
    }
    if( flagSet( flags, MaterialFlags::DIFFUSE_MAP ) && !hasDiffuseMap )
    {
        return false;
    }
    if( flagSet( flags, MaterialFlags::ALPHA_MAP ) && !hasAlphaMap )
    {
        return false;
    }
    return true;
}

bool hasGeneratedMdlUnsupportedTextureReference( const otk::pbrt::PbrtMaterial& material, const MaterialGroup& group )
{
    if( !supportsGeneratedMdlTextureReferences( material, group ) )
    {
        return true;
    }
    for( const auto& texture : material.graph.textures )
    {
        if( texture.second.name.empty() )
        {
            return true;
        }
    }
    return false;
}

bool usesGeneratedMdlMaterial( const Options& options, const GeometryInstance& instance, const MaterialGroup& group )
{
    return options.useMdlMaterials && instance.primitive == GeometryPrimitive::TRIANGLE
           && instance.groups.size() == 1 && group.pbrtMaterial && supportsGeneratedMdlMaterial( group.pbrtMaterial->type )
           && group.pbrtMaterial->graph.fallbackReasons.empty() && supportsGeneratedMdlNamedMaterialReferences( *group.pbrtMaterial )
           && !hasGeneratedMdlUnsupportedTextureReference( *group.pbrtMaterial, group );
}

bool usesGeneratedMdlUnsupportedFallback( const Options& options, const GeometryInstance& instance, const MaterialGroup& group )
{
    return options.useMdlMaterials && instance.primitive == GeometryPrimitive::TRIANGLE
           && instance.groups.size() == 1 && group.pbrtMaterial;
}

MdlMaterialInstanceKey makeMaterialGroupMdlMaterialInstanceKey( const MaterialGroup& group )
{
    if( group.pbrtMaterial )
    {
        return makeMdlMaterialInstanceKey( *group.pbrtMaterial );
    }
    const MdlShaderKey sourceKey{ "pbrt-mdl-v1|missing-pbrt-material" };
    return MdlMaterialInstanceKey{ sourceKey, "pbrt-mdl-instance-v1|missing-pbrt-material" };
}
#endif

void setMaterialState( SceneSyncState& sync, uint_t materialId, MaterialState state )
{
    grow( sync.materialStates, materialId + 1 );
    sync.materialStates[materialId] = state;
}

bool isReadyLocalFallback( const SceneSyncState& sync, uint_t materialId )
{
    if( materialId >= sync.materialStates.size() )
    {
        return false;
    }

    const MaterialState& state{ sync.materialStates[materialId] };
    return state.materialId == materialId && state.backend == MaterialBackend::LOCAL_FALLBACK;
}

#ifdef OTK_USE_MDL
void PbrtMaterialResolver::queuePendingMdlMaterial( const MdlMaterialInstanceKey& key, const PendingMdlMaterial& material )
{
    std::vector<PendingMdlMaterial>& materials{ m_pendingMdlMaterials[key] };
    const auto exists = std::find_if( materials.begin(), materials.end(), [&]( const PendingMdlMaterial& pending ) {
        return pending.materialId == material.materialId;
    } );
    if( exists == materials.end() )
    {
        materials.push_back( material );
    }
}

MaterialResolution PbrtMaterialResolver::resolvePendingMdlMaterial( SceneSyncState& sync )
{
    if( m_pendingMdlMaterials.empty() )
    {
        return MaterialResolution::NONE;
    }

    const MdlMaterialInstanceKey          materialKey{ m_pendingMdlMaterials.begin()->first };
    const std::vector<PendingMdlMaterial> pendingMaterials{ m_pendingMdlMaterials.begin()->second };
    const PendingMdlMaterial&             firstMaterial{ pendingMaterials.front() };
    try
    {
        m_mdlShaderCompileCache.markCompiling( materialKey );
        const MdlMaterialShader shader{ m_programGroups->realizeMdlMaterialShader( firstMaterial.instance, firstMaterial.shaderKeyId ) };
        grow( sync.mdlMaterialShaders, firstMaterial.shaderKeyId + 1 );
        sync.mdlMaterialShaders[firstMaterial.shaderKeyId] = shader;
        for( const PendingMdlMaterial& material : pendingMaterials )
        {
            OTK_ASSERT( material.shaderKeyId == firstMaterial.shaderKeyId );
            setMaterialState( sync, material.materialId, mdlReadyState( material.materialId, material.shaderKeyId ) );
        }
        m_mdlShaderCompileCache.markReady( materialKey );
    }
    catch( const MdlMaterialBuildPending& )
    {
        return MaterialResolution::NONE;
    }
    catch( const std::exception& e )
    {
        m_mdlShaderCompileCache.markFailed( materialKey, e.what() );
        for( const PendingMdlMaterial& material : pendingMaterials )
        {
            setMaterialState( sync, material.materialId,
                              makeMaterialState( material.materialId, MaterialBackend::MDL_FAILED, material.shaderKeyId ) );
            ++m_stats.numMdlFallbackShaders;
        }
    }
    catch( ... )
    {
        m_mdlShaderCompileCache.markFailed( materialKey, "Unknown MDL shader compile failure" );
        for( const PendingMdlMaterial& material : pendingMaterials )
        {
            setMaterialState( sync, material.materialId,
                              makeMaterialState( material.materialId, MaterialBackend::MDL_FAILED, material.shaderKeyId ) );
            ++m_stats.numMdlFallbackShaders;
        }
    }

    m_pendingMdlMaterials.erase( materialKey );
    return MaterialResolution::FULL;
}

MaterialState PbrtMaterialResolver::resolveMdlMaterialState( SceneSyncState&      sync,
                                                             GeometryInstance&    instance,
                                                             const MaterialGroup& group,
                                                             uint_t               materialId )
{
    const MdlMaterialInstanceKey  materialKey{ makeMaterialGroupMdlMaterialInstanceKey( group ) };
    const MdlShaderCompileRecord& record{ m_mdlShaderCompileCache.getRecord( materialKey ) };
    const uint_t                  shaderKeyId{ record.shaderKeyId };
    const MdlShaderCompileState   state{ record.state };
    const auto                    bindMdlProgram = [&]() {
        instance.instance.sbtOffset = m_programGroups->getMdlMaterialSbtOffset( instance );
    };

    const auto fallbackState = [&]( MaterialBackend backend ) {
        ++m_stats.numMdlFallbackShaders;
        return makeMaterialState( materialId, backend, shaderKeyId );
    };
    const auto queuePending = [&]() {
        queuePendingMdlMaterial( materialKey, PendingMdlMaterial{ instance, materialId, shaderKeyId } );
    };
    const auto bindMdlShader = [&]() {
        const MdlMaterialShader shader{ m_programGroups->realizeMdlMaterialShader( instance, shaderKeyId ) };
        grow( sync.mdlMaterialShaders, shaderKeyId + 1 );
        sync.mdlMaterialShaders[shaderKeyId] = shader;
    };

    bindMdlProgram();
    switch( state )
    {
        case MdlShaderCompileState::READY:
            return mdlReadyState( materialId, shaderKeyId );
        case MdlShaderCompileState::QUEUED:
            if( !m_options.mdlSynchronousCompilation )
            {
                queuePending();
                return fallbackState( MaterialBackend::MDL_PENDING );
            }
            break;
        case MdlShaderCompileState::COMPILING:
            return fallbackState( MaterialBackend::MDL_PENDING );
        case MdlShaderCompileState::FAILED:
            return fallbackState( MaterialBackend::MDL_FAILED );
        case MdlShaderCompileState::MISSING:
            break;
    }

    if( m_mdlShaderCompileCache.requestCompile( materialKey ) )
    {
        ++m_stats.numGeneratedMdlMaterialCompileRequests;
        if( !m_options.mdlSynchronousCompilation )
        {
            try
            {
                bindMdlShader();
                m_mdlShaderCompileCache.markReady( materialKey );
                return mdlReadyState( materialId, shaderKeyId );
            }
            catch( const MdlMaterialBuildPending& )
            {
                queuePending();
                return fallbackState( MaterialBackend::MDL_PENDING );
            }
            catch( const std::exception& e )
            {
                m_mdlShaderCompileCache.markFailed( materialKey, e.what() );
            }
            catch( ... )
            {
                m_mdlShaderCompileCache.markFailed( materialKey, "Unknown MDL shader compile failure" );
            }
            return fallbackState( MaterialBackend::MDL_FAILED );
        }
        m_mdlShaderCompileCache.markCompiling( materialKey );
    }

    try
    {
        bindMdlShader();
        m_mdlShaderCompileCache.markReady( materialKey );
        return mdlReadyState( materialId, shaderKeyId );
    }
    catch( const std::exception& e )
    {
        m_mdlShaderCompileCache.markFailed( materialKey, e.what() );
    }
    catch( ... )
    {
        m_mdlShaderCompileCache.markFailed( materialKey, "Unknown MDL shader compile failure" );
    }

    return fallbackState( MaterialBackend::MDL_FAILED );
}
#endif

MaterialState PbrtMaterialResolver::resolveMaterialState( SceneSyncState& sync, GeometryInstance& instance, const MaterialGroup& group, uint_t materialId )
{
#ifdef OTK_USE_MDL
    if( usesGeneratedMdlMaterial( m_options, instance, group ) )
    {
        return resolveMdlMaterialState( sync, instance, group, materialId );
    }
    if( usesGeneratedMdlUnsupportedFallback( m_options, instance, group ) )
    {
        instance.instance.sbtOffset = m_programGroups->getRealizedMaterialSbtOffset( instance );
        ++m_stats.numMdlFallbackShaders;
        return unsupportedFallbackState( materialId );
    }
#endif

    instance.instance.sbtOffset = m_programGroups->getRealizedMaterialSbtOffset( instance );
    return localFallbackState( materialId );
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
            setMaterialState( sync, groupMaterialId, localFallbackState( groupMaterialId ) );
            geom->instance.instance.sbtOffset = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA;
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

    const uint_t materialId{ groupMaterialId };
    grow( sync.realizedMaterials, materialId + 1 );
    sync.realizedMaterials[materialId] = group.material;
    setMaterialState( sync, materialId, resolveMaterialState( sync, geom->instance, group, materialId ) );
    const uint_t materialIndex{ geom->instance.instance.instanceId };
    OTK_ASSERT( materialIndex < sync.materialIndices.size() );
    sync.primitiveMaterials[sync.materialIndices[materialIndex].primitiveMaterialBegin + index].materialId = materialId;
    if( m_options.verboseProxyMaterialResolution )
    {
        std::cout << "Resolved proxy material id " << groupMaterialId << " for instance id "
                  << geom->instance.instance.instanceId << ", material group " << index;
        if( flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
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
    MaterialResolution resolution{ MaterialResolution::NONE };
#ifdef OTK_USE_MDL
    const bool hadPendingMdlMaterial{ !m_pendingMdlMaterials.empty() };
    resolution = resolvePendingMdlMaterial( syncState );

    if( hadPendingMdlMaterial && resolution == MaterialResolution::NONE )
    {
        return MaterialResolution::NONE;
    }
#endif

    if( resolution == MaterialResolution::NONE && m_options.oneShotMaterial && !m_resolveOneMaterial )
    {
        return MaterialResolution::NONE;
    }

    if( resolution == MaterialResolution::NONE )
    {
        const unsigned int  MIN_REALIZED{ 512 };
        unsigned int        realizedCount{};
        std::vector<uint_t> requestedMaterials{ m_materialLoader->requestedMaterialIds() };
        m_stats.numRequestedMaterialPages += static_cast<unsigned int>( requestedMaterials.size() );
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
    }

    switch( resolution )
    {
        case MaterialResolution::NONE:
            break;
        case MaterialResolution::PARTIAL:
            syncState.materialStates.copyToDeviceAsync( stream );
#ifdef OTK_USE_MDL
            syncState.mdlMaterialShaders.copyToDeviceAsync( stream );
#endif
            syncState.partialMaterials.copyToDeviceAsync( stream );
            syncState.partialUVs.copyToDeviceAsync( stream );
            break;
        case MaterialResolution::FULL:
            syncState.materialStates.copyToDeviceAsync( stream );
#ifdef OTK_USE_MDL
            syncState.mdlMaterialShaders.copyToDeviceAsync( stream );
#endif
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
    const bool hasDiffuseMap{ flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP ) };
    const bool hasAlphaMap{ flagSet( group.material.flags, MaterialFlags::ALPHA_MAP ) };

    // Check for loaded diffuse map
    if( hasDiffuseMap && !m_demandTextureCache->hasDiffuseTextureForFile( group.diffuseMapFileName ) )
    {
        return {};
    }

    // Check for loaded alpha map
    if( hasAlphaMap && !m_demandTextureCache->hasAlphaTextureForFile( group.alphaMapFileName ) )
    {
        return {};
    }

    std::optional<uint_t> diffuseTextureId;
    if( hasDiffuseMap )
    {
        diffuseTextureId = m_demandTextureCache->createDiffuseTextureFromFile( group.diffuseMapFileName );
    }
    std::optional<uint_t> alphaTextureId;
    if( hasAlphaMap )
    {
        alphaTextureId = m_demandTextureCache->createAlphaTextureFromFile( group.alphaMapFileName );
    }

    // TODO: consider a sorted container for binary search instead of linear search of m_realizedMaterials
    for( uint_t materialId = 0; materialId < syncState.realizedMaterials.size(); ++materialId )
    {
        if( !isReadyLocalFallback( syncState, materialId ) )
        {
            continue;
        }

        const PhongMaterial& entry{ syncState.realizedMaterials[materialId] };
        if( group.material.Ka == entry.Ka                 //
            && group.material.Kd == entry.Kd              //
            && group.material.Ks == entry.Ks              //
            && group.material.Kr == entry.Kr              //
            && group.material.phongExp == entry.phongExp  //
            && ( group.material.flags & ( MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP ) )
                   == ( entry.flags & ( MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP ) )
            && ( !diffuseTextureId || diffuseTextureId == entry.diffuseTextureId )
            && ( !alphaTextureId || alphaTextureId == entry.alphaTextureId ) )
        {
            return { materialId };
        }
    }

    return {};
}

bool PbrtMaterialResolver::resolveGeometryToExistingMaterial( uint_t                  proxyGeomId,
                                                              uint_t                  materialId,
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
    geom->materialIds.push_back( materialId );
    OTK_ASSERT( materialId < syncState.realizedMaterials.size() );
    OTK_ASSERT( isReadyLocalFallback( syncState, materialId ) );
    const uint_t index{ geom->instance.instance.instanceId };
    grow( syncState.realizedNormals, index + 1 );
    grow( syncState.realizedUVs, index + 1 );
    OTK_ASSERT( index < syncState.realizedNormals.size() );
    OTK_ASSERT( index < syncState.realizedUVs.size() );
    syncState.realizedNormals[index] = geom->instance.devNormals;
    syncState.realizedUVs[index]     = geom->instance.devUVs;
    syncState.primitiveMaterials.push_back( PrimitiveMaterialRange{ group.primitiveIndexEnd, materialId } );
    m_proxyMaterialGeometries[materialId] = geom;
    ++m_stats.numMaterialsReused;

    if( m_options.verboseProxyGeometryResolution )
    {
        std::cout << "Resolved proxy geometry id " << proxyGeomId << " to geometry instance id " << geom->instanceIndex
                  << " with existing material id " << materialId << '\n';
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
    setMaterialState( syncState, materialId, localFallbackState( materialId ) );
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
