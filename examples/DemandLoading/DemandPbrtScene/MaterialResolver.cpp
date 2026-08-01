// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MaterialResolver.h"

#include "DemandPbrtScene/Config.h"
#include "DemandPbrtScene/Conversions.h"
#include "DemandPbrtScene/DemandTextureCache.h"
#include "DemandPbrtScene/FrameStopwatch.h"
#ifdef OTK_USE_MDL
#include "DemandPbrtScene/FourierBsdfTable.h"
#include "DemandPbrtScene/MaterialAdapters.h"
#include "DemandPbrtScene/MdlShaderCache.h"
#endif
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/ProgramGroups.h"
#include "DemandPbrtScene/SceneGeometry.h"
#include "DemandPbrtScene/SceneProxy.h"
#include "DemandPbrtScene/SceneSyncState.h"

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <algorithm>
#include <cmath>
#ifdef OTK_USE_MDL
#include <filesystem>
#include <fstream>
#endif
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
    MaterialState resolveFourierMaterialState( SceneSyncState& sync, GeometryInstance& instance, const MaterialGroup& group, uint_t materialId );
    FourierBsdfTableLoadResult loadFourierBsdfTableResourceState( const MaterialGroup& group );
    void queuePendingMdlMaterial( const MdlMaterialInstanceKey& key, const PendingMdlMaterial& material );
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

void includeDiffuseTextureId( SceneSyncState& sync, uint_t textureId )
{
    sync.minDiffuseTextureId = std::min( textureId, sync.minDiffuseTextureId );
    sync.maxDiffuseTextureId = std::max( textureId, sync.maxDiffuseTextureId );
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

MaterialState fourierTableReadyState( uint_t materialId, uint_t resourceId )
{
    return makeMaterialState( materialId, MaterialBackend::FOURIER_TABLE_READY, resourceId );
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

otk::pbrt::PbrtMaterial generatedMdlMaterialForNamedMaterial( const otk::pbrt::PbrtMaterial&      parent,
                                                              const otk::pbrt::PbrtNamedMaterial& namedMaterial )
{
    otk::pbrt::PbrtMaterial material;
    material.type   = generatedMdlNamedMaterialType( namedMaterial );
    material.params = namedMaterial.params;
    material.graph  = parent.graph;
    return material;
}

bool hasGeneratedMdlConstantFloatTexture( const otk::pbrt::PbrtMaterial& material, const std::string& textureName )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( material ) };
    for( const MdlBoundMaterialParameter& parameter : parameters )
    {
        if( parameter.name == "amount" && parameter.type == MdlBoundParameterType::FLOAT
            && material.params.FindTexture( "amount" ) == textureName )
        {
            return true;
        }
    }
    return false;
}

bool hasGeneratedMdlFoldableTextureParameter( const otk::pbrt::PbrtMaterial& material, const std::string& paramName, MdlBoundParameterType type )
{
    if( material.params.FindTexture( paramName ).empty() )
    {
        return false;
    }

    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( material ) };
    for( const MdlBoundMaterialParameter& parameter : parameters )
    {
        if( parameter.name == paramName && parameter.type == type )
        {
            return true;
        }
    }
    return false;
}

bool isGeneratedMdlNamedMaterialAlphaTextureParam( const std::string& paramName )
{
    return paramName == "alpha" || paramName == "shadowalpha" || paramName == "opacity";
}

bool isGeneratedMdlNamedMaterialFloatTextureParam( const std::string& paramName )
{
    return paramName == "bumpmap" || isGeneratedMdlNamedMaterialAlphaTextureParam( paramName );
}

bool usesGeneratedMdlNamedMaterialKd( const std::string& type )
{
    return type == "matte" || type == "plastic" || type == "uber" || type == "substrate" || type == "translucent";
}

bool usesGeneratedMdlNamedMaterialKs( const std::string& type )
{
    return type == "plastic" || type == "uber" || type == "substrate" || type == "translucent";
}

bool usesGeneratedMdlNamedMaterialKr( const std::string& type )
{
    return type == "uber" || type == "mirror" || type == "glass";
}

bool usesGeneratedMdlKt( const std::string& type )
{
    return type == "uber" || type == "glass";
}

bool hasGeneratedMdlDemandTexture( MaterialFlags flags, MaterialFlags mapFlag, MaterialFlags allocatedFlag, const std::string& fileName )
{
    return flagSet( flags, mapFlag | allocatedFlag ) && !fileName.empty();
}

bool isDirectGeneratedMdlDemandTexture( const PbrtDemandTextureBinding& binding )
{
    return hasPbrtDemandTextureBinding( binding ) && !binding.transformed;
}

PbrtDemandTextureBinding generatedMdlNamedMaterialRuntimeTextureBinding( const otk::pbrt::PbrtMaterial& parent,
                                                                         const otk::pbrt::PbrtNamedMaterial& namedMaterial,
                                                                         const std::string& paramName )
{
    const otk::pbrt::PbrtMaterial  material{ generatedMdlMaterialForNamedMaterial( parent, namedMaterial ) };
    const PbrtDemandTextureBinding binding{ isGeneratedMdlNamedMaterialFloatTextureParam( paramName ) ?
                                                pbrtFloatTextureBinding( material, paramName.c_str() ) :
                                                pbrtColorTextureBinding( material, paramName.c_str() ) };
    if( !hasPbrtDemandTextureBinding( binding ) )
    {
        return pbrtDemandTextureBinding();
    }

    const std::string type{ material.type };
    if( paramName == "Kd" && usesGeneratedMdlNamedMaterialKd( type ) )
    {
        return binding;
    }
    if( paramName == "Ks" && usesGeneratedMdlNamedMaterialKs( type ) && isDirectGeneratedMdlDemandTexture( binding ) )
    {
        return binding;
    }
    if( paramName == "Kr" && usesGeneratedMdlNamedMaterialKr( type ) && isDirectGeneratedMdlDemandTexture( binding ) )
    {
        return binding;
    }
    if( paramName == "bumpmap" || isGeneratedMdlNamedMaterialAlphaTextureParam( paramName ) )
    {
        return binding;
    }
    return pbrtDemandTextureBinding();
}

bool hasGeneratedMdlNamedMaterialRuntimeTextureBinding( const otk::pbrt::PbrtMaterial&      parent,
                                                        const otk::pbrt::PbrtNamedMaterial& namedMaterial,
                                                        const std::string&                  paramName )
{
    return hasPbrtDemandTextureBinding( generatedMdlNamedMaterialRuntimeTextureBinding( parent, namedMaterial, paramName ) );
}

bool supportsGeneratedMdlNamedMaterialTextureReference( const otk::pbrt::PbrtMaterial&      parent,
                                                        const otk::pbrt::PbrtNamedMaterial& namedMaterial,
                                                        const std::string&                  paramName )
{
    if( namedMaterial.params.FindTexture( paramName ).empty() )
    {
        return true;
    }

    const otk::pbrt::PbrtMaterial material{ generatedMdlMaterialForNamedMaterial( parent, namedMaterial ) };
    if( hasGeneratedMdlFoldableTextureParameter( material, paramName, MdlBoundParameterType::COLOR )
        || hasGeneratedMdlFoldableTextureParameter( material, paramName, MdlBoundParameterType::FLOAT ) )
    {
        return true;
    }
    return hasGeneratedMdlNamedMaterialRuntimeTextureBinding( parent, namedMaterial, paramName );
}

bool supportsGeneratedMdlNamedMaterialTextureReferences( const otk::pbrt::PbrtMaterial&      parent,
                                                         const otk::pbrt::PbrtNamedMaterial& namedMaterial )
{
    static const char* const textureParams[] = {
        "Kd", "Kr",      "Ks",      "Kt",        "alpha",       "amount", "bumpmap",  "eta",        "index",
        "k",  "opacity", "reflect", "roughness", "shadowalpha", "sigma",  "transmit", "uroughness", "vroughness",
    };

    for( const char* const param : textureParams )
    {
        if( !supportsGeneratedMdlNamedMaterialTextureReference( parent, namedMaterial, param ) )
        {
            return false;
        }
    }
    return true;
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
           && supportsGeneratedMdlNamedMaterialTextureReferences( material, namedMaterial->second );
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

bool findGeneratedMdlSpectrumParamScalarity( const ::pbrt::ParamSet& params, const char* paramName, bool& scalar )
{
    if( !params.FindTexture( paramName ).empty() )
    {
        return false;
    }

    int                     count{};
    const ::pbrt::Spectrum* values{ params.FindSpectrum( paramName, &count ) };
    if( count <= 0 || values == nullptr )
    {
        return false;
    }

    float rgb[3]{};
    values[0].ToRGB( rgb );
    constexpr float epsilon{ 1.0e-6f };
    scalar = std::fabs( rgb[0] - rgb[1] ) <= epsilon && std::fabs( rgb[0] - rgb[2] ) <= epsilon;
    return true;
}

bool hasGeneratedMdlFloatParam( const ::pbrt::ParamSet& params, const char* paramName )
{
    if( !params.FindTexture( paramName ).empty() )
    {
        return false;
    }

    int          count{};
    const float* values{ params.FindFloat( paramName, &count ) };
    return count > 0 && values != nullptr;
}

bool supportsGeneratedMdlMixAmount( const otk::pbrt::PbrtMaterial& material )
{
    if( material.type != "mix" || !material.params.FindTexture( "amount" ).empty() )
    {
        return true;
    }
    if( hasGeneratedMdlFloatParam( material.params, "amount" ) )
    {
        return true;
    }

    bool scalarAmount{};
    if( findGeneratedMdlSpectrumParamScalarity( material.params, "amount", scalarAmount ) )
    {
        return scalarAmount;
    }
    return true;
}

PbrtDemandTextureBinding generatedMdlRuntimeTextureBinding( const otk::pbrt::PbrtMaterial& material,
                                                            const MaterialGroup&           group,
                                                            const std::string&             paramName )
{
    const PbrtDemandTextureBinding binding{ paramName == "bumpmap" ? pbrtFloatTextureBinding( material, paramName.c_str() ) :
                                                                     pbrtColorTextureBinding( material, paramName.c_str() ) };
    if( !hasPbrtDemandTextureBinding( binding ) )
    {
        return pbrtDemandTextureBinding();
    }

    if( paramName == "Kd" )
    {
        if( flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP ) && binding.fileName == group.diffuseMapFileName )
        {
            return binding;
        }
        return pbrtDemandTextureBinding();
    }
    if( paramName == "Kr" && material.type == "mirror" )
    {
        if( flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP ) && binding.fileName == group.diffuseMapFileName )
        {
            return binding;
        }
        return pbrtDemandTextureBinding();
    }
    if( material.type == "uber" && ( paramName == "Ks" || paramName == "Kr" ) && isDirectGeneratedMdlDemandTexture( binding ) )
    {
        return binding;
    }
    if( paramName == "Kt" && usesGeneratedMdlKt( material.type ) && isDirectGeneratedMdlDemandTexture( binding ) )
    {
        return binding;
    }
    if( paramName == "bumpmap" )
    {
        return binding;
    }
    return pbrtDemandTextureBinding();
}

bool hasGeneratedMdlRuntimeTextureBinding( const otk::pbrt::PbrtMaterial& material, const MaterialGroup& group, const std::string& paramName )
{
    return hasPbrtDemandTextureBinding( generatedMdlRuntimeTextureBinding( material, group, paramName ) );
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
        if( paramName == "bumpmap" )
        {
            if( !hasGeneratedMdlRuntimeTextureBinding( material, group, paramName ) )
            {
                return false;
            }
            continue;
        }
        if( hasGeneratedMdlFoldableTextureParameter( material, paramName, MdlBoundParameterType::COLOR ) )
        {
            continue;
        }
        if( hasGeneratedMdlFoldableTextureParameter( material, paramName, MdlBoundParameterType::FLOAT ) )
        {
            continue;
        }
        if( paramName == "Kd" )
        {
            if( !hasDiffuseMap || !hasGeneratedMdlRuntimeTextureBinding( material, group, "Kd" ) )
            {
                return false;
            }
            continue;
        }
        if( paramName == "Kr" && material.type == "mirror" )
        {
            if( !hasDiffuseMap || !hasGeneratedMdlRuntimeTextureBinding( material, group, "Kr" ) )
            {
                return false;
            }
            continue;
        }
        if( material.type == "uber" && ( paramName == "Ks" || paramName == "Kr" ) )
        {
            if( !hasGeneratedMdlRuntimeTextureBinding( material, group, paramName ) )
            {
                return false;
            }
            continue;
        }
        if( paramName == "Kt" && usesGeneratedMdlKt( material.type ) )
        {
            if( !hasGeneratedMdlRuntimeTextureBinding( material, group, paramName ) )
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
    if( !supportsGeneratedMdlMixAmount( material ) || !supportsGeneratedMdlTextureReferences( material, group ) )
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

void clearMaterialGroupMdlTextureBindings( MaterialGroup& group )
{
    for( MaterialGroupMdlTextureBinding& binding : group.mdlTextureBindings )
    {
        binding.fileName.clear();
        binding.binding = invalidMdlMaterialTextureBinding();
    }
}

void setMaterialGroupMdlTextureBinding( MaterialGroup& group, uint_t index, const PbrtDemandTextureBinding& binding, uint_t textureId )
{
    OTK_ASSERT( index < group.mdlTextureBindings.size() );
    group.mdlTextureBindings[index].fileName = binding.fileName;
    group.mdlTextureBindings[index].binding  = MdlMaterialTextureBinding{ textureId, binding.scale, binding.bias };
}

void setGeneratedMdlDiffuseTextureBinding( MaterialGroup& group, SceneSyncState& sync, DemandTextureCache& demandTextureCache )
{
    if( !group.pbrtMaterial || !flagSet( group.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
    {
        return;
    }

    const char* const paramName{ group.pbrtMaterial->type == "mirror" ? "Kr" : "Kd" };
    const PbrtDemandTextureBinding binding{ generatedMdlRuntimeTextureBinding( *group.pbrtMaterial, group, paramName ) };
    if( hasPbrtDemandTextureBinding( binding ) )
    {
        const uint_t textureId{ demandTextureCache.createLinearTextureFromFile( binding.fileName, binding.gamma ) };
        includeDiffuseTextureId( sync, textureId );
        setMaterialGroupMdlTextureBinding( group, MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX, binding, textureId );
    }
}

void createGeneratedMdlTextureBinding( MaterialGroup&                  group,
                                       SceneSyncState&                 sync,
                                       DemandTextureCache&             demandTextureCache,
                                       const PbrtDemandTextureBinding& binding,
                                       uint_t                          index )
{
    if( !hasPbrtDemandTextureBinding( binding ) )
    {
        return;
    }

    const uint_t textureId{ demandTextureCache.createLinearTextureFromFile( binding.fileName, binding.gamma ) };
    includeDiffuseTextureId( sync, textureId );
    setMaterialGroupMdlTextureBinding( group, index, binding, textureId );
}

void createGeneratedMdlTextureBinding( MaterialGroup&      group,
                                       SceneSyncState&     sync,
                                       DemandTextureCache& demandTextureCache,
                                       const char*         paramName,
                                       uint_t              index )
{
    createGeneratedMdlTextureBinding( group, sync, demandTextureCache,
                                      generatedMdlRuntimeTextureBinding( *group.pbrtMaterial, group, paramName ), index );
}

uint_t generatedMdlMixNamedMaterialTextureBindingIndex( uint_t namedMaterialIndex, uint_t offset )
{
    const uint_t base{ namedMaterialIndex == 0U ? MDL_MATERIAL_MIX_NAMED_0_TEXTURE_BINDING_BASE :
                                                  MDL_MATERIAL_MIX_NAMED_1_TEXTURE_BINDING_BASE };
    return base + offset;
}

const otk::pbrt::PbrtNamedMaterial* findGeneratedMdlMixNamedMaterial( const MaterialGroup& group, const char* paramName )
{
    if( !group.pbrtMaterial )
    {
        return nullptr;
    }

    const std::string materialName{ group.pbrtMaterial->params.FindOneString( paramName, std::string{} ) };
    if( materialName.empty() )
    {
        return nullptr;
    }

    const otk::pbrt::PbrtNamedMaterialMap::const_iterator namedMaterial =
        group.pbrtMaterial->graph.namedMaterials.find( materialName );
    if( namedMaterial == group.pbrtMaterial->graph.namedMaterials.end() )
    {
        return nullptr;
    }
    return &namedMaterial->second;
}

PbrtDemandTextureBinding generatedMdlNamedMaterialAlphaCutoutBinding( const otk::pbrt::PbrtMaterial& parent,
                                                                      const otk::pbrt::PbrtNamedMaterial& namedMaterial )
{
    for( const char* const paramName : { "alpha", "shadowalpha", "opacity" } )
    {
        const PbrtDemandTextureBinding binding{ generatedMdlNamedMaterialRuntimeTextureBinding( parent, namedMaterial, paramName ) };
        if( hasPbrtDemandTextureBinding( binding ) )
        {
            return binding;
        }
    }
    return pbrtDemandTextureBinding();
}

void setGeneratedMdlMixAlphaCutout( const Options& options, const GeometryInstance& instance, MaterialGroup& group )
{
    if( !options.useMdlMaterials || instance.primitive != GeometryPrimitive::TRIANGLE
        || instance.groups.size() != 1 || !group.pbrtMaterial || group.pbrtMaterial->type != "mix"
        || flagSet( group.material.flags, MaterialFlags::ALPHA_MAP ) || !group.pbrtMaterial->graph.fallbackReasons.empty()
        || !supportsGeneratedMdlNamedMaterialReferences( *group.pbrtMaterial ) || !supportsGeneratedMdlMixAmount( *group.pbrtMaterial )
        || !supportsGeneratedMdlTextureReferences( *group.pbrtMaterial, group ) )
    {
        return;
    }

    for( const char* const paramName : { "namedmaterial1", "namedmaterial2" } )
    {
        const otk::pbrt::PbrtNamedMaterial* namedMaterial{ findGeneratedMdlMixNamedMaterial( group, paramName ) };
        if( namedMaterial == nullptr )
        {
            continue;
        }

        const PbrtDemandTextureBinding binding{ generatedMdlNamedMaterialAlphaCutoutBinding( *group.pbrtMaterial, *namedMaterial ) };
        if( hasPbrtDemandTextureBinding( binding ) )
        {
            group.alphaMapFileName = binding.fileName;
            group.material.flags |= MaterialFlags::ALPHA_MAP;
            return;
        }
    }
}

void createGeneratedMdlNamedMaterialTextureBinding( MaterialGroup&                      group,
                                                    SceneSyncState&                     sync,
                                                    DemandTextureCache&                 demandTextureCache,
                                                    const otk::pbrt::PbrtNamedMaterial& namedMaterial,
                                                    uint_t                              namedMaterialIndex,
                                                    const char*                         paramName,
                                                    uint_t                              offset )
{
    createGeneratedMdlTextureBinding( group, sync, demandTextureCache,
                                      generatedMdlNamedMaterialRuntimeTextureBinding( *group.pbrtMaterial, namedMaterial, paramName ),
                                      generatedMdlMixNamedMaterialTextureBindingIndex( namedMaterialIndex, offset ) );
}

void createGeneratedMdlNamedMaterialAlphaTextureBinding( MaterialGroup&                      group,
                                                         SceneSyncState&                     sync,
                                                         DemandTextureCache&                 demandTextureCache,
                                                         const otk::pbrt::PbrtNamedMaterial& namedMaterial,
                                                         uint_t                              namedMaterialIndex )
{
    const PbrtDemandTextureBinding binding{ generatedMdlNamedMaterialAlphaCutoutBinding( *group.pbrtMaterial, namedMaterial ) };
    if( !hasPbrtDemandTextureBinding( binding ) )
    {
        return;
    }
    createGeneratedMdlTextureBinding( group, sync, demandTextureCache, binding,
                                      generatedMdlMixNamedMaterialTextureBindingIndex(
                                          namedMaterialIndex, MDL_MATERIAL_MIX_NAMED_ALPHA_TEXTURE_BINDING_OFFSET ) );
}

void createGeneratedMdlMixNamedMaterialTextureBindings( MaterialGroup&      group,
                                                        SceneSyncState&     sync,
                                                        DemandTextureCache& demandTextureCache,
                                                        const char*         paramName,
                                                        uint_t              namedMaterialIndex )
{
    const otk::pbrt::PbrtNamedMaterial* namedMaterial{ findGeneratedMdlMixNamedMaterial( group, paramName ) };
    if( namedMaterial == nullptr )
    {
        return;
    }

    createGeneratedMdlNamedMaterialTextureBinding( group, sync, demandTextureCache, *namedMaterial, namedMaterialIndex,
                                                   "Kd", MDL_MATERIAL_MIX_NAMED_KD_TEXTURE_BINDING_OFFSET );
    createGeneratedMdlNamedMaterialTextureBinding( group, sync, demandTextureCache, *namedMaterial, namedMaterialIndex,
                                                   "Ks", MDL_MATERIAL_MIX_NAMED_KS_TEXTURE_BINDING_OFFSET );
    createGeneratedMdlNamedMaterialTextureBinding( group, sync, demandTextureCache, *namedMaterial, namedMaterialIndex,
                                                   "Kr", MDL_MATERIAL_MIX_NAMED_KR_TEXTURE_BINDING_OFFSET );
    createGeneratedMdlNamedMaterialAlphaTextureBinding( group, sync, demandTextureCache, *namedMaterial, namedMaterialIndex );
    createGeneratedMdlNamedMaterialTextureBinding( group, sync, demandTextureCache, *namedMaterial, namedMaterialIndex,
                                                   "bumpmap", MDL_MATERIAL_MIX_NAMED_BUMPMAP_TEXTURE_BINDING_OFFSET );
}

void createGeneratedMdlMixTextureBindings( MaterialGroup& group, SceneSyncState& sync, DemandTextureCache& demandTextureCache )
{
    createGeneratedMdlMixNamedMaterialTextureBindings( group, sync, demandTextureCache, "namedmaterial1", 0U );
    createGeneratedMdlMixNamedMaterialTextureBindings( group, sync, demandTextureCache, "namedmaterial2", 1U );
}

void resolveGeneratedMdlTextureBindings( const Options&          options,
                                         const GeometryInstance& instance,
                                         MaterialGroup&          group,
                                         DemandTextureCache&     demandTextureCache,
                                         SceneSyncState&         sync )
{
    clearMaterialGroupMdlTextureBindings( group );
    if( !options.useMdlMaterials || instance.primitive != GeometryPrimitive::TRIANGLE
        || instance.groups.size() != 1 || !group.pbrtMaterial || !supportsGeneratedMdlMaterial( group.pbrtMaterial->type )
        || !group.pbrtMaterial->graph.fallbackReasons.empty() || !supportsGeneratedMdlNamedMaterialReferences( *group.pbrtMaterial )
        || !supportsGeneratedMdlMixAmount( *group.pbrtMaterial )
        || !supportsGeneratedMdlTextureReferences( *group.pbrtMaterial, group ) )
    {
        return;
    }

    setGeneratedMdlDiffuseTextureBinding( group, sync, demandTextureCache );
    if( group.pbrtMaterial->type == "uber" )
    {
        createGeneratedMdlTextureBinding( group, sync, demandTextureCache, "Ks", MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX );
        createGeneratedMdlTextureBinding( group, sync, demandTextureCache, "Kr", MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX );
    }
    if( usesGeneratedMdlKt( group.pbrtMaterial->type ) )
    {
        createGeneratedMdlTextureBinding( group, sync, demandTextureCache, "Kt", MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX );
    }
    createGeneratedMdlTextureBinding( group, sync, demandTextureCache, "bumpmap", MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX );
    if( group.pbrtMaterial->type == "mix" )
    {
        createGeneratedMdlMixTextureBindings( group, sync, demandTextureCache );
    }
}

bool usesGeneratedMdlMaterial( const Options& options, const GeometryInstance& instance, const MaterialGroup& group )
{
    return options.useMdlMaterials && instance.primitive == GeometryPrimitive::TRIANGLE
           && instance.groups.size() == 1 && group.pbrtMaterial && supportsGeneratedMdlMaterial( group.pbrtMaterial->type )
           && group.pbrtMaterial->graph.fallbackReasons.empty() && supportsGeneratedMdlNamedMaterialReferences( *group.pbrtMaterial )
           && !hasGeneratedMdlUnsupportedTextureReference( *group.pbrtMaterial, group );
}

bool usesGeneratedMdlFourierMaterial( const Options& options, const GeometryInstance& instance, const MaterialGroup& group )
{
    return options.useMdlMaterials && instance.primitive == GeometryPrimitive::TRIANGLE
           && instance.groups.size() == 1 && group.pbrtMaterial && group.pbrtMaterial->type == "fourier";
}

bool usesGeneratedMdlUnsupportedFallback( const Options& options, const GeometryInstance& instance, const MaterialGroup& group )
{
    return options.useMdlMaterials && instance.primitive == GeometryPrimitive::TRIANGLE
           && instance.groups.size() == 1 && group.pbrtMaterial;
}

std::string resolveFourierBsdfTableFileName( const otk::pbrt::PbrtMaterial& material, const std::string& sceneFile )
{
    const std::string rawFileName{ material.params.FindOneString( "bsdffile", std::string{} ) };
    if( rawFileName.empty() )
    {
        return {};
    }

    std::filesystem::path fileName{ rawFileName };
    if( fileName.is_relative() )
    {
        const std::filesystem::path sceneDir{ std::filesystem::path( sceneFile ).parent_path() };
        if( !sceneDir.empty() )
        {
            fileName = sceneDir / fileName;
        }
    }
    return fileName.lexically_normal().string();
}

MdlMaterialInstanceKey makeMaterialGroupMdlMaterialInstanceKey( const MaterialGroup& group )
{
    if( group.pbrtMaterial )
    {
        return makeMdlMaterialInstanceKey( *group.pbrtMaterial );
    }
    const MdlShaderKey sourceKey{ "pbrt-mdl-v1|missing-pbrt-material" };
    return MdlMaterialInstanceKey{ sourceKey, "pbrt-mdl-instance-v1|missing-pbrt-material", false };
}
#endif

void setMaterialState( SceneSyncState& sync, uint_t materialId, MaterialState state )
{
    grow( sync.materialStates, materialId + 1 );
    sync.materialStates[materialId] = state;
}

#ifdef OTK_USE_MDL
void setMdlMaterialShader( SceneSyncState& sync, uint_t materialId, const MdlMaterialShader& shader )
{
    grow( sync.mdlMaterialShaders, materialId + 1 );
    sync.mdlMaterialShaders[materialId] = shader;
    ++sync.materialShaderDataVersion;
}
#endif

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
    try
    {
        if( m_mdlShaderCompileCache.state( materialKey ) != MdlShaderCompileState::READY )
        {
            m_mdlShaderCompileCache.markCompiling( materialKey );
        }
        for( const PendingMdlMaterial& material : pendingMaterials )
        {
            const MdlMaterialShader shader{ m_programGroups->realizeMdlMaterialShader( material.instance, material.shaderKeyId ) };
            setMdlMaterialShader( sync, material.materialId, shader );
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
        std::cerr << "Generated MDL material build failed for " << toString( materialKey ) << ": " << e.what() << '\n';
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
        std::cerr << "Generated MDL material build failed for " << toString( materialKey ) << ": unknown failure\n";
        m_mdlShaderCompileCache.markFailed( materialKey, "Unknown MDL shader compile failure" );
        for( const PendingMdlMaterial& material : pendingMaterials )
        {
            setMaterialState( sync, material.materialId,
                              makeMaterialState( material.materialId, MaterialBackend::MDL_FAILED, material.shaderKeyId ) );
            ++m_stats.numMdlFallbackShaders;
        }
    }

    m_pendingMdlMaterials.erase( materialKey );
    return MaterialResolution::SHADER_DATA_ONLY;
}

MaterialState PbrtMaterialResolver::resolveMdlMaterialState( SceneSyncState&      sync,
                                                             GeometryInstance&    instance,
                                                             const MaterialGroup& group,
                                                             uint_t               materialId )
{
    const MdlMaterialInstanceKey  materialKey{ makeMaterialGroupMdlMaterialInstanceKey( group ) };
    const MdlShaderCompileRecord& record{ m_mdlShaderCompileCache.getRecord( materialKey ) };
    const uint_t                  shaderKeyId{ record.shaderKeyId };
    const MdlShaderCompileState   state{ m_mdlShaderCompileCache.state( materialKey ) };
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
        setMdlMaterialShader( sync, materialId, shader );
    };

    bindMdlProgram();
    switch( state )
    {
        case MdlShaderCompileState::READY:
            bindMdlShader();
            return mdlReadyState( materialId, shaderKeyId );
        case MdlShaderCompileState::QUEUED:
            if( !m_options.mdlSynchronousCompilation )
            {
                queuePending();
                return fallbackState( MaterialBackend::MDL_PENDING );
            }
            break;
        case MdlShaderCompileState::COMPILING:
            if( !m_options.mdlSynchronousCompilation )
            {
                queuePending();
            }
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
            queuePending();
            return fallbackState( MaterialBackend::MDL_PENDING );
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
        std::cerr << "Generated MDL material build failed for " << toString( materialKey ) << ": " << e.what() << '\n';
        m_mdlShaderCompileCache.markFailed( materialKey, e.what() );
    }
    catch( ... )
    {
        std::cerr << "Generated MDL material build failed for " << toString( materialKey ) << ": unknown failure\n";
        m_mdlShaderCompileCache.markFailed( materialKey, "Unknown MDL shader compile failure" );
    }

    return fallbackState( MaterialBackend::MDL_FAILED );
}

FourierBsdfTableLoadResult PbrtMaterialResolver::loadFourierBsdfTableResourceState( const MaterialGroup& group )
{
    if( !group.pbrtMaterial || group.pbrtMaterial->type != "fourier" )
    {
        return FourierBsdfTableLoadResult{};
    }

    const std::string fileName{ resolveFourierBsdfTableFileName( *group.pbrtMaterial, m_options.sceneFile ) };
    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName ) };
    switch( result.status )
    {
        case FourierBsdfTableLoadStatus::SUCCESS:
            ++m_stats.numFourierBsdfTableResourcesResolved;
            break;
        case FourierBsdfTableLoadStatus::FILE_NOT_FOUND:
            ++m_stats.numFourierBsdfTableResourcesMissing;
            break;
        case FourierBsdfTableLoadStatus::INVALID_HEADER:
        case FourierBsdfTableLoadStatus::TRUNCATED:
        case FourierBsdfTableLoadStatus::UNSUPPORTED:
        case FourierBsdfTableLoadStatus::MALFORMED:
            ++m_stats.numFourierBsdfTableResourcesInvalid;
            break;
    }
    return result;
}

MaterialState PbrtMaterialResolver::resolveFourierMaterialState( SceneSyncState&      sync,
                                                                 GeometryInstance&    instance,
                                                                 const MaterialGroup& group,
                                                                 uint_t               materialId )
{
    const FourierBsdfTableLoadResult table{ loadFourierBsdfTableResourceState( group ) };
    ++m_stats.numMdlFallbackShaders;
    if( !table )
    {
        instance.instance.sbtOffset = m_programGroups->getRealizedMaterialSbtOffset( instance );
        return unsupportedFallbackState( materialId );
    }

    try
    {
        const FourierMaterialResource resource{ m_programGroups->realizeFourierMaterialResource( instance, table.table ) };
        grow( sync.fourierMaterialResources, materialId + 1 );
        sync.fourierMaterialResources[materialId] = resource;
        instance.instance.sbtOffset               = m_programGroups->getFourierMaterialSbtOffset( instance );
        return fourierTableReadyState( materialId, resource.resourceId );
    }
    catch( const std::exception& e )
    {
        std::cerr << "Fourier BSDF table resource binding failed: " << e.what() << '\n';
    }
    catch( ... )
    {
        std::cerr << "Fourier BSDF table resource binding failed: unknown failure\n";
    }

    instance.instance.sbtOffset = m_programGroups->getRealizedMaterialSbtOffset( instance );
    return unsupportedFallbackState( materialId );
}
#endif

MaterialState PbrtMaterialResolver::resolveMaterialState( SceneSyncState& sync, GeometryInstance& instance, const MaterialGroup& group, uint_t materialId )
{
#ifdef OTK_USE_MDL
    if( usesGeneratedMdlFourierMaterial( m_options, instance, group ) )
    {
        return resolveFourierMaterialState( sync, instance, group, materialId );
    }
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
#ifdef OTK_USE_MDL
        setGeneratedMdlMixAlphaCutout( m_options, geom->instance, group );
#endif

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
            includeDiffuseTextureId( sync, diffuseTextureId );
            group.material.diffuseTextureId = diffuseTextureId;
            group.material.flags |= MaterialFlags::DIFFUSE_MAP_ALLOCATED;
        }
    }

#ifdef OTK_USE_MDL
    resolveGeneratedMdlTextureBindings( m_options, geom->instance, group, *m_demandTextureCache, sync );
#endif

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
        case MaterialResolution::SHADER_DATA_ONLY:
            syncState.materialStates.copyToDeviceAsync( stream );
#ifdef OTK_USE_MDL
            syncState.mdlMaterialShaders.copyToDeviceAsync( stream );
#endif
            break;
        case MaterialResolution::PARTIAL:
            syncState.materialStates.copyToDeviceAsync( stream );
#ifdef OTK_USE_MDL
            syncState.mdlMaterialShaders.copyToDeviceAsync( stream );
            syncState.fourierMaterialResources.copyToDeviceAsync( stream );
#endif
            syncState.partialMaterials.copyToDeviceAsync( stream );
            syncState.partialUVs.copyToDeviceAsync( stream );
            break;
        case MaterialResolution::FULL:
            syncState.materialStates.copyToDeviceAsync( stream );
#ifdef OTK_USE_MDL
            syncState.mdlMaterialShaders.copyToDeviceAsync( stream );
            syncState.fourierMaterialResources.copyToDeviceAsync( stream );
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
