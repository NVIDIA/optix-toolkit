// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlShaderCache.h"

#ifdef OTK_USE_MDL

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <vector>

namespace demandPbrtScene {
namespace {

std::vector<std::string> sortedNames( std::initializer_list<const char*> names )
{
    std::vector<std::string> result;
    result.reserve( names.size() );
    std::copy( names.begin(), names.end(), std::back_inserter( result ) );
    std::sort( result.begin(), result.end() );
    return result;
}

const std::vector<std::string>& materialTextureParamNames()
{
    static const std::vector<std::string> names{ sortedNames( {
        "Kd",
        "Kr",
        "Ks",
        "Kt",
        "alpha",
        "bumpmap",
        "eta",
        "index",
        "opacity",
        "roughness",
        "shadowalpha",
        "sigma",
        "uroughness",
        "vroughness",
    } ) };
    return names;
}

const std::vector<std::string>& textureTextureParamNames()
{
    static const std::vector<std::string> names{ sortedNames( {
        "amount",
        "scale",
        "tex",
        "tex1",
        "tex2",
    } ) };
    return names;
}

const std::vector<std::string>& namedMaterialParamNames()
{
    static const std::vector<std::string> names{ sortedNames( {
        "material",
        "material1",
        "material2",
        "namedmaterial1",
        "namedmaterial2",
    } ) };
    return names;
}

bool contains( const std::vector<std::string>& values, const std::string& value )
{
    return std::find( values.begin(), values.end(), value ) != values.end();
}

void appendTextureSignature( std::ostringstream&                 out,
                             const std::string&                  graphKey,
                             const otk::pbrt::PbrtTexture&       texture,
                             const otk::pbrt::PbrtMaterialGraph& graph,
                             std::vector<std::string>&           textureStack );

void appendTextureReference( std::ostringstream&                 out,
                             const std::string&                  paramName,
                             const std::string&                  textureName,
                             const otk::pbrt::PbrtMaterialGraph& graph,
                             std::vector<std::string>&           textureStack )
{
    out << "|texture-ref(" << paramName << ")=";
    bool found{ false };
    for( otk::pbrt::PbrtTextureMap::const_iterator it = graph.textures.begin(); it != graph.textures.end(); ++it )
    {
        if( it->second.name == textureName )
        {
            if( found )
                out << ",";
            appendTextureSignature( out, it->first, it->second, graph, textureStack );
            found = true;
        }
    }
    if( !found )
    {
        out << "missing";
    }
}

void appendTextureReferences( std::ostringstream&                 out,
                              const ::pbrt::ParamSet&             params,
                              const std::vector<std::string>&     paramNames,
                              const otk::pbrt::PbrtMaterialGraph& graph,
                              std::vector<std::string>&           textureStack )
{
    for( std::vector<std::string>::const_iterator it = paramNames.begin(); it != paramNames.end(); ++it )
    {
        const std::string textureName{ params.FindTexture( *it ) };
        if( !textureName.empty() )
        {
            appendTextureReference( out, *it, textureName, graph, textureStack );
        }
    }
}

void appendMaterialSignature( std::ostringstream&                 out,
                              const std::string&                  type,
                              const ::pbrt::ParamSet&             params,
                              const otk::pbrt::PbrtMaterialGraph& graph,
                              std::vector<std::string>&           materialStack,
                              std::vector<std::string>&           textureStack );

void appendMaterialReference( std::ostringstream&                 out,
                              const std::string&                  paramName,
                              const std::string&                  materialName,
                              const otk::pbrt::PbrtMaterialGraph& graph,
                              std::vector<std::string>&           materialStack,
                              std::vector<std::string>&           textureStack )
{
    out << "|material-ref(" << paramName << ")=";
    if( contains( materialStack, materialName ) )
    {
        out << "recursive";
        return;
    }

    const otk::pbrt::PbrtNamedMaterialMap::const_iterator material = graph.namedMaterials.find( materialName );
    if( material == graph.namedMaterials.end() )
    {
        out << "missing";
        return;
    }

    materialStack.push_back( materialName );
    appendMaterialSignature( out, material->second.type, material->second.params, graph, materialStack, textureStack );
    materialStack.pop_back();
}

void appendMaterialReferences( std::ostringstream&                 out,
                               const ::pbrt::ParamSet&             params,
                               const otk::pbrt::PbrtMaterialGraph& graph,
                               std::vector<std::string>&           materialStack,
                               std::vector<std::string>&           textureStack )
{
    const std::vector<std::string>& paramNames{ namedMaterialParamNames() };
    for( std::vector<std::string>::const_iterator it = paramNames.begin(); it != paramNames.end(); ++it )
    {
        const std::string materialName{ params.FindOneString( *it, std::string{} ) };
        if( !materialName.empty() )
        {
            appendMaterialReference( out, *it, materialName, graph, materialStack, textureStack );
        }
    }
}

void appendTextureSignature( std::ostringstream&                 out,
                             const std::string&                  graphKey,
                             const otk::pbrt::PbrtTexture&       texture,
                             const otk::pbrt::PbrtMaterialGraph& graph,
                             std::vector<std::string>&           textureStack )
{
    if( contains( textureStack, graphKey ) )
    {
        out << "texture(" << texture.valueType << ":" << texture.type << ";recursive)";
        return;
    }

    textureStack.push_back( graphKey );
    out << "texture(" << texture.valueType << ":" << texture.type;
    appendTextureReferences( out, texture.params, textureTextureParamNames(), graph, textureStack );
    out << ")";
    textureStack.pop_back();
}

void appendMaterialSignature( std::ostringstream&                 out,
                              const std::string&                  type,
                              const ::pbrt::ParamSet&             params,
                              const otk::pbrt::PbrtMaterialGraph& graph,
                              std::vector<std::string>&           materialStack,
                              std::vector<std::string>&           textureStack )
{
    out << "material(" << type;
    appendTextureReferences( out, params, materialTextureParamNames(), graph, textureStack );
    appendMaterialReferences( out, params, graph, materialStack, textureStack );
    out << ")";
}

std::string stableHash( const std::string& text )
{
    std::uint64_t hash{ 14695981039346656037ULL };
    for( std::string::const_iterator it = text.begin(); it != text.end(); ++it )
    {
        hash ^= static_cast<unsigned char>( *it );
        hash *= 1099511628211ULL;
    }

    std::ostringstream out;
    out << std::hex << std::setfill( '0' ) << std::setw( 16 ) << hash;
    return out.str();
}

GeneratedMdlSource generateSource( const MdlShaderKey& key )
{
    const std::string suffix{ stableHash( key.signature ) };

    GeneratedMdlSource result;
    result.moduleName   = "::otk::demand_pbrt_scene::pbrt_" + suffix;
    result.materialName = "material_" + suffix;

    std::ostringstream source;
    source << "mdl 1.6;\n"
           << "import ::df::*;\n"
           << "\n"
           << "export material " << result.materialName << "() = material(\n"
           << "    surface: material_surface(\n"
           << "        scattering: ::df::diffuse_reflection_bsdf(\n"
           << "            tint: color(0.8, 0.8, 0.8))));\n";
    result.source = source.str();
    return result;
}

}  // namespace

bool operator==( const MdlShaderKey& lhs, const MdlShaderKey& rhs )
{
    return lhs.signature == rhs.signature;
}

bool operator!=( const MdlShaderKey& lhs, const MdlShaderKey& rhs )
{
    return !( lhs == rhs );
}

bool operator<( const MdlShaderKey& lhs, const MdlShaderKey& rhs )
{
    return lhs.signature < rhs.signature;
}

std::string toString( const MdlShaderKey& key )
{
    return key.signature;
}

MdlShaderKey makeMdlShaderKey( const otk::pbrt::PbrtMaterial& material )
{
    std::ostringstream       signature;
    std::vector<std::string> materialStack;
    std::vector<std::string> textureStack;

    signature << "pbrt-mdl-v1";
    if( !material.graph.fallbackReasons.empty() )
    {
        signature << "|graph-fallback";
    }
    appendMaterialSignature( signature, material.type, material.params, material.graph, materialStack, textureStack );

    return MdlShaderKey{ signature.str() };
}

const GeneratedMdlSource& MdlGeneratedSourceCache::getOrCreate( const MdlShaderKey& key )
{
    std::map<MdlShaderKey, GeneratedMdlSource>::iterator it = m_sources.find( key );
    if( it == m_sources.end() )
    {
        it = m_sources.insert( std::make_pair( key, generateSource( key ) ) ).first;
    }
    return it->second;
}

bool MdlGeneratedSourceCache::contains( const MdlShaderKey& key ) const
{
    return m_sources.find( key ) != m_sources.end();
}

std::size_t MdlGeneratedSourceCache::size() const
{
    return m_sources.size();
}

void MdlGeneratedSourceCache::clear()
{
    m_sources.clear();
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
