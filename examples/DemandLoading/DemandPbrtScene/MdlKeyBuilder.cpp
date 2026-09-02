// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlKeyBuilder.h"

#ifdef OTK_USE_MDL

#include <algorithm>
#include <initializer_list>
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
        "Kd",          "Kr",    "Ks",      "Kt",      "alpha",    "amount",     "bumpmap",
        "eta",         "index", "k",       "mfp",     "opacity",  "reflect",    "roughness",
        "shadowalpha", "sigma", "sigma_a", "sigma_s", "transmit", "uroughness", "vroughness",
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

std::string paramSetToString( const ::pbrt::ParamSet& params )
{
    ::pbrt::ParamSet copy{ params };
    return copy.ToString();
}

struct SourceSignatureEmitter
{
    void appendTextureReference( std::ostringstream& out, const std::string& paramName, const std::string& ) const
    {
        out << "|texture-ref(" << paramName << ")=";
    }

    void appendMaterialReference( std::ostringstream& out, const std::string& paramName, const std::string& ) const
    {
        out << "|material-ref(" << paramName << ")=";
    }

    void appendRecursiveTexture( std::ostringstream& out, const std::string&, const otk::pbrt::PbrtTexture& texture ) const
    {
        out << "texture(" << texture.valueType << ":" << texture.type << ";recursive)";
    }

    void appendTexture( std::ostringstream& out, const std::string&, const otk::pbrt::PbrtTexture& texture ) const
    {
        out << "texture(" << texture.valueType << ":" << texture.type;
    }

    void appendMaterial( std::ostringstream& out, const std::string& type, const ::pbrt::ParamSet& ) const
    {
        out << "material(" << type;
    }
};

struct InstanceSignatureEmitter
{
    void appendTextureReference( std::ostringstream& out, const std::string& paramName, const std::string& textureName ) const
    {
        out << "|texture-ref(" << paramName << ")=" << textureName << ':';
    }

    void appendMaterialReference( std::ostringstream& out, const std::string& paramName, const std::string& materialName ) const
    {
        out << "|material-ref(" << paramName << ")=" << materialName << ':';
    }

    void appendRecursiveTexture( std::ostringstream& out,
                                 const std::string& graphKey,
                                 const otk::pbrt::PbrtTexture& texture ) const
    {
        out << "texture(key=" << graphKey << ",name=" << texture.name << ",kind=" << texture.valueType << ':'
            << texture.type << ";recursive)";
    }

    void appendTexture( std::ostringstream& out, const std::string& graphKey, const otk::pbrt::PbrtTexture& texture ) const
    {
        out << "texture(key=" << graphKey << ",name=" << texture.name << ",kind=" << texture.valueType << ':'
            << texture.type << "|params=" << paramSetToString( texture.params );
    }

    void appendMaterial( std::ostringstream& out, const std::string& type, const ::pbrt::ParamSet& params ) const
    {
        out << "material(" << type << "|params=" << paramSetToString( params );
    }
};

template <typename Emitter>
void appendTextureGraphSignature( std::ostringstream&                 out,
                                  const std::string&                  graphKey,
                                  const otk::pbrt::PbrtTexture&       texture,
                                  const otk::pbrt::PbrtMaterialGraph& graph,
                                  std::vector<std::string>&           textureStack,
                                  const Emitter&                      emitter );

template <typename Emitter>
void appendTextureReference( std::ostringstream&                 out,
                             const std::string&                  paramName,
                             const std::string&                  textureName,
                             const otk::pbrt::PbrtMaterialGraph& graph,
                             std::vector<std::string>&           textureStack,
                             const Emitter&                      emitter )
{
    emitter.appendTextureReference( out, paramName, textureName );
    bool found{ false };
    for( otk::pbrt::PbrtTextureMap::const_iterator it = graph.textures.begin(); it != graph.textures.end(); ++it )
    {
        if( it->second.name == textureName )
        {
            if( found )
                out << ",";
            appendTextureGraphSignature( out, it->first, it->second, graph, textureStack, emitter );
            found = true;
        }
    }
    if( !found )
    {
        out << "missing";
    }
}

template <typename Emitter>
void appendTextureReferences( std::ostringstream&                 out,
                              const ::pbrt::ParamSet&             params,
                              const std::vector<std::string>&     paramNames,
                              const otk::pbrt::PbrtMaterialGraph& graph,
                              std::vector<std::string>&           textureStack,
                              const Emitter&                      emitter )
{
    for( std::vector<std::string>::const_iterator it = paramNames.begin(); it != paramNames.end(); ++it )
    {
        const std::string textureName{ params.FindTexture( *it ) };
        if( !textureName.empty() )
        {
            appendTextureReference( out, *it, textureName, graph, textureStack, emitter );
        }
    }
}

template <typename Emitter>
void appendMaterialGraphSignature( std::ostringstream&                 out,
                                   const std::string&                  type,
                                   const ::pbrt::ParamSet&             params,
                                   const otk::pbrt::PbrtMaterialGraph& graph,
                                   std::vector<std::string>&           materialStack,
                                   std::vector<std::string>&           textureStack,
                                   const Emitter&                      emitter );

template <typename Emitter>
void appendMaterialReference( std::ostringstream&                 out,
                              const std::string&                  paramName,
                              const std::string&                  materialName,
                              const otk::pbrt::PbrtMaterialGraph& graph,
                              std::vector<std::string>&           materialStack,
                              std::vector<std::string>&           textureStack,
                              const Emitter&                      emitter )
{
    emitter.appendMaterialReference( out, paramName, materialName );
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
    appendMaterialGraphSignature( out, material->second.type, material->second.params, graph, materialStack, textureStack,
                                  emitter );
    materialStack.pop_back();
}

template <typename Emitter>
void appendMaterialReferences( std::ostringstream&                 out,
                               const ::pbrt::ParamSet&             params,
                               const otk::pbrt::PbrtMaterialGraph& graph,
                               std::vector<std::string>&           materialStack,
                               std::vector<std::string>&           textureStack,
                               const Emitter&                      emitter )
{
    const std::vector<std::string>& paramNames{ namedMaterialParamNames() };
    for( std::vector<std::string>::const_iterator it = paramNames.begin(); it != paramNames.end(); ++it )
    {
        const std::string materialName{ params.FindOneString( *it, std::string{} ) };
        if( !materialName.empty() )
        {
            appendMaterialReference( out, *it, materialName, graph, materialStack, textureStack, emitter );
        }
    }
}

template <typename Emitter>
void appendTextureGraphSignature( std::ostringstream&                 out,
                                  const std::string&                  graphKey,
                                  const otk::pbrt::PbrtTexture&       texture,
                                  const otk::pbrt::PbrtMaterialGraph& graph,
                                  std::vector<std::string>&           textureStack,
                                  const Emitter&                      emitter )
{
    if( contains( textureStack, graphKey ) )
    {
        emitter.appendRecursiveTexture( out, graphKey, texture );
        return;
    }

    textureStack.push_back( graphKey );
    emitter.appendTexture( out, graphKey, texture );
    appendTextureReferences( out, texture.params, textureTextureParamNames(), graph, textureStack, emitter );
    out << ")";
    textureStack.pop_back();
}

template <typename Emitter>
void appendMaterialGraphSignature( std::ostringstream&                 out,
                                   const std::string&                  type,
                                   const ::pbrt::ParamSet&             params,
                                   const otk::pbrt::PbrtMaterialGraph& graph,
                                   std::vector<std::string>&           materialStack,
                                   std::vector<std::string>&           textureStack,
                                   const Emitter&                      emitter )
{
    emitter.appendMaterial( out, type, params );
    appendTextureReferences( out, params, materialTextureParamNames(), graph, textureStack, emitter );
    appendMaterialReferences( out, params, graph, materialStack, textureStack, emitter );
    out << ")";
}

void appendMaterialSignature( std::ostringstream&                 out,
                              const std::string&                  type,
                              const ::pbrt::ParamSet&             params,
                              const otk::pbrt::PbrtMaterialGraph& graph,
                              std::vector<std::string>&           materialStack,
                              std::vector<std::string>&           textureStack )
{
    appendMaterialGraphSignature( out, type, params, graph, materialStack, textureStack, SourceSignatureEmitter{} );
}

void appendMaterialInstanceSignature( std::ostringstream&                 out,
                                      const std::string&                  type,
                                      const ::pbrt::ParamSet&             params,
                                      const otk::pbrt::PbrtMaterialGraph& graph,
                                      std::vector<std::string>&           materialStack,
                                      std::vector<std::string>&           textureStack )
{
    appendMaterialGraphSignature( out, type, params, graph, materialStack, textureStack, InstanceSignatureEmitter{} );
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

bool operator==( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs )
{
    return lhs.sourceKey == rhs.sourceKey && lhs.signature == rhs.signature
           && lhs.sourceShapeProgramReusable == rhs.sourceShapeProgramReusable;
}

bool operator!=( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs )
{
    return !( lhs == rhs );
}

bool operator<( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs )
{
    if( lhs.sourceKey != rhs.sourceKey )
    {
        return lhs.sourceKey < rhs.sourceKey;
    }
    if( lhs.signature != rhs.signature )
    {
        return lhs.signature < rhs.signature;
    }
    return lhs.sourceShapeProgramReusable < rhs.sourceShapeProgramReusable;
}

std::string toString( const MdlMaterialInstanceKey& key )
{
    return "source=" + toString( key.sourceKey ) + "|instance=" + key.signature
           + ( key.sourceShapeProgramReusable ? "|source-shape-program=reusable" : "|source-shape-program=instance" );
}

MdlMaterialInstanceKey makeMdlMaterialInstanceKey( const otk::pbrt::PbrtMaterial& material )
{
    MdlMaterialInstanceKey result;
    result.sourceKey                  = makeMdlShaderKey( material );
    result.sourceShapeProgramReusable = true;

    std::ostringstream       signature;
    std::vector<std::string> materialStack;
    std::vector<std::string> textureStack;

    signature << "pbrt-mdl-instance-v1";
    for( std::vector<std::string>::const_iterator it = material.graph.fallbackReasons.begin();
         it != material.graph.fallbackReasons.end(); ++it )
    {
        signature << "|graph-fallback=" << *it;
    }
    appendMaterialInstanceSignature( signature, material.type, material.params, material.graph, materialStack, textureStack );

    result.signature = signature.str();
    return result;
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL

