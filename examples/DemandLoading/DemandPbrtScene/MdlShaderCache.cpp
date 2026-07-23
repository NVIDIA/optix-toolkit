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
#include <utility>
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

void appendUnsupportedReason( GeneratedMdlSource& result, const std::string& reason )
{
    if( !contains( result.unsupportedReasons, reason ) )
    {
        result.unsupportedReasons.push_back( reason );
    }
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

struct TextureLookup
{
    std::string                   graphKey;
    const otk::pbrt::PbrtTexture* texture;
};

TextureLookup findTexture( const otk::pbrt::PbrtMaterialGraph& graph, const std::string& textureName, const std::string& preferredValueType )
{
    TextureLookup fallback{ std::string{}, nullptr };
    for( otk::pbrt::PbrtTextureMap::const_iterator it = graph.textures.begin(); it != graph.textures.end(); ++it )
    {
        if( it->second.name != textureName )
        {
            continue;
        }
        if( preferredValueType.empty() || it->second.valueType == preferredValueType )
        {
            return TextureLookup{ it->first, &it->second };
        }
        if( fallback.texture == nullptr )
        {
            fallback = TextureLookup{ it->first, &it->second };
        }
    }
    return fallback;
}

std::string textureKind( const otk::pbrt::PbrtTexture& texture )
{
    return texture.valueType + ":" + texture.type;
}

bool isUnsupportedProceduralTexture( const otk::pbrt::PbrtTexture& texture )
{
    return texture.type == "marble" || texture.type == "fbm" || texture.type == "windy" || texture.type == "wrinkled";
}

class MdlTextureGraphGenerator
{
  public:
    MdlTextureGraphGenerator( const otk::pbrt::PbrtMaterialGraph& graph, GeneratedMdlSource& result )
        : m_graph( graph )
        , m_result( result )
    {
        for( std::vector<std::string>::const_iterator it = m_graph.fallbackReasons.begin();
             it != m_graph.fallbackReasons.end(); ++it )
        {
            appendUnsupportedReason( m_result, "PBRT material graph fallback: " + *it );
        }
    }

    std::string materialColorExpression( const ::pbrt::ParamSet& params,
                                         const std::string&      paramName,
                                         const std::string&      preferredValueType,
                                         const std::string&      defaultExpression )
    {
        const std::string textureName{ params.FindTexture( paramName ) };
        if( textureName.empty() )
        {
            return defaultExpression;
        }
        return textureReference( textureName, preferredValueType );
    }

    std::string sourcePreamble() const
    {
        std::ostringstream out;
        if( m_usesImageMap )
        {
            out << "color pbrt_demand_texture_2d(int texture_id) = color(0.8, 0.8, 0.8);\n";
        }
        if( m_usesCheckerboard )
        {
            out << "color pbrt_checkerboard_2d(color tex1, color tex2) = (tex1 + tex2) * 0.5;\n";
        }
        if( m_usesUnsupported )
        {
            out << "color pbrt_unsupported_texture() = color(1.0, 0.0, 1.0);\n";
        }
        if( !m_usesImageMap && !m_usesCheckerboard && !m_usesUnsupported )
        {
            return std::string{};
        }
        out << "\n";
        return out.str();
    }

    std::string functionDefinitions() const
    {
        std::ostringstream out;
        for( std::vector<std::string>::const_iterator it = m_functions.begin(); it != m_functions.end(); ++it )
        {
            out << *it << "\n";
        }
        return out.str();
    }

  private:
    std::string textureReference( const std::string& textureName, const std::string& preferredValueType )
    {
        const TextureLookup lookup{ findTexture( m_graph, textureName, preferredValueType ) };
        if( lookup.texture == nullptr )
        {
            appendUnsupportedReason( m_result, "Missing PBRT texture '" + textureName + "'" );
            return unsupportedTextureExpression();
        }
        if( contains( m_textureStack, lookup.graphKey ) )
        {
            appendUnsupportedReason( m_result, "Recursive PBRT texture reference '" + textureName + "'" );
            return unsupportedTextureExpression();
        }

        std::map<std::string, std::string>::const_iterator cached = m_textureFunctions.find( lookup.graphKey );
        if( cached != m_textureFunctions.end() )
        {
            return cached->second + "()";
        }

        m_textureStack.push_back( lookup.graphKey );
        const std::string functionName{ defineTextureFunction( lookup.graphKey, *lookup.texture ) };
        m_textureStack.pop_back();
        m_textureFunctions.insert( std::make_pair( lookup.graphKey, functionName ) );
        return functionName + "()";
    }

    std::string defineTextureFunction( const std::string& /*graphKey*/, const otk::pbrt::PbrtTexture& texture )
    {
        const std::string functionName{ "texture_" + std::to_string( m_nextTextureFunction++ ) };
        const std::string expression{ textureExpression( texture ) };

        std::ostringstream out;
        out << "// pbrt texture node: " << textureKind( texture ) << "\n";
        if( texture.type == "imagemap" )
        {
            out << "// demand texture parameter: texture_2d image_" << ( m_nextImageParameter - 1U ) << "\n";
        }
        out << "color " << functionName << "() = " << expression << ";\n";
        m_functions.push_back( out.str() );
        return functionName;
    }

    std::string textureExpression( const otk::pbrt::PbrtTexture& texture )
    {
        if( texture.type == "imagemap" )
        {
            m_usesImageMap = true;
            return "pbrt_demand_texture_2d(" + std::to_string( m_nextImageParameter++ ) + ")";
        }
        if( texture.type == "constant" )
        {
            return defaultColorExpression();
        }
        if( texture.type == "scale" )
        {
            const std::string tex1{ textureInputExpression( texture, "tex1", defaultColorExpression() ) };
            const std::string tex2{ textureInputExpression( texture, "tex2", defaultColorExpression() ) };
            return tex1 + " * " + tex2;
        }
        if( texture.type == "mix" )
        {
            const std::string tex1{ textureInputExpression( texture, "tex1", defaultColorExpression() ) };
            const std::string tex2{ textureInputExpression( texture, "tex2", defaultColorExpression() ) };
            return tex1 + " * (1.0 - 0.5) + " + tex2 + " * 0.5";
        }
        if( texture.type == "checkerboard" )
        {
            if( texture.params.FindOneString( "dimension", "2d" ) != "2d" )
            {
                appendUnsupportedReason( m_result, "Unsupported PBRT checkerboard dimension in " + textureKind( texture ) );
                return unsupportedTextureExpression();
            }
            m_usesCheckerboard = true;
            const std::string tex1{ textureInputExpression( texture, "tex1", "color(1.0, 1.0, 1.0)" ) };
            const std::string tex2{ textureInputExpression( texture, "tex2", "color(0.0, 0.0, 0.0)" ) };
            return "pbrt_checkerboard_2d(" + tex1 + ", " + tex2 + ")";
        }

        if( isUnsupportedProceduralTexture( texture ) )
        {
            appendUnsupportedReason( m_result, "Unsupported PBRT texture type " + textureKind( texture ) );
            return unsupportedTextureExpression();
        }

        appendUnsupportedReason( m_result, "Unsupported PBRT texture type " + textureKind( texture ) );
        return unsupportedTextureExpression();
    }

    std::string textureInputExpression( const otk::pbrt::PbrtTexture& texture, const std::string& paramName, const std::string& defaultExpression )
    {
        const std::string textureName{ texture.params.FindTexture( paramName ) };
        if( textureName.empty() )
        {
            return defaultExpression;
        }
        return textureReference( textureName, texture.valueType );
    }

    std::string unsupportedTextureExpression()
    {
        m_usesUnsupported = true;
        return "pbrt_unsupported_texture()";
    }

    static std::string defaultColorExpression() { return "color(1.0, 1.0, 1.0)"; }

    const otk::pbrt::PbrtMaterialGraph& m_graph;
    GeneratedMdlSource&                 m_result;
    std::vector<std::string>            m_textureStack;
    std::map<std::string, std::string>  m_textureFunctions;
    std::vector<std::string>            m_functions;
    unsigned int                        m_nextTextureFunction{};
    unsigned int                        m_nextImageParameter{};
    bool                                m_usesImageMap{};
    bool                                m_usesCheckerboard{};
    bool                                m_usesUnsupported{};
};

struct MdlMaterialParameter
{
    std::string type;
    std::string name;
    std::string defaultValue;
};

struct MdlMaterialModel
{
    std::vector<MdlMaterialParameter> parameters;
    std::vector<std::string>          comments;
    std::string                       helperDefinitions;
    std::string                       body;
};

void appendMaterialParameter( MdlMaterialModel& model, const std::string& type, const std::string& name, const std::string& defaultValue )
{
    model.parameters.push_back( MdlMaterialParameter{ type, name, defaultValue } );
}

std::string mdlParameterList( const std::vector<MdlMaterialParameter>& parameters )
{
    if( parameters.empty() )
    {
        return "()";
    }

    std::ostringstream out;
    out << "(\n";
    for( std::vector<MdlMaterialParameter>::const_iterator it = parameters.begin(); it != parameters.end(); ++it )
    {
        out << "    " << it->type << " " << it->name << " = " << it->defaultValue;
        if( it + 1 != parameters.end() )
        {
            out << ",";
        }
        out << "\n";
    }
    out << ")";
    return out.str();
}

std::string materialTextureCommentExpression( MdlTextureGraphGenerator& textureGraph,
                                              const ::pbrt::ParamSet&   params,
                                              const std::string&        paramName,
                                              const std::string&        preferredValueType )
{
    if( params.FindTexture( paramName ).empty() )
    {
        return "none";
    }
    return textureGraph.materialColorExpression( params, paramName, preferredValueType, "none" );
}

MdlMaterialModel makeMatteMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "float", "sigma", "0.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };

    model.comments.push_back( "pbrt material model: matte" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input sigma: sigma" );
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::diffuse_reflection_bsdf(\n"
        "            tint: "
        + kd + "))\n";
    return model;
}

MdlMaterialModel makePlasticMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string bumpmap{ materialTextureCommentExpression( textureGraph, material.params, "bumpmap", "float" ) };

    model.comments.push_back( "pbrt material model: plastic" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input bumpmap: " + bumpmap );
    model.comments.push_back( "pbrt material approximation: glossy lobe is represented but not connected" );
    model.helperDefinitions = "color pbrt_plastic_approximation_tint(color kd, color ks, float roughness) = kd;\n\n";
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::diffuse_reflection_bsdf(\n"
        "            tint: pbrt_plastic_approximation_tint("
        + kd + ", " + ks + ", roughness)))\n";
    return model;
}

MdlMaterialModel makeUberMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "color", "Kr", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "color", "Kt", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "float", "index", "1.5" );
    appendMaterialParameter( model, "float", "alpha", "1.0" );
    appendMaterialParameter( model, "float", "opacity", "1.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };
    const std::string kt{ textureGraph.materialColorExpression( material.params, "Kt", "color", "Kt" ) };
    const std::string alphaTexture{
        materialTextureCommentExpression( textureGraph, material.params, "alpha", "float" ) };
    const std::string opacityTexture{
        materialTextureCommentExpression( textureGraph, material.params, "opacity", "float" ) };
    const std::string bumpmap{ materialTextureCommentExpression( textureGraph, material.params, "bumpmap", "float" ) };

    model.comments.push_back( "pbrt material model: uber" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.comments.push_back( "pbrt material input Kt: " + kt );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input index: index" );
    model.comments.push_back( "pbrt material input alpha: alpha; texture=" + alphaTexture );
    model.comments.push_back( "pbrt material input opacity: opacity; texture=" + opacityTexture );
    model.comments.push_back( "pbrt material input bumpmap: " + bumpmap );
    model.comments.push_back(
        "pbrt material approximation: specular, reflection, and transmission lobes are represented but not connected" );
    model.helperDefinitions =
        "color pbrt_uber_approximation_tint(color kd, color ks, color kr, color kt, float roughness) = kd;\n\n";
    model.body = "    ior: color(index, index, index),\n"
                 "    surface: material_surface(\n"
                 "        scattering: ::df::diffuse_reflection_bsdf(\n"
                 "            tint: pbrt_uber_approximation_tint("
                 + kd + ", " + ks + ", " + kr + ", " + kt + ", roughness))),\n"
                 "    geometry: material_geometry(\n"
                 "        cutout_opacity: alpha * opacity)\n";
    return model;
}

MdlMaterialModel makeUnsupportedMaterialModel( const otk::pbrt::PbrtMaterial& material, GeneratedMdlSource& result )
{
    const std::string type{ material.type.empty() ? std::string{ "<empty>" } : material.type };
    appendUnsupportedReason( result, "Unsupported PBRT material type " + type );

    MdlMaterialModel model;
    model.comments.push_back( "pbrt material model: " + type );
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::diffuse_reflection_bsdf(\n"
        "            tint: color(1.0, 0.0, 1.0)))\n";
    return model;
}

MdlMaterialModel makeMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph, GeneratedMdlSource& result )
{
    if( material.type == "matte" )
    {
        return makeMatteMaterialModel( material, textureGraph );
    }
    if( material.type == "plastic" )
    {
        return makePlasticMaterialModel( material, textureGraph );
    }
    if( material.type == "uber" )
    {
        return makeUberMaterialModel( material, textureGraph );
    }
    return makeUnsupportedMaterialModel( material, result );
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

GeneratedMdlSource generateMdlSource( const otk::pbrt::PbrtMaterial& material )
{
    const MdlShaderKey key{ makeMdlShaderKey( material ) };
    const std::string  suffix{ stableHash( key.signature ) };

    GeneratedMdlSource result;
    result.moduleName   = "::otk::demand_pbrt_scene::pbrt_" + suffix;
    result.materialName = "material_" + suffix;

    MdlTextureGraphGenerator textureGraph{ material.graph, result };
    const MdlMaterialModel   materialModel{ makeMaterialModel( material, textureGraph, result ) };

    std::ostringstream source;
    source << "mdl 1.6;\n"
           << "import ::df::*;\n"
           << "\n";
    for( std::vector<std::string>::const_iterator it = materialModel.comments.begin(); it != materialModel.comments.end(); ++it )
    {
        source << "// " << *it << "\n";
    }
    if( !materialModel.comments.empty() )
    {
        source << "\n";
    }
    for( std::vector<std::string>::const_iterator it = result.unsupportedReasons.begin();
         it != result.unsupportedReasons.end(); ++it )
    {
        source << "// unsupported: " << *it << "\n";
    }
    if( !result.unsupportedReasons.empty() )
    {
        source << "\n";
    }
    source << textureGraph.sourcePreamble() << materialModel.helperDefinitions << textureGraph.functionDefinitions()
           << "export material " << result.materialName << mdlParameterList( materialModel.parameters ) << " = material(\n"
           << materialModel.body << ");\n";
    result.source = source.str();
    return result;
}

MdlShaderCompileRecord& MdlShaderCompileCache::getMutableRecord( const MdlShaderKey& key )
{
    std::map<MdlShaderKey, MdlShaderCompileRecord>::iterator it = m_records.find( key );
    if( it == m_records.end() )
    {
        MdlShaderCompileRecord record{};
        record.shaderKeyId = m_nextShaderKeyId++;
        it                 = m_records.insert( std::make_pair( key, record ) ).first;
    }
    return it->second;
}

const MdlShaderCompileRecord& MdlShaderCompileCache::getRecord( const MdlShaderKey& key )
{
    ++m_stats.numShaderRequests;
    if( m_records.find( key ) != m_records.end() )
    {
        ++m_stats.numShaderCacheHits;
    }
    return getMutableRecord( key );
}

MdlShaderCompileState MdlShaderCompileCache::state( const MdlShaderKey& key ) const
{
    std::map<MdlShaderKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    return it == m_records.end() ? MdlShaderCompileState::MISSING : it->second.state;
}

unsigned int MdlShaderCompileCache::shaderKeyId( const MdlShaderKey& key ) const
{
    std::map<MdlShaderKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    return it == m_records.end() ? 0U : it->second.shaderKeyId;
}

std::string MdlShaderCompileCache::diagnostics( const MdlShaderKey& key ) const
{
    std::map<MdlShaderKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    return it == m_records.end() ? std::string{} : it->second.diagnostics;
}

bool MdlShaderCompileCache::requestCompile( const MdlShaderKey& key )
{
    std::map<MdlShaderKey, MdlShaderCompileRecord>::iterator it = m_records.find( key );
    if( it != m_records.end() && it->second.state != MdlShaderCompileState::MISSING )
    {
        ++m_stats.numShaderCacheHits;
        return false;
    }

    MdlShaderCompileRecord& record = it == m_records.end() ? getMutableRecord( key ) : it->second;
    if( record.state != MdlShaderCompileState::MISSING )
    {
        return false;
    }

    record.state = MdlShaderCompileState::QUEUED;
    record.diagnostics.clear();
    ++m_stats.numCompileRequests;
    return true;
}

void MdlShaderCompileCache::markCompiling( const MdlShaderKey& key )
{
    MdlShaderCompileRecord& record = getMutableRecord( key );
    record.state                   = MdlShaderCompileState::COMPILING;
    record.diagnostics.clear();
}

void MdlShaderCompileCache::markReady( const MdlShaderKey& key )
{
    MdlShaderCompileRecord& record = getMutableRecord( key );
    if( record.state != MdlShaderCompileState::READY )
    {
        ++m_stats.numCompletedCompiles;
    }
    record.state = MdlShaderCompileState::READY;
    record.diagnostics.clear();
}

void MdlShaderCompileCache::markFailed( const MdlShaderKey& key, const std::string& diagnostics )
{
    MdlShaderCompileRecord& record = getMutableRecord( key );
    record.state                   = MdlShaderCompileState::FAILED;
    record.diagnostics             = diagnostics;
}

MdlShaderCompileCacheStatistics MdlShaderCompileCache::getStatistics() const
{
    MdlShaderCompileCacheStatistics stats{ m_stats };
    for( std::map<MdlShaderKey, MdlShaderCompileRecord>::const_iterator it = m_records.begin(); it != m_records.end(); ++it )
    {
        switch( it->second.state )
        {
            case MdlShaderCompileState::MISSING:
                ++stats.numMissingShaders;
                break;
            case MdlShaderCompileState::QUEUED:
                ++stats.numQueuedShaders;
                break;
            case MdlShaderCompileState::COMPILING:
                ++stats.numCompilingShaders;
                break;
            case MdlShaderCompileState::READY:
                ++stats.numReadyShaders;
                break;
            case MdlShaderCompileState::FAILED:
                ++stats.numFailedShaders;
                break;
        }
    }
    return stats;
}

std::size_t MdlShaderCompileCache::size() const
{
    return m_records.size();
}

void MdlShaderCompileCache::clear()
{
    m_records.clear();
    m_stats           = MdlShaderCompileCacheStatistics{};
    m_nextShaderKeyId = 1U;
}

const GeneratedMdlSource& MdlGeneratedSourceCache::getSource( const MdlShaderKey& key )
{
    std::map<MdlShaderKey, GeneratedMdlSource>::iterator it = m_sources.find( key );
    if( it == m_sources.end() )
    {
        it = m_sources.insert( std::make_pair( key, generateSource( key ) ) ).first;
    }
    return it->second;
}

const GeneratedMdlSource& MdlGeneratedSourceCache::getSource( const otk::pbrt::PbrtMaterial& material )
{
    const MdlShaderKey                                   key{ makeMdlShaderKey( material ) };
    std::map<MdlShaderKey, GeneratedMdlSource>::iterator it = m_sources.find( key );
    if( it == m_sources.end() )
    {
        it = m_sources.insert( std::make_pair( key, generateMdlSource( material ) ) ).first;
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
