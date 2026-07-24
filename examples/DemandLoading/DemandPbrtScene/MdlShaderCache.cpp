// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlShaderCache.h"

#ifdef OTK_USE_MDL
#include "DemandPbrtScene/PbrtCheckerboardImageSource.h"

#include <algorithm>
#include <cmath>
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
        "amount",
        "bumpmap",
        "eta",
        "index",
        "k",
        "opacity",
        "reflect",
        "roughness",
        "shadowalpha",
        "sigma",
        "transmit",
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

std::string paramSetToString( const ::pbrt::ParamSet& params )
{
    ::pbrt::ParamSet copy{ params };
    return copy.ToString();
}

void appendTextureInstanceSignature( std::ostringstream&                 out,
                                     const std::string&                  graphKey,
                                     const otk::pbrt::PbrtTexture&       texture,
                                     const otk::pbrt::PbrtMaterialGraph& graph,
                                     std::vector<std::string>&           textureStack );

void appendTextureInstanceReference( std::ostringstream&                 out,
                                     const std::string&                  paramName,
                                     const std::string&                  textureName,
                                     const otk::pbrt::PbrtMaterialGraph& graph,
                                     std::vector<std::string>&           textureStack )
{
    out << "|texture-ref(" << paramName << ")=" << textureName << ':';
    bool found{ false };
    for( otk::pbrt::PbrtTextureMap::const_iterator it = graph.textures.begin(); it != graph.textures.end(); ++it )
    {
        if( it->second.name == textureName )
        {
            if( found )
                out << ",";
            appendTextureInstanceSignature( out, it->first, it->second, graph, textureStack );
            found = true;
        }
    }
    if( !found )
    {
        out << "missing";
    }
}

void appendTextureInstanceReferences( std::ostringstream&                 out,
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
            appendTextureInstanceReference( out, *it, textureName, graph, textureStack );
        }
    }
}

void appendMaterialInstanceSignature( std::ostringstream&                 out,
                                      const std::string&                  type,
                                      const ::pbrt::ParamSet&             params,
                                      const otk::pbrt::PbrtMaterialGraph& graph,
                                      std::vector<std::string>&           materialStack,
                                      std::vector<std::string>&           textureStack );

void appendMaterialInstanceReference( std::ostringstream&                 out,
                                      const std::string&                  paramName,
                                      const std::string&                  materialName,
                                      const otk::pbrt::PbrtMaterialGraph& graph,
                                      std::vector<std::string>&           materialStack,
                                      std::vector<std::string>&           textureStack )
{
    out << "|material-ref(" << paramName << ")=" << materialName << ':';
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
    appendMaterialInstanceSignature( out, material->second.type, material->second.params, graph, materialStack, textureStack );
    materialStack.pop_back();
}

void appendMaterialInstanceReferences( std::ostringstream&                 out,
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
            appendMaterialInstanceReference( out, *it, materialName, graph, materialStack, textureStack );
        }
    }
}

void appendTextureInstanceSignature( std::ostringstream&                 out,
                                     const std::string&                  graphKey,
                                     const otk::pbrt::PbrtTexture&       texture,
                                     const otk::pbrt::PbrtMaterialGraph& graph,
                                     std::vector<std::string>&           textureStack )
{
    if( contains( textureStack, graphKey ) )
    {
        out << "texture(key=" << graphKey << ",name=" << texture.name << ",kind=" << texture.valueType << ':'
            << texture.type << ";recursive)";
        return;
    }

    textureStack.push_back( graphKey );
    out << "texture(key=" << graphKey << ",name=" << texture.name << ",kind=" << texture.valueType << ':'
        << texture.type << "|params=" << paramSetToString( texture.params );
    appendTextureInstanceReferences( out, texture.params, textureTextureParamNames(), graph, textureStack );
    out << ")";
    textureStack.pop_back();
}

void appendMaterialInstanceSignature( std::ostringstream&                 out,
                                      const std::string&                  type,
                                      const ::pbrt::ParamSet&             params,
                                      const otk::pbrt::PbrtMaterialGraph& graph,
                                      std::vector<std::string>&           materialStack,
                                      std::vector<std::string>&           textureStack )
{
    out << "material(" << type << "|params=" << paramSetToString( params );
    appendTextureInstanceReferences( out, params, materialTextureParamNames(), graph, textureStack );
    appendMaterialInstanceReferences( out, params, graph, materialStack, textureStack );
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
        if( isFoldableTextureReference( textureName, preferredValueType ) )
        {
            return defaultExpression;
        }
        return textureReference( textureName, preferredValueType );
    }

    std::string materialFloatExpression( const ::pbrt::ParamSet& params,
                                         const std::string&      paramName,
                                         const std::string&      preferredValueType,
                                         const std::string&      defaultExpression )
    {
        const std::string textureName{ params.FindTexture( paramName ) };
        if( textureName.empty() )
        {
            return defaultExpression;
        }
        if( isFoldableTextureReference( textureName, preferredValueType ) && defaultExpression != "0.0" )
        {
            return defaultExpression;
        }
        m_usesTextureFloat = true;
        return "pbrt_texture_float(" + textureReference( textureName, preferredValueType ) + ")";
    }

    std::string sourcePreamble() const
    {
        std::ostringstream out;
        if( m_usesTextureFloat )
        {
            out << "float pbrt_texture_float(color value) = ::math::luminance(value);\n";
        }
        if( m_usesDemandTexture )
        {
            out << "color pbrt_demand_texture_2d(int texture_id) = color(1.0, 1.0, 1.0);\n";
        }
        if( m_usesCheckerboard )
        {
            out << "color pbrt_checkerboard_2d(color tex1, color tex2) = (tex1 + tex2) * 0.5;\n";
        }
        if( m_usesUnsupported )
        {
            out << "color pbrt_unsupported_texture() = color(1.0, 0.0, 1.0);\n";
        }
        if( !m_usesTextureFloat && !m_usesDemandTexture && !m_usesCheckerboard && !m_usesUnsupported )
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
        if( isDemandTexture( texture ) )
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
            return demandTextureExpression();
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
            if( !pbrtCheckerboardTextureKey( texture ).empty() )
            {
                return demandTextureExpression();
            }
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

    std::string demandTextureExpression()
    {
        m_usesDemandTexture = true;
        return "pbrt_demand_texture_2d(" + std::to_string( m_nextImageParameter++ ) + ")";
    }

    static bool isDemandTexture( const otk::pbrt::PbrtTexture& texture )
    {
        return texture.type == "imagemap" || ( texture.type == "checkerboard" && !pbrtCheckerboardTextureKey( texture ).empty() );
    }

    bool isFoldableTextureReference( const std::string& textureName, const std::string& preferredValueType ) const
    {
        std::vector<std::string> textureStack;
        return isFoldableTextureReference( textureName, preferredValueType, textureStack );
    }

    bool isFoldableTextureReference( const std::string&        textureName,
                                     const std::string&        preferredValueType,
                                     std::vector<std::string>& textureStack ) const
    {
        const TextureLookup lookup{ findTexture( m_graph, textureName, preferredValueType ) };
        if( lookup.texture == nullptr || contains( textureStack, lookup.graphKey ) )
        {
            return false;
        }

        textureStack.push_back( lookup.graphKey );
        const bool result{ isFoldableTexture( *lookup.texture, textureStack ) };
        textureStack.pop_back();
        return result;
    }

    bool isFoldableTextureInput( const otk::pbrt::PbrtTexture& texture, const char* paramName, std::vector<std::string>& textureStack ) const
    {
        const std::string textureName{ texture.params.FindTexture( paramName ) };
        return textureName.empty() || isFoldableTextureReference( textureName, texture.valueType, textureStack );
    }

    bool isFoldableTexture( const otk::pbrt::PbrtTexture& texture, std::vector<std::string>& textureStack ) const
    {
        if( texture.type == "constant" )
        {
            return true;
        }
        if( texture.type == "scale" )
        {
            return isFoldableTextureInput( texture, "tex1", textureStack ) && isFoldableTextureInput( texture, "tex2", textureStack );
        }
        if( texture.type == "mix" )
        {
            return isFoldableTextureInput( texture, "tex1", textureStack ) && isFoldableTextureInput( texture, "tex2", textureStack )
                   && isFoldableTextureInput( texture, "amount", textureStack );
        }
        return false;
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
    bool                                m_usesTextureFloat{};
    bool                                m_usesDemandTexture{};
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

std::string namedMaterialParameterName( unsigned int index, const std::string& paramName );

struct PbrtMaterialGapPolicy
{
    std::string type;
    std::string policy;
    std::string coverageReason;
};

struct BoundParameterSpec
{
    MdlBoundParameterType type;
    const char*           name;
};

struct FoldedColor
{
    float red{};
    float green{};
    float blue{};
};

FoldedColor operator*( const FoldedColor& lhs, const FoldedColor& rhs )
{
    return FoldedColor{ lhs.red * rhs.red, lhs.green * rhs.green, lhs.blue * rhs.blue };
}

FoldedColor mix( const FoldedColor& lhs, const FoldedColor& rhs, float amount )
{
    return FoldedColor{ lhs.red * ( 1.0f - amount ) + rhs.red * amount, lhs.green * ( 1.0f - amount ) + rhs.green * amount,
                        lhs.blue * ( 1.0f - amount ) + rhs.blue * amount };
}

bool findConstantColor( const ::pbrt::ParamSet& params, const char* name, float& red, float& green, float& blue )
{
    if( !params.FindTexture( name ).empty() )
    {
        return false;
    }

    int                     count{};
    const ::pbrt::Spectrum* values = params.FindSpectrum( name, &count );
    if( count <= 0 || values == nullptr )
    {
        return false;
    }

    float rgb[3]{};
    values[0].ToRGB( rgb );
    red   = rgb[0];
    green = rgb[1];
    blue  = rgb[2];
    return true;
}

bool findConstantFloat( const ::pbrt::ParamSet& params, const char* name, float& value )
{
    if( !params.FindTexture( name ).empty() )
    {
        return false;
    }

    int          count{};
    const float* values = params.FindFloat( name, &count );
    if( count <= 0 || values == nullptr )
    {
        return false;
    }

    value = values[0];
    return true;
}

bool scalarColorValue( const FoldedColor& color, float& value )
{
    constexpr float epsilon{ 1.0e-6f };
    if( std::fabs( color.red - color.green ) > epsilon || std::fabs( color.red - color.blue ) > epsilon )
    {
        return false;
    }
    value = color.red;
    return true;
}

bool findConstantScalarColor( const ::pbrt::ParamSet& params, const char* name, float& value )
{
    FoldedColor color{};
    return findConstantColor( params, name, color.red, color.green, color.blue ) && scalarColorValue( color, value );
}

bool findTextureColorValue( const ::pbrt::ParamSet& params, const char* name, const FoldedColor& defaultValue, FoldedColor& value )
{
    if( findConstantColor( params, name, value.red, value.green, value.blue ) )
    {
        return true;
    }

    float floatValue{};
    if( findConstantFloat( params, name, floatValue ) )
    {
        value = FoldedColor{ floatValue, floatValue, floatValue };
        return true;
    }

    value = defaultValue;
    return true;
}

bool findTextureFloatValue( const ::pbrt::ParamSet& params, const char* name, float defaultValue, float& value )
{
    if( findConstantFloat( params, name, value ) )
    {
        return true;
    }

    FoldedColor color{};
    if( findConstantColor( params, name, color.red, color.green, color.blue ) )
    {
        return scalarColorValue( color, value );
    }

    value = defaultValue;
    return true;
}

bool findFoldableTextureColor( const otk::pbrt::PbrtMaterialGraph& graph,
                               const std::string&                  textureName,
                               const std::string&                  preferredValueType,
                               std::vector<std::string>&           textureStack,
                               FoldedColor&                        value );

bool findFoldableTextureFloat( const otk::pbrt::PbrtMaterialGraph& graph,
                               const std::string&                  textureName,
                               std::vector<std::string>&           textureStack,
                               float&                              value );

bool findTextureInputColor( const otk::pbrt::PbrtMaterialGraph& graph,
                            const otk::pbrt::PbrtTexture&       texture,
                            const char*                         name,
                            const FoldedColor&                  defaultValue,
                            std::vector<std::string>&           textureStack,
                            FoldedColor&                        value )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( !inputTextureName.empty() )
    {
        return findFoldableTextureColor( graph, inputTextureName, texture.valueType, textureStack, value );
    }
    return findTextureColorValue( texture.params, name, defaultValue, value );
}

bool findTextureInputFloat( const otk::pbrt::PbrtMaterialGraph& graph,
                            const otk::pbrt::PbrtTexture&       texture,
                            const char*                         name,
                            float                               defaultValue,
                            std::vector<std::string>&           textureStack,
                            float&                              value )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( !inputTextureName.empty() )
    {
        return findFoldableTextureFloat( graph, inputTextureName, textureStack, value );
    }
    return findTextureFloatValue( texture.params, name, defaultValue, value );
}

bool findFoldableTextureColor( const otk::pbrt::PbrtMaterialGraph& graph,
                               const std::string&                  textureName,
                               const std::string&                  preferredValueType,
                               std::vector<std::string>&           textureStack,
                               FoldedColor&                        value )
{
    const TextureLookup lookup{ findTexture( graph, textureName, preferredValueType ) };
    if( lookup.texture == nullptr || contains( textureStack, lookup.graphKey ) )
    {
        return false;
    }

    textureStack.push_back( lookup.graphKey );
    const otk::pbrt::PbrtTexture& texture{ *lookup.texture };
    bool                          folded{ false };
    if( texture.type == "constant" )
    {
        folded = findTextureColorValue( texture.params, "value", FoldedColor{ 1.0f, 1.0f, 1.0f }, value );
    }
    else if( texture.type == "scale" )
    {
        FoldedColor tex1{};
        FoldedColor tex2{};
        folded = findTextureInputColor( graph, texture, "tex1", FoldedColor{ 1.0f, 1.0f, 1.0f }, textureStack, tex1 )
                 && findTextureInputColor( graph, texture, "tex2", FoldedColor{ 1.0f, 1.0f, 1.0f }, textureStack, tex2 );
        if( folded )
        {
            value = tex1 * tex2;
        }
    }
    else if( texture.type == "mix" )
    {
        FoldedColor tex1{};
        FoldedColor tex2{};
        float       amount{};
        folded = findTextureInputColor( graph, texture, "tex1", FoldedColor{ 1.0f, 1.0f, 1.0f }, textureStack, tex1 )
                 && findTextureInputColor( graph, texture, "tex2", FoldedColor{ 1.0f, 1.0f, 1.0f }, textureStack, tex2 )
                 && findTextureInputFloat( graph, texture, "amount", 0.5f, textureStack, amount );
        if( folded )
        {
            value = mix( tex1, tex2, amount );
        }
    }

    textureStack.pop_back();
    return folded;
}

bool findFoldableTextureFloat( const otk::pbrt::PbrtMaterialGraph& graph,
                               const std::string&                  textureName,
                               std::vector<std::string>&           textureStack,
                               float&                              value )
{
    const TextureLookup lookup{ findTexture( graph, textureName, "float" ) };
    if( lookup.texture == nullptr || contains( textureStack, lookup.graphKey ) )
    {
        return false;
    }

    textureStack.push_back( lookup.graphKey );
    const otk::pbrt::PbrtTexture& texture{ *lookup.texture };
    bool                          folded{ false };
    if( texture.type == "constant" )
    {
        folded = findTextureFloatValue( texture.params, "value", 1.0f, value );
    }
    else if( texture.type == "scale" )
    {
        float tex1{};
        float tex2{};
        folded = findTextureInputFloat( graph, texture, "tex1", 1.0f, textureStack, tex1 )
                 && findTextureInputFloat( graph, texture, "tex2", 1.0f, textureStack, tex2 );
        if( folded )
        {
            value = tex1 * tex2;
        }
    }
    else if( texture.type == "mix" )
    {
        float tex1{};
        float tex2{};
        float amount{};
        folded = findTextureInputFloat( graph, texture, "tex1", 1.0f, textureStack, tex1 )
                 && findTextureInputFloat( graph, texture, "tex2", 1.0f, textureStack, tex2 )
                 && findTextureInputFloat( graph, texture, "amount", 0.5f, textureStack, amount );
        if( folded )
        {
            value = tex1 * ( 1.0f - amount ) + tex2 * amount;
        }
    }

    textureStack.pop_back();
    return folded;
}

bool findFoldableTextureColor( const otk::pbrt::PbrtMaterialGraph& graph,
                               const std::string&                  textureName,
                               const std::string&                  preferredValueType,
                               FoldedColor&                        value )
{
    std::vector<std::string> textureStack;
    return findFoldableTextureColor( graph, textureName, preferredValueType, textureStack, value );
}

bool findFoldableTextureFloat( const otk::pbrt::PbrtMaterialGraph& graph, const std::string& textureName, float& value )
{
    std::vector<std::string> textureStack;
    return findFoldableTextureFloat( graph, textureName, textureStack, value );
}

void appendBoundParameter( std::vector<MdlBoundMaterialParameter>& result, const ::pbrt::ParamSet& params, const BoundParameterSpec& spec )
{
    MdlBoundMaterialParameter parameter{};
    parameter.name = spec.name;
    parameter.type = spec.type;
    if( spec.type == MdlBoundParameterType::COLOR )
    {
        if( findConstantColor( params, spec.name, parameter.red, parameter.green, parameter.blue ) )
        {
            result.push_back( parameter );
        }
        return;
    }

    if( findConstantFloat( params, spec.name, parameter.value ) )
    {
        result.push_back( parameter );
        return;
    }
    if( std::string{ spec.name } == "amount" && findConstantScalarColor( params, spec.name, parameter.value ) )
    {
        result.push_back( parameter );
    }
}

void appendTextureBackedBoundParameter( std::vector<MdlBoundMaterialParameter>& result,
                                        const otk::pbrt::PbrtMaterial&          material,
                                        const BoundParameterSpec&               spec )
{
    const std::string textureName{ material.params.FindTexture( spec.name ) };
    if( textureName.empty() )
    {
        return;
    }

    MdlBoundMaterialParameter parameter{};
    parameter.name = spec.name;
    parameter.type = spec.type;
    if( spec.type == MdlBoundParameterType::COLOR )
    {
        FoldedColor value{};
        if( findFoldableTextureColor( material.graph, textureName, "color", value ) )
        {
            parameter.red   = value.red;
            parameter.green = value.green;
            parameter.blue  = value.blue;
            result.push_back( parameter );
        }
        return;
    }

    if( findFoldableTextureFloat( material.graph, textureName, parameter.value ) )
    {
        result.push_back( parameter );
    }
}

void appendBoundParameters( std::vector<MdlBoundMaterialParameter>& result,
                            const ::pbrt::ParamSet&                 params,
                            const BoundParameterSpec*               begin,
                            const BoundParameterSpec*               end )
{
    for( const BoundParameterSpec* it = begin; it != end; ++it )
    {
        appendBoundParameter( result, params, *it );
    }
}

void appendMaterialBoundParameters( std::vector<MdlBoundMaterialParameter>& result,
                                    const otk::pbrt::PbrtMaterial&          material,
                                    const BoundParameterSpec*               begin,
                                    const BoundParameterSpec*               end )
{
    for( const BoundParameterSpec* it = begin; it != end; ++it )
    {
        appendBoundParameter( result, material.params, *it );
        appendTextureBackedBoundParameter( result, material, *it );
    }
}

void appendNamedBoundParameter( std::vector<MdlBoundMaterialParameter>& result,
                                const ::pbrt::ParamSet&                 params,
                                unsigned int                            index,
                                const BoundParameterSpec&               spec )
{
    MdlBoundMaterialParameter parameter{};
    parameter.name = namedMaterialParameterName( index, spec.name );
    parameter.type = spec.type;
    if( spec.type == MdlBoundParameterType::COLOR )
    {
        if( findConstantColor( params, spec.name, parameter.red, parameter.green, parameter.blue ) )
        {
            result.push_back( parameter );
        }
        return;
    }

    if( findConstantFloat( params, spec.name, parameter.value ) )
    {
        result.push_back( parameter );
    }
}

void appendNamedBoundParameters( std::vector<MdlBoundMaterialParameter>& result,
                                 const ::pbrt::ParamSet&                 params,
                                 unsigned int                            index,
                                 const BoundParameterSpec*               begin,
                                 const BoundParameterSpec*               end )
{
    for( const BoundParameterSpec* it = begin; it != end; ++it )
    {
        appendNamedBoundParameter( result, params, index, *it );
    }
}

void appendMaterialParameter( MdlMaterialModel& model, const std::string& type, const std::string& name, const std::string& defaultValue )
{
    model.parameters.push_back( MdlMaterialParameter{ type, name, defaultValue } );
}

const PbrtMaterialGapPolicy* explicitMaterialGapPolicy( const std::string& type )
{
    static const PbrtMaterialGapPolicy policies[] = {
        { "fourier", "unsupported with visible fallback",
          "PBRT Fourier tables are data-driven BSDF resources found in the corpus; DemandPbrtScene preserves the "
          "resource metadata but does not yet evaluate the Fourier table on the GPU" },
        { "hair", "unsupported with visible fallback",
          "low-frequency PBRT corpus material; no current target scene or reference fixture requires approximation" },
        { "subsurface", "unsupported with visible fallback",
          "low-frequency PBRT corpus material; no current target scene or reference fixture requires approximation or "
          "baking" },
        { "kdsubsurface", "unsupported with visible fallback",
          "distinct low-frequency subsurface parameterization; no current target scene or reference fixture requires "
          "support" },
        { "measured", "unsupported with visible fallback",
          "PBRT parity completeness gap; current corpus sample did not find a target scene requiring support" },
    };

    for( const PbrtMaterialGapPolicy& policy : policies )
    {
        if( policy.type == type )
        {
            return &policy;
        }
    }
    return nullptr;
}

bool hasFourierBsdfFile( const otk::pbrt::PbrtMaterial& material )
{
    return !material.params.FindOneString( "bsdffile", std::string{} ).empty();
}

void appendRoughnessGapComment( MdlMaterialModel& model )
{
    model.comments.push_back( "pbrt material gap: PBRT-exact roughness/remapping behavior is approximated" );
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

std::string materialBumpmapExpression( MdlTextureGraphGenerator& textureGraph, const ::pbrt::ParamSet& params )
{
    if( params.FindTexture( "bumpmap" ).empty() )
    {
        return "none";
    }
    return textureGraph.materialFloatExpression( params, "bumpmap", "float", "0.0" );
}

bool hasBumpmapExpression( const std::string& bumpmap )
{
    return bumpmap != "none";
}

void appendBumpmapCommentsAndHelpers( MdlMaterialModel& model, const std::string& bumpmap )
{
    model.comments.push_back( "pbrt material input bumpmap: " + bumpmap );
    if( !hasBumpmapExpression( bumpmap ) )
    {
        return;
    }

    model.comments.push_back( "pbrt material implementation: bumpmap is evaluated with runtime finite differences" );
}

std::string materialGeometryExpression( const std::string& cutoutOpacity, const std::string& bumpmap )
{
    (void)bumpmap;
    if( cutoutOpacity.empty() )
    {
        return std::string{};
    }

    std::ostringstream out;
    out << "    geometry: material_geometry(\n";
    out << "        cutout_opacity: " << cutoutOpacity << "\n";
    out << "    )\n";
    return out.str();
}

std::string namedMaterialParameterName( unsigned int index, const std::string& paramName )
{
    return "named_" + std::to_string( index ) + "_" + paramName;
}

std::string namedMaterialColorExpression( MdlMaterialModel&                   model,
                                          MdlTextureGraphGenerator&           textureGraph,
                                          const otk::pbrt::PbrtNamedMaterial& material,
                                          unsigned int                        index,
                                          const std::string&                  paramName,
                                          const std::string&                  defaultValue )
{
    const std::string parameterName{ namedMaterialParameterName( index, paramName ) };
    appendMaterialParameter( model, "color", parameterName, defaultValue );
    return textureGraph.materialColorExpression( material.params, paramName, "color", parameterName );
}

std::string namedMaterialFloatExpression( MdlMaterialModel&                   model,
                                          const otk::pbrt::PbrtNamedMaterial& material,
                                          unsigned int                        index,
                                          const std::string&                  paramName,
                                          const std::string&                  defaultValue )
{
    const std::string parameterName{ namedMaterialParameterName( index, paramName ) };
    appendMaterialParameter( model, "float", parameterName, defaultValue );
    if( !material.params.FindTexture( paramName ).empty() )
    {
        return defaultValue;
    }
    return parameterName;
}

std::string namedMaterialType( const otk::pbrt::PbrtNamedMaterial& material )
{
    if( !material.type.empty() )
    {
        return material.type;
    }
    return material.params.FindOneString( "type", std::string{} );
}

std::string namedMaterialMatteBsdfExpression( MdlMaterialModel&                   model,
                                              MdlTextureGraphGenerator&           textureGraph,
                                              const otk::pbrt::PbrtNamedMaterial& material,
                                              unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string sigma{ namedMaterialFloatExpression( model, material, index, "sigma", "0.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input sigma: " + sigma );
    return "::df::diffuse_reflection_bsdf(\n"
           "                        tint: "
           + kd
           + ",\n"
             "                        roughness: pbrt_mix_matte_sigma_roughness("
           + sigma + "))";
}

std::string namedMaterialPlasticBsdfExpression( MdlMaterialModel&                   model,
                                                MdlTextureGraphGenerator&           textureGraph,
                                                const otk::pbrt::PbrtNamedMaterial& material,
                                                unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.0, 0.0, 0.0)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    return "::df::color_normalized_mix(\n"
           "                        components: ::df::color_bsdf_component[](\n"
           "                            ::df::color_bsdf_component(\n"
           "                                weight: "
           + kd
           + ",\n"
             "                                component: ::df::diffuse_reflection_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + ks
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: "
           + roughness
           + ",\n"
             "                                    roughness_v: "
           + roughness
           + ",\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect))))";
}

std::string namedMaterialSubstrateBsdfExpression( MdlMaterialModel&                   model,
                                                  MdlTextureGraphGenerator&           textureGraph,
                                                  const otk::pbrt::PbrtNamedMaterial& material,
                                                  unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.5, 0.5, 0.5)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.5, 0.5, 0.5)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    const std::string uroughness{ namedMaterialFloatExpression( model, material, index, "uroughness", "-1.0" ) };
    const std::string vroughness{ namedMaterialFloatExpression( model, material, index, "vroughness", "-1.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input uroughness: " + uroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input vroughness: " + vroughness );
    return "::df::color_weighted_layer(\n"
           "                        weight: "
           + ks
           + ",\n"
             "                        layer: ::df::simple_glossy_bsdf(\n"
             "                            roughness_u: pbrt_mix_resolved_roughness("
           + roughness + ", " + uroughness
           + "),\n"
             "                            roughness_v: pbrt_mix_resolved_roughness("
           + roughness + ", " + vroughness
           + "),\n"
             "                            tint: color(1.0, 1.0, 1.0),\n"
             "                            mode: ::df::scatter_reflect),\n"
             "                        base: ::df::diffuse_reflection_bsdf(\n"
             "                            tint: "
           + kd + "))";
}

std::string namedMaterialUberBsdfExpression( MdlMaterialModel&                   model,
                                             MdlTextureGraphGenerator&           textureGraph,
                                             const otk::pbrt::PbrtNamedMaterial& material,
                                             unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.0, 0.0, 0.0)" ) };
    const std::string kr{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kr", "color(0.0, 0.0, 0.0)" ) };
    const std::string kt{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kt", "color(0.0, 0.0, 0.0)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    const std::string uroughness{ namedMaterialFloatExpression( model, material, index, "uroughness", "-1.0" ) };
    const std::string vroughness{ namedMaterialFloatExpression( model, material, index, "vroughness", "-1.0" ) };
    const std::string alpha{ namedMaterialFloatExpression( model, material, index, "alpha", "1.0" ) };
    const std::string opacity{ namedMaterialFloatExpression( model, material, index, "opacity", "1.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kr: " + kr );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kt: " + kt );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input uroughness: " + uroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input vroughness: " + vroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input opacity: " + opacity );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input alpha: " + alpha
                              + "; cutout does not compose through mix" );
    return std::string{ "::df::color_normalized_mix(\n"
                        "                        components: ::df::color_bsdf_component[](\n"
                        "                            ::df::color_bsdf_component(\n"
                        "                                weight: " }
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + kd
           + ",\n"
             "                                component: ::df::diffuse_reflection_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + ks
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: pbrt_mix_resolved_roughness("
           + roughness + ", " + uroughness
           + "),\n"
             "                                    roughness_v: pbrt_mix_resolved_roughness("
           + roughness + ", " + vroughness
           + "),\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + kr
           + ",\n"
             "                                component: ::df::specular_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + kt
           + ",\n"
             "                                component: ::df::specular_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_transmit)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: pbrt_mix_transparency_weight("
           + opacity
           + "),\n"
             "                                component: ::df::specular_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_transmit))))";
}

std::string namedMaterialMirrorBsdfExpression( MdlMaterialModel&                   model,
                                               MdlTextureGraphGenerator&           textureGraph,
                                               const otk::pbrt::PbrtNamedMaterial& material,
                                               unsigned int                        index )
{
    const std::string kr{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kr", "color(1.0, 1.0, 1.0)" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kr: " + kr );
    return "::df::specular_bsdf(\n"
           "                        tint: "
           + kr
           + ",\n"
             "                        mode: ::df::scatter_reflect)";
}

std::string namedMaterialGlassBsdfExpression( MdlMaterialModel&                   model,
                                              MdlTextureGraphGenerator&           textureGraph,
                                              const otk::pbrt::PbrtNamedMaterial& material,
                                              unsigned int                        index )
{
    const std::string kr{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kr", "color(1.0, 1.0, 1.0)" ) };
    const std::string kt{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kt", "color(1.0, 1.0, 1.0)" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kr: " + kr );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kt: " + kt );
    return "::df::tint(\n"
           "                        "
           + kr
           + ",\n"
             "                        "
           + kt
           + ",\n"
             "                        ::df::specular_bsdf(\n"
             "                            tint: color(1.0, 1.0, 1.0),\n"
             "                            mode: ::df::scatter_reflect_transmit))";
}

std::string namedMaterialMetalBsdfExpression( MdlMaterialModel&                   model,
                                              MdlTextureGraphGenerator&           textureGraph,
                                              const otk::pbrt::PbrtNamedMaterial& material,
                                              unsigned int                        index )
{
    const std::string eta{
        namedMaterialColorExpression( model, textureGraph, material, index, "eta", "color(0.2, 0.2, 0.2)" ) };
    const std::string k{
        namedMaterialColorExpression( model, textureGraph, material, index, "k", "color(3.0, 3.0, 3.0)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    const std::string uroughness{ namedMaterialFloatExpression( model, material, index, "uroughness", "-1.0" ) };
    const std::string vroughness{ namedMaterialFloatExpression( model, material, index, "vroughness", "-1.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input eta: " + eta );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input k: " + k );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input uroughness: " + uroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input vroughness: " + vroughness );
    return "::df::microfacet_ggx_smith_bsdf(\n"
           "                        roughness_u: pbrt_mix_resolved_roughness("
           + roughness + ", " + uroughness
           + "),\n"
             "                        roughness_v: pbrt_mix_resolved_roughness("
           + roughness + ", " + vroughness
           + "),\n"
             "                        tint: pbrt_mix_metal_conductor_tint("
           + eta + ", " + k
           + "),\n"
             "                        mode: ::df::scatter_reflect)";
}

std::string namedMaterialTranslucentBsdfExpression( MdlMaterialModel&                   model,
                                                    MdlTextureGraphGenerator&           textureGraph,
                                                    const otk::pbrt::PbrtNamedMaterial& material,
                                                    unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.0, 0.0, 0.0)" ) };
    const std::string reflect{
        namedMaterialColorExpression( model, textureGraph, material, index, "reflect", "color(0.5, 0.5, 0.5)" ) };
    const std::string transmit{
        namedMaterialColorExpression( model, textureGraph, material, index, "transmit", "color(0.5, 0.5, 0.5)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input reflect: " + reflect );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input transmit: " + transmit );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    return "::df::color_normalized_mix(\n"
           "                        components: ::df::color_bsdf_component[](\n"
           "                            ::df::color_bsdf_component(\n"
           "                                weight: "
           + kd + " * " + reflect
           + ",\n"
             "                                component: ::df::diffuse_reflection_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + kd + " * " + transmit
           + ",\n"
             "                                component: ::df::diffuse_transmission_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + ks + " * " + reflect
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: "
           + roughness
           + ",\n"
             "                                    roughness_v: "
           + roughness
           + ",\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + ks + " * " + transmit
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: "
           + roughness
           + ",\n"
             "                                    roughness_v: "
           + roughness
           + ",\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_transmit))))";
}

std::string unsupportedNamedMaterialBsdfExpression()
{
    return "::df::diffuse_reflection_bsdf(\n"
           "                        tint: color(1.0, 0.0, 1.0))";
}

std::string namedMaterialBsdfExpression( MdlMaterialModel&                   model,
                                         MdlTextureGraphGenerator&           textureGraph,
                                         GeneratedMdlSource&                 result,
                                         const otk::pbrt::PbrtNamedMaterial& material,
                                         unsigned int                        index )
{
    const std::string type{ namedMaterialType( material ) };
    const std::string typeComment{ type.empty() ? std::string{ "<empty>" } : type };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " model: " + typeComment );

    if( type == "matte" )
    {
        return namedMaterialMatteBsdfExpression( model, textureGraph, material, index );
    }
    if( type == "plastic" )
    {
        return namedMaterialPlasticBsdfExpression( model, textureGraph, material, index );
    }
    if( type == "substrate" )
    {
        return namedMaterialSubstrateBsdfExpression( model, textureGraph, material, index );
    }
    if( type == "uber" )
    {
        return namedMaterialUberBsdfExpression( model, textureGraph, material, index );
    }
    if( type == "mirror" )
    {
        return namedMaterialMirrorBsdfExpression( model, textureGraph, material, index );
    }
    if( type == "glass" )
    {
        return namedMaterialGlassBsdfExpression( model, textureGraph, material, index );
    }
    if( type == "metal" )
    {
        return namedMaterialMetalBsdfExpression( model, textureGraph, material, index );
    }
    if( type == "translucent" )
    {
        return namedMaterialTranslucentBsdfExpression( model, textureGraph, material, index );
    }

    appendUnsupportedReason( result, "Unsupported PBRT named material type " + typeComment );
    return unsupportedNamedMaterialBsdfExpression();
}

MdlMaterialModel makeMatteMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "float", "sigma", "0.0" );
    appendMaterialParameter( model, "float", "alpha", "1.0" );
    appendMaterialParameter( model, "float", "opacity", "1.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string alphaTexture{
        materialTextureCommentExpression( textureGraph, material.params, "alpha", "float" ) };
    const std::string shadowAlphaTexture{
        materialTextureCommentExpression( textureGraph, material.params, "shadowalpha", "float" ) };
    const std::string opacityTexture{
        materialTextureCommentExpression( textureGraph, material.params, "opacity", "float" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: matte" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input sigma: sigma" );
    model.comments.push_back( "pbrt material approximation: sigma degrees map to MDL Oren-Nayar roughness sigma / 90" );
    model.comments.push_back( "pbrt material input alpha: alpha; texture=" + alphaTexture );
    model.comments.push_back( "pbrt material input shadowalpha: any-hit texture=" + shadowAlphaTexture );
    model.comments.push_back( "pbrt material input opacity: opacity; texture=" + opacityTexture );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    model.helperDefinitions =
        "float pbrt_matte_sigma_roughness(float sigma_degrees) = ::math::clamp(sigma_degrees / 90.0, 0.0, 1.0);\n\n"
        + model.helperDefinitions;
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::diffuse_reflection_bsdf(\n"
        "            tint: "
        + kd + ",\n"
               "            roughness: pbrt_matte_sigma_roughness(sigma))),\n"
        + materialGeometryExpression( "alpha * opacity", bumpmap );
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
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: plastic" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: diffuse and glossy reflection use an MDL color-normalized mix" );
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kd + ",\n"
        "                    component: ::df::diffuse_reflection_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + ks + ",\n"
        "                    component: ::df::simple_glossy_bsdf(\n"
        "                        roughness_u: roughness,\n"
        "                        roughness_v: roughness,\n"
        "                        tint: color(1.0, 1.0, 1.0),\n"
        "                        mode: ::df::scatter_reflect)))))"
        + "\n";
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
    appendMaterialParameter( model, "float", "uroughness", "-1.0" );
    appendMaterialParameter( model, "float", "vroughness", "-1.0" );
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
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: uber" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.comments.push_back( "pbrt material input Kt: " + kt );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    model.comments.push_back( "pbrt material input index: index" );
    model.comments.push_back( "pbrt material input alpha: alpha; texture=" + alphaTexture );
    model.comments.push_back( "pbrt material input opacity: opacity; texture=" + opacityTexture );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    appendRoughnessGapComment( model );
    model.comments.push_back( "pbrt material approximation: PBRT uber lobes use an MDL color-normalized mix" );
    model.comments.push_back(
        "pbrt material approximation: opacity weights BSDF lobes and adds transparent transmission; alpha remains "
        "cutout" );
    model.helperDefinitions =
        "float pbrt_uber_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        "float pbrt_uber_clamped_opacity(float opacity) = ::math::clamp(opacity, 0.0, 1.0);\n\n"
        "color pbrt_uber_opacity_weight(float opacity) =\n"
        "    color(pbrt_uber_clamped_opacity(opacity), pbrt_uber_clamped_opacity(opacity), "
        "pbrt_uber_clamped_opacity(opacity));\n\n"
        "color pbrt_uber_transparency_weight(float opacity) =\n"
        "    color(1.0 - pbrt_uber_clamped_opacity(opacity), 1.0 - pbrt_uber_clamped_opacity(opacity), "
        "1.0 - pbrt_uber_clamped_opacity(opacity));\n\n"
        + model.helperDefinitions;
    model.body = std::string{ "    ior: color(index, index, index),\n"
                              "    surface: material_surface(\n"
                              "        scattering: ::df::color_normalized_mix(\n"
                              "            components: ::df::color_bsdf_component[](\n"
                              "                ::df::color_bsdf_component(\n"
                              "                    weight: " }
                 + "pbrt_uber_opacity_weight(opacity) * " + kd
                 + ",\n"
                   "                    component: ::df::diffuse_reflection_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0))),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: "
                 + "pbrt_uber_opacity_weight(opacity) * " + ks
                 + ",\n"
                   "                    component: ::df::simple_glossy_bsdf(\n"
                   "                        roughness_u: pbrt_uber_resolved_roughness(roughness, uroughness),\n"
                   "                        roughness_v: pbrt_uber_resolved_roughness(roughness, vroughness),\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_reflect)),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: pbrt_uber_opacity_weight(opacity) * "
                 + kr
                 + ",\n"
                   "                    component: ::df::specular_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_reflect)),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: pbrt_uber_opacity_weight(opacity) * "
                 + kt
                 + ",\n"
                   "                    component: ::df::specular_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_transmit)),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: pbrt_uber_transparency_weight(opacity),\n"
                   "                    component: ::df::specular_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_transmit))))),\n"
        + materialGeometryExpression( "alpha", bumpmap );
    return model;
}

MdlMaterialModel makeMirrorMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kr", "color(1.0, 1.0, 1.0)" );

    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };

    model.comments.push_back( "pbrt material model: mirror" );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::specular_bsdf(\n"
        "            tint: " + kr + ",\n"
        "            mode: ::df::scatter_reflect))\n";
    return model;
}

MdlMaterialModel makeGlassMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kr", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "color", "Kt", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "float", "index", "1.5" );
    appendMaterialParameter( model, "float", "roughness", "0.0" );
    appendMaterialParameter( model, "float", "uroughness", "0.0" );
    appendMaterialParameter( model, "float", "vroughness", "0.0" );

    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };
    const std::string kt{ textureGraph.materialColorExpression( material.params, "Kt", "color", "Kt" ) };

    model.comments.push_back( "pbrt material model: glass" );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.comments.push_back( "pbrt material input Kt: " + kt );
    model.comments.push_back( "pbrt material input index/eta: index" );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    model.comments.push_back(
        "pbrt material gap: rough glass microfacet behavior is not implemented; roughness inputs "
        "are bound but the MDL glass lobe is specular" );
    model.body =
        "    ior: color(index, index, index),\n"
        "    surface: material_surface(\n"
        "        scattering: ::df::tint(\n"
        "            "
        + kr + ",\n"
        "            " + kt
        + ",\n"
          "            ::df::specular_bsdf(\n"
          "                tint: color(1.0, 1.0, 1.0),\n"
          "                mode: ::df::scatter_reflect_transmit)))\n";
    return model;
}

MdlMaterialModel makeMetalMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "eta", "color(0.2, 0.2, 0.2)" );
    appendMaterialParameter( model, "color", "k", "color(3.0, 3.0, 3.0)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "float", "uroughness", "-1.0" );
    appendMaterialParameter( model, "float", "vroughness", "-1.0" );

    const std::string eta{ textureGraph.materialColorExpression( material.params, "eta", "color", "eta" ) };
    const std::string k{ textureGraph.materialColorExpression( material.params, "k", "color", "k" ) };

    model.comments.push_back( "pbrt material model: metal" );
    model.comments.push_back( "pbrt material input eta: " + eta );
    model.comments.push_back( "pbrt material input k: " + k );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    model.comments.push_back( "pbrt material gap: PBRT-exact spectral conductor behavior is approximated" );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: RGB eta/k maps to MDL microfacet tint using normal-incidence conductor "
        "reflectance" );
    model.helperDefinitions =
        "float pbrt_metal_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        "color pbrt_metal_conductor_tint(color eta, color k) =\n"
        "    ((eta - color(1.0, 1.0, 1.0)) * (eta - color(1.0, 1.0, 1.0)) + k * k) /\n"
        "    ((eta + color(1.0, 1.0, 1.0)) * (eta + color(1.0, 1.0, 1.0)) + k * k);\n\n";
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::microfacet_ggx_smith_bsdf(\n"
        "            roughness_u: pbrt_metal_resolved_roughness(roughness, uroughness),\n"
        "            roughness_v: pbrt_metal_resolved_roughness(roughness, vroughness),\n"
        "            tint: pbrt_metal_conductor_tint("
        + eta + ", " + k
        + "),\n"
          "            mode: ::df::scatter_reflect))\n";
    return model;
}

MdlMaterialModel makeSubstrateMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "float", "uroughness", "-1.0" );
    appendMaterialParameter( model, "float", "vroughness", "-1.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: substrate" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: diffuse base and glossy layer use an MDL color-weighted layer" );
    model.helperDefinitions =
        "float pbrt_substrate_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        + model.helperDefinitions;
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::color_weighted_layer(\n"
        "            weight: " + ks + ",\n"
        "            layer: ::df::simple_glossy_bsdf(\n"
        "                roughness_u: pbrt_substrate_resolved_roughness(roughness, uroughness),\n"
        "                roughness_v: pbrt_substrate_resolved_roughness(roughness, vroughness),\n"
        "                tint: color(1.0, 1.0, 1.0),\n"
        "                mode: ::df::scatter_reflect),\n"
        "            base: ::df::diffuse_reflection_bsdf(\n"
        "                tint: "
        + kd + ")))"
        + "\n";
    return model;
}

MdlMaterialModel makeTranslucentMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "color", "reflect", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "color", "transmit", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "float", "opacity", "1.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string reflect{ textureGraph.materialColorExpression( material.params, "reflect", "color", "reflect" ) };
    const std::string transmit{
        textureGraph.materialColorExpression( material.params, "transmit", "color", "transmit" ) };
    const std::string opacityTexture{
        materialTextureCommentExpression( textureGraph, material.params, "opacity", "float" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: translucent" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input reflect: " + reflect );
    model.comments.push_back( "pbrt material input transmit: " + transmit );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input opacity: opacity; texture=" + opacityTexture );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    model.comments.push_back( "pbrt material input eta: fixed 1.5" );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: diffuse/glossy reflection and transmission use an MDL color-normalized mix" );
    model.body =
        "    ior: color(1.5, 1.5, 1.5),\n"
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kd + " * " + reflect + ",\n"
        "                    component: ::df::diffuse_reflection_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kd + " * " + transmit + ",\n"
        "                    component: ::df::diffuse_transmission_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + ks + " * " + reflect + ",\n"
        "                    component: ::df::simple_glossy_bsdf(\n"
        "                        roughness_u: roughness,\n"
        "                        roughness_v: roughness,\n"
        "                        tint: color(1.0, 1.0, 1.0),\n"
        "                        mode: ::df::scatter_reflect)),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + ks + " * " + transmit + ",\n"
        "                    component: ::df::simple_glossy_bsdf(\n"
        "                        roughness_u: roughness,\n"
        "                        roughness_v: roughness,\n"
        "                        tint: color(1.0, 1.0, 1.0),\n"
        "                        mode: ::df::scatter_transmit))))),\n"
        + materialGeometryExpression( "opacity", bumpmap );
    return model;
}

std::string mixNamedMaterialBsdfExpression( MdlMaterialModel&              model,
                                            MdlTextureGraphGenerator&      textureGraph,
                                            GeneratedMdlSource&            result,
                                            const otk::pbrt::PbrtMaterial& material,
                                            const std::string&             paramName,
                                            unsigned int                   index )
{
    const std::string materialName{ material.params.FindOneString( paramName, std::string{} ) };
    if( materialName.empty() )
    {
        model.comments.push_back( "pbrt material input " + paramName + ": missing" );
        appendUnsupportedReason( result, "Missing PBRT mix " + paramName );
        return unsupportedNamedMaterialBsdfExpression();
    }

    model.comments.push_back( "pbrt material input " + paramName + ": named material " + std::to_string( index ) );
    const otk::pbrt::PbrtNamedMaterialMap::const_iterator namedMaterial = material.graph.namedMaterials.find( materialName );
    if( namedMaterial == material.graph.namedMaterials.end() )
    {
        appendUnsupportedReason( result, "Missing PBRT named material reference" );
        return unsupportedNamedMaterialBsdfExpression();
    }

    return namedMaterialBsdfExpression( model, textureGraph, result, namedMaterial->second, index );
}

MdlMaterialModel makeMixMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph, GeneratedMdlSource& result )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "float", "amount", "0.5" );

    model.comments.push_back( "pbrt material model: mix" );
    const std::string first{ mixNamedMaterialBsdfExpression( model, textureGraph, result, material, "namedmaterial1", 0U ) };
    const std::string second{ mixNamedMaterialBsdfExpression( model, textureGraph, result, material, "namedmaterial2", 1U ) };
    const std::string amountTexture{
        materialTextureCommentExpression( textureGraph, material.params, "amount", "float" ) };

    model.comments.push_back( "pbrt material input amount: amount; texture=" + amountTexture );
    model.comments.push_back( "pbrt material weighting: namedmaterial1 uses amount; namedmaterial2 uses 1 - amount" );
    model.comments.push_back( "pbrt material approximation: mix composes supported named material MDL closures" );
    model.helperDefinitions =
        "float pbrt_mix_matte_sigma_roughness(float sigma_degrees) = ::math::clamp(sigma_degrees / 90.0, 0.0, 1.0);\n\n"
        "float pbrt_mix_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        "float pbrt_mix_clamped_opacity(float opacity) = ::math::clamp(opacity, 0.0, 1.0);\n\n"
        "color pbrt_mix_opacity_weight(float opacity) =\n"
        "    color(pbrt_mix_clamped_opacity(opacity), pbrt_mix_clamped_opacity(opacity), "
        "pbrt_mix_clamped_opacity(opacity));\n\n"
        "color pbrt_mix_transparency_weight(float opacity) =\n"
        "    color(1.0 - pbrt_mix_clamped_opacity(opacity), 1.0 - pbrt_mix_clamped_opacity(opacity), "
        "1.0 - pbrt_mix_clamped_opacity(opacity));\n\n"
        "color pbrt_mix_metal_conductor_tint(color eta, color k) =\n"
        "    ((eta - color(1.0, 1.0, 1.0)) * (eta - color(1.0, 1.0, 1.0)) + k * k) /\n"
        "    ((eta + color(1.0, 1.0, 1.0)) * (eta + color(1.0, 1.0, 1.0)) + k * k);\n\n";
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: color(amount, amount, amount),\n"
        "                    component: "
        + first
        + "),\n"
          "                ::df::color_bsdf_component(\n"
          "                    weight: color(1.0 - amount, 1.0 - amount, 1.0 - amount),\n"
          "                    component: "
        + second + "))))\n";
    return model;
}

MdlMaterialModel makeUnsupportedMaterialModel( const otk::pbrt::PbrtMaterial& material, GeneratedMdlSource& result )
{
    const std::string type{ material.type.empty() ? std::string{ "<empty>" } : material.type };

    MdlMaterialModel model;
    model.comments.push_back( "pbrt material model: " + type );
    const PbrtMaterialGapPolicy* const policy{ explicitMaterialGapPolicy( material.type ) };
    if( policy != nullptr )
    {
        model.comments.push_back( "pbrt material gap policy: " + policy->policy );
        model.comments.push_back( "pbrt material gap coverage: " + policy->coverageReason );
        appendUnsupportedReason( result, "Explicit PBRT material gap " + type + ": " + policy->policy );
        if( material.type == "fourier" )
        {
            if( hasFourierBsdfFile( material ) )
            {
                model.comments.push_back( "pbrt fourier bsdffile: preserved as material metadata" );
            }
            else
            {
                model.comments.push_back( "pbrt fourier bsdffile: missing" );
                appendUnsupportedReason( result, "PBRT Fourier material missing bsdffile" );
            }
        }
    }
    else
    {
        model.comments.push_back( "pbrt material gap policy: unknown material type" );
        appendUnsupportedReason( result, "Unsupported PBRT material type " + type );
    }
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
    if( material.type == "mirror" )
    {
        return makeMirrorMaterialModel( material, textureGraph );
    }
    if( material.type == "glass" )
    {
        return makeGlassMaterialModel( material, textureGraph );
    }
    if( material.type == "metal" )
    {
        return makeMetalMaterialModel( material, textureGraph );
    }
    if( material.type == "substrate" )
    {
        return makeSubstrateMaterialModel( material, textureGraph );
    }
    if( material.type == "translucent" )
    {
        return makeTranslucentMaterialModel( material, textureGraph );
    }
    if( material.type == "mix" )
    {
        return makeMixMaterialModel( material, textureGraph, result );
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
    source << "mdl 1.10;\n"
           << "import ::df::*;\n"
           << "import ::math::*;\n"
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

bool operator==( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs )
{
    return lhs.sourceKey == rhs.sourceKey && lhs.signature == rhs.signature;
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
    return lhs.signature < rhs.signature;
}

std::string toString( const MdlMaterialInstanceKey& key )
{
    return "source=" + toString( key.sourceKey ) + "|instance=" + key.signature;
}

MdlMaterialInstanceKey makeMdlMaterialInstanceKey( const otk::pbrt::PbrtMaterial& material )
{
    MdlMaterialInstanceKey result;
    result.sourceKey = makeMdlShaderKey( material );

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

std::vector<MdlBoundMaterialParameter> makeMdlBoundMaterialParameters( const otk::pbrt::PbrtMaterial& material )
{
    static const BoundParameterSpec matteParams[] = {
        { MdlBoundParameterType::COLOR, "Kd" },
        { MdlBoundParameterType::FLOAT, "sigma" },
        { MdlBoundParameterType::FLOAT, "alpha" },
        { MdlBoundParameterType::FLOAT, "opacity" },
    };
    static const BoundParameterSpec plasticParams[] = {
        { MdlBoundParameterType::COLOR, "Kd" },
        { MdlBoundParameterType::COLOR, "Ks" },
        { MdlBoundParameterType::FLOAT, "roughness" },
    };
    static const BoundParameterSpec uberParams[] = {
        { MdlBoundParameterType::COLOR, "Kd" },         { MdlBoundParameterType::COLOR, "Ks" },
        { MdlBoundParameterType::COLOR, "Kr" },         { MdlBoundParameterType::COLOR, "Kt" },
        { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
        { MdlBoundParameterType::FLOAT, "vroughness" }, { MdlBoundParameterType::FLOAT, "index" },
        { MdlBoundParameterType::FLOAT, "alpha" },      { MdlBoundParameterType::FLOAT, "opacity" },
    };
    static const BoundParameterSpec namedUberParams[] = {
        { MdlBoundParameterType::COLOR, "Kd" },         { MdlBoundParameterType::COLOR, "Ks" },
        { MdlBoundParameterType::COLOR, "Kr" },         { MdlBoundParameterType::COLOR, "Kt" },
        { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
        { MdlBoundParameterType::FLOAT, "vroughness" }, { MdlBoundParameterType::FLOAT, "alpha" },
        { MdlBoundParameterType::FLOAT, "opacity" },
    };
    static const BoundParameterSpec mirrorParams[] = {
        { MdlBoundParameterType::COLOR, "Kr" },
    };
    static const BoundParameterSpec glassParams[] = {
        { MdlBoundParameterType::COLOR, "Kr" },         { MdlBoundParameterType::COLOR, "Kt" },
        { MdlBoundParameterType::FLOAT, "index" },      { MdlBoundParameterType::FLOAT, "roughness" },
        { MdlBoundParameterType::FLOAT, "uroughness" }, { MdlBoundParameterType::FLOAT, "vroughness" },
    };
    static const BoundParameterSpec metalParams[] = {
        { MdlBoundParameterType::COLOR, "eta" },        { MdlBoundParameterType::COLOR, "k" },
        { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
        { MdlBoundParameterType::FLOAT, "vroughness" },
    };
    static const BoundParameterSpec substrateParams[] = {
        { MdlBoundParameterType::COLOR, "Kd" },         { MdlBoundParameterType::COLOR, "Ks" },
        { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
        { MdlBoundParameterType::FLOAT, "vroughness" },
    };
    static const BoundParameterSpec translucentParams[] = {
        { MdlBoundParameterType::COLOR, "Kd" },        { MdlBoundParameterType::COLOR, "Ks" },
        { MdlBoundParameterType::COLOR, "reflect" },   { MdlBoundParameterType::COLOR, "transmit" },
        { MdlBoundParameterType::FLOAT, "roughness" }, { MdlBoundParameterType::FLOAT, "opacity" },
    };
    static const BoundParameterSpec mixParams[] = {
        { MdlBoundParameterType::FLOAT, "amount" },
    };

    std::vector<MdlBoundMaterialParameter> result;
    const auto appendNamedMaterialParameters = [&]( const std::string& paramName, unsigned int index ) {
        const std::string materialName{ material.params.FindOneString( paramName, std::string{} ) };
        if( materialName.empty() )
        {
            return;
        }

        const otk::pbrt::PbrtNamedMaterialMap::const_iterator namedMaterial = material.graph.namedMaterials.find( materialName );
        if( namedMaterial == material.graph.namedMaterials.end() )
        {
            return;
        }

        const std::string type{ namedMaterialType( namedMaterial->second ) };
        if( type == "matte" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( matteParams ),
                                        std::end( matteParams ) );
        }
        else if( type == "plastic" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( plasticParams ),
                                        std::end( plasticParams ) );
        }
        else if( type == "uber" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( namedUberParams ),
                                        std::end( namedUberParams ) );
        }
        else if( type == "mirror" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( mirrorParams ),
                                        std::end( mirrorParams ) );
        }
        else if( type == "glass" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( glassParams ),
                                        std::end( glassParams ) );
        }
        else if( type == "metal" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( metalParams ),
                                        std::end( metalParams ) );
        }
        else if( type == "substrate" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( substrateParams ),
                                        std::end( substrateParams ) );
        }
        else if( type == "translucent" )
        {
            appendNamedBoundParameters( result, namedMaterial->second.params, index, std::begin( translucentParams ),
                                        std::end( translucentParams ) );
        }
    };

    if( material.type == "matte" )
    {
        appendMaterialBoundParameters( result, material, std::begin( matteParams ), std::end( matteParams ) );
    }
    else if( material.type == "plastic" )
    {
        appendMaterialBoundParameters( result, material, std::begin( plasticParams ), std::end( plasticParams ) );
    }
    else if( material.type == "uber" )
    {
        appendMaterialBoundParameters( result, material, std::begin( uberParams ), std::end( uberParams ) );
    }
    else if( material.type == "mirror" )
    {
        appendMaterialBoundParameters( result, material, std::begin( mirrorParams ), std::end( mirrorParams ) );
    }
    else if( material.type == "glass" )
    {
        appendMaterialBoundParameters( result, material, std::begin( glassParams ), std::end( glassParams ) );
    }
    else if( material.type == "metal" )
    {
        appendMaterialBoundParameters( result, material, std::begin( metalParams ), std::end( metalParams ) );
    }
    else if( material.type == "substrate" )
    {
        appendMaterialBoundParameters( result, material, std::begin( substrateParams ), std::end( substrateParams ) );
    }
    else if( material.type == "translucent" )
    {
        appendMaterialBoundParameters( result, material, std::begin( translucentParams ), std::end( translucentParams ) );
    }
    else if( material.type == "mix" )
    {
        appendMaterialBoundParameters( result, material, std::begin( mixParams ), std::end( mixParams ) );
        appendNamedMaterialParameters( "namedmaterial1", 0U );
        appendNamedMaterialParameters( "namedmaterial2", 1U );
    }
    return result;
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
    source << "mdl 1.10;\n"
           << "import ::df::*;\n"
           << "import ::math::*;\n"
           << "import ::state::*;\n"
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

MdlShaderCompileRecord& MdlShaderCompileCache::getMutableRecord( const MdlMaterialInstanceKey& key )
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::iterator it = m_records.find( key );
    if( it == m_records.end() )
    {
        MdlShaderCompileRecord record{};
        record.sourceKey   = key.sourceKey;
        record.shaderKeyId = m_nextShaderKeyId++;

        std::map<MdlShaderKey, unsigned int>::iterator source = m_sourceKeyUseCounts.find( key.sourceKey );
        if( source == m_sourceKeyUseCounts.end() )
        {
            m_sourceKeyUseCounts.insert( std::make_pair( key.sourceKey, 1U ) );
        }
        else
        {
            ++source->second;
            ++m_stats.numSourceCacheHits;
            ++m_stats.numShaderCacheHits;
        }

        it = m_records.insert( std::make_pair( key, record ) ).first;
    }
    return it->second;
}

const MdlShaderCompileRecord& MdlShaderCompileCache::getRecord( const MdlMaterialInstanceKey& key )
{
    ++m_stats.numShaderRequests;
    if( m_records.find( key ) != m_records.end() )
    {
        ++m_stats.numMaterialInstanceCacheHits;
        ++m_stats.numShaderCacheHits;
    }
    return getMutableRecord( key );
}

MdlShaderCompileState MdlShaderCompileCache::state( const MdlMaterialInstanceKey& key ) const
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    return it == m_records.end() ? MdlShaderCompileState::MISSING : it->second.state;
}

unsigned int MdlShaderCompileCache::shaderKeyId( const MdlMaterialInstanceKey& key ) const
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    return it == m_records.end() ? 0U : it->second.shaderKeyId;
}

std::string MdlShaderCompileCache::diagnostics( const MdlMaterialInstanceKey& key ) const
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    return it == m_records.end() ? std::string{} : it->second.diagnostics;
}

bool MdlShaderCompileCache::requestCompile( const MdlMaterialInstanceKey& key )
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::iterator it = m_records.find( key );
    if( it != m_records.end() && it->second.state != MdlShaderCompileState::MISSING )
    {
        ++m_stats.numMaterialInstanceCacheHits;
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

void MdlShaderCompileCache::markCompiling( const MdlMaterialInstanceKey& key )
{
    MdlShaderCompileRecord& record = getMutableRecord( key );
    record.state                   = MdlShaderCompileState::COMPILING;
    record.diagnostics.clear();
}

void MdlShaderCompileCache::markReady( const MdlMaterialInstanceKey& key )
{
    MdlShaderCompileRecord& record = getMutableRecord( key );
    if( record.state != MdlShaderCompileState::READY )
    {
        ++m_stats.numCompletedCompiles;
    }
    record.state = MdlShaderCompileState::READY;
    record.diagnostics.clear();
}

void MdlShaderCompileCache::markFailed( const MdlMaterialInstanceKey& key, const std::string& diagnostics )
{
    MdlShaderCompileRecord& record = getMutableRecord( key );
    record.state                   = MdlShaderCompileState::FAILED;
    record.diagnostics             = diagnostics;
}

MdlShaderCompileCacheStatistics MdlShaderCompileCache::getStatistics() const
{
    MdlShaderCompileCacheStatistics stats{ m_stats };
    for( std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.begin();
         it != m_records.end(); ++it )
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
    m_sourceKeyUseCounts.clear();
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
