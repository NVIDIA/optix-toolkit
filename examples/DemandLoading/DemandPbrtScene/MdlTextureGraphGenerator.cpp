// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlTextureGraphGenerator.h"

#ifdef OTK_USE_MDL

#include "DemandPbrtScene/MdlMaterialModelBuilder.h"
#include "DemandPbrtScene/MdlShaderCache.h"
#include "DemandPbrtScene/PbrtCheckerboardImageSource.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace demandPbrtScene {

MdlTextureLookup findMdlTexture( const otk::pbrt::PbrtMaterialGraph& graph, const std::string& textureName, const std::string& preferredValueType )
{
    MdlTextureLookup fallback{ std::string{}, nullptr };
    for( otk::pbrt::PbrtTextureMap::const_iterator it = graph.textures.begin(); it != graph.textures.end(); ++it )
    {
        if( it->second.name != textureName )
        {
            continue;
        }
        if( preferredValueType.empty() || it->second.valueType == preferredValueType )
        {
            return MdlTextureLookup{ it->first, &it->second };
        }
        if( fallback.texture == nullptr )
        {
            fallback = MdlTextureLookup{ it->first, &it->second };
        }
    }
    return fallback;
}

namespace {

std::string textureKind( const otk::pbrt::PbrtTexture& texture )
{
    return texture.valueType + ":" + texture.type;
}

bool isUnsupportedProceduralTexture( const otk::pbrt::PbrtTexture& texture )
{
    return texture.type == "marble" || texture.type == "fbm" || texture.type == "windy" || texture.type == "wrinkled";
}

}  // namespace

class MdlTextureGraphGenerator::Impl
{
  public:
    Impl( const otk::pbrt::PbrtMaterialGraph& graph, GeneratedMdlSource& result )
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
        const MdlTextureLookup lookup{ findMdlTexture( m_graph, textureName, preferredValueType ) };
        if( lookup.texture == nullptr )
        {
            appendUnsupportedReason( m_result, "Missing PBRT texture '" + textureName + "'" );
            return unsupportedTextureExpression();
        }
        if( std::find( m_textureStack.begin(), m_textureStack.end(), lookup.graphKey ) != m_textureStack.end() )
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
        const MdlTextureLookup lookup{ findMdlTexture( m_graph, textureName, preferredValueType ) };
        if( lookup.texture == nullptr
            || std::find( textureStack.begin(), textureStack.end(), lookup.graphKey ) != textureStack.end() )
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

MdlTextureGraphGenerator::MdlTextureGraphGenerator( const otk::pbrt::PbrtMaterialGraph& graph,
                                                    GeneratedMdlSource&                 result )
    : m_impl{ std::make_unique<Impl>( graph, result ) }
{
}

MdlTextureGraphGenerator::~MdlTextureGraphGenerator() = default;

std::string MdlTextureGraphGenerator::materialColorExpression( const ::pbrt::ParamSet& params,
                                                               const std::string&      paramName,
                                                               const std::string&      preferredValueType,
                                                               const std::string&      defaultExpression )
{
    return m_impl->materialColorExpression( params, paramName, preferredValueType, defaultExpression );
}

std::string MdlTextureGraphGenerator::materialFloatExpression( const ::pbrt::ParamSet& params,
                                                               const std::string&      paramName,
                                                               const std::string&      preferredValueType,
                                                               const std::string&      defaultExpression )
{
    return m_impl->materialFloatExpression( params, paramName, preferredValueType, defaultExpression );
}

std::string MdlTextureGraphGenerator::sourcePreamble() const
{
    return m_impl->sourcePreamble();
}

std::string MdlTextureGraphGenerator::functionDefinitions() const
{
    return m_impl->functionDefinitions();
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL

