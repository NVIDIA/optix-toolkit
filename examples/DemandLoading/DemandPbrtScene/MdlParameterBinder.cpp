// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlParameterBinder.h"

#ifdef OTK_USE_MDL

#include "DemandPbrtScene/MdlTextureGraphGenerator.h"
#include "DemandPbrtScene/PbrtMaterialKind.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace demandPbrtScene {

std::string namedMaterialParameterName( unsigned int index, const std::string& paramName )
{
    return "named_" + std::to_string( index ) + "_" + paramName;
}

std::string namedMaterialType( const otk::pbrt::PbrtNamedMaterial& material )
{
    if( !material.type.empty() )
    {
        return material.type;
    }
    return material.params.FindOneString( "type", std::string{} );
}

namespace {

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

bool promotesFloatToColorParameter( const char* name )
{
    const std::string parameterName{ name };
    return parameterName == "opacity" || parameterName == "amount";
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
    const MdlTextureLookup lookup{ findMdlTexture( graph, textureName, preferredValueType ) };
    if( lookup.texture == nullptr || std::find( textureStack.begin(), textureStack.end(), lookup.graphKey ) != textureStack.end() )
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
    const MdlTextureLookup lookup{ findMdlTexture( graph, textureName, "float" ) };
    if( lookup.texture == nullptr || std::find( textureStack.begin(), textureStack.end(), lookup.graphKey ) != textureStack.end() )
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
        else if( promotesFloatToColorParameter( spec.name ) && findConstantFloat( params, spec.name, parameter.value ) )
        {
            parameter.red = parameter.green = parameter.blue = parameter.value;
            result.push_back( parameter );
        }
        return;
    }

    if( findConstantFloat( params, spec.name, parameter.value ) )
    {
        result.push_back( parameter );
        return;
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

constexpr BoundParameterSpec matteParams[] = {
    { MdlBoundParameterType::COLOR, "Kd" },
    { MdlBoundParameterType::FLOAT, "sigma" },
    { MdlBoundParameterType::FLOAT, "alpha" },
    { MdlBoundParameterType::FLOAT, "opacity" },
};
constexpr BoundParameterSpec plasticParams[] = {
    { MdlBoundParameterType::COLOR, "Kd" },
    { MdlBoundParameterType::COLOR, "Ks" },
    { MdlBoundParameterType::FLOAT, "roughness" },
};
constexpr BoundParameterSpec uberParams[] = {
    { MdlBoundParameterType::COLOR, "Kd" },         { MdlBoundParameterType::COLOR, "Ks" },
    { MdlBoundParameterType::COLOR, "Kr" },         { MdlBoundParameterType::COLOR, "Kt" },
    { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
    { MdlBoundParameterType::FLOAT, "vroughness" }, { MdlBoundParameterType::FLOAT, "index" },
    { MdlBoundParameterType::FLOAT, "alpha" },      { MdlBoundParameterType::COLOR, "opacity" },
};
constexpr BoundParameterSpec namedUberParams[] = {
    { MdlBoundParameterType::COLOR, "Kd" },         { MdlBoundParameterType::COLOR, "Ks" },
    { MdlBoundParameterType::COLOR, "Kr" },         { MdlBoundParameterType::COLOR, "Kt" },
    { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
    { MdlBoundParameterType::FLOAT, "vroughness" }, { MdlBoundParameterType::FLOAT, "alpha" },
    { MdlBoundParameterType::FLOAT, "opacity" },
};
constexpr BoundParameterSpec mirrorParams[] = {
    { MdlBoundParameterType::COLOR, "Kr" },
};
constexpr BoundParameterSpec glassParams[] = {
    { MdlBoundParameterType::COLOR, "Kr" },         { MdlBoundParameterType::COLOR, "Kt" },
    { MdlBoundParameterType::FLOAT, "index" },      { MdlBoundParameterType::FLOAT, "roughness" },
    { MdlBoundParameterType::FLOAT, "uroughness" }, { MdlBoundParameterType::FLOAT, "vroughness" },
};
constexpr BoundParameterSpec metalParams[] = {
    { MdlBoundParameterType::COLOR, "eta" },        { MdlBoundParameterType::COLOR, "k" },
    { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
    { MdlBoundParameterType::FLOAT, "vroughness" },
};
constexpr BoundParameterSpec substrateParams[] = {
    { MdlBoundParameterType::COLOR, "Kd" },         { MdlBoundParameterType::COLOR, "Ks" },
    { MdlBoundParameterType::FLOAT, "roughness" },  { MdlBoundParameterType::FLOAT, "uroughness" },
    { MdlBoundParameterType::FLOAT, "vroughness" },
};
constexpr BoundParameterSpec translucentParams[] = {
    { MdlBoundParameterType::COLOR, "Kd" },        { MdlBoundParameterType::COLOR, "Ks" },
    { MdlBoundParameterType::COLOR, "reflect" },   { MdlBoundParameterType::COLOR, "transmit" },
    { MdlBoundParameterType::FLOAT, "roughness" }, { MdlBoundParameterType::COLOR, "opacity" },
};
constexpr BoundParameterSpec subsurfaceParams[] = {
    { MdlBoundParameterType::COLOR, "Kr" },         { MdlBoundParameterType::COLOR, "Kt" },
    { MdlBoundParameterType::COLOR, "sigma_a" },    { MdlBoundParameterType::COLOR, "sigma_s" },
    { MdlBoundParameterType::FLOAT, "scale" },      { MdlBoundParameterType::FLOAT, "g" },
    { MdlBoundParameterType::FLOAT, "eta" },        { MdlBoundParameterType::FLOAT, "uroughness" },
    { MdlBoundParameterType::FLOAT, "vroughness" },
};
constexpr BoundParameterSpec kdSubsurfaceParams[] = {
    { MdlBoundParameterType::COLOR, "Kd" },         { MdlBoundParameterType::COLOR, "Kr" },
    { MdlBoundParameterType::COLOR, "Kt" },         { MdlBoundParameterType::COLOR, "mfp" },
    { MdlBoundParameterType::FLOAT, "scale" },      { MdlBoundParameterType::FLOAT, "g" },
    { MdlBoundParameterType::FLOAT, "eta" },        { MdlBoundParameterType::FLOAT, "uroughness" },
    { MdlBoundParameterType::FLOAT, "vroughness" },
};
constexpr BoundParameterSpec mixParams[] = {
    { MdlBoundParameterType::COLOR, "amount" },
};

struct BoundParameterSpecs
{
    const BoundParameterSpec* begin{};
    const BoundParameterSpec* end{};
};

template <std::size_t N>
constexpr BoundParameterSpecs makeBoundParameterSpecs( const BoundParameterSpec ( &specs )[N] )
{
    return { specs, specs + N };
}

enum class BoundMaterialKind
{
    ROOT,
    NAMED,
};

struct MaterialBoundParameterSpecs
{
    PbrtMaterialKind    kind;
    BoundParameterSpecs root;
    BoundParameterSpecs named;
};

constexpr MaterialBoundParameterSpecs materialBoundParameterSpecs[] = {
    { PbrtMaterialKind::MATTE, makeBoundParameterSpecs( matteParams ), makeBoundParameterSpecs( matteParams ) },
    { PbrtMaterialKind::PLASTIC, makeBoundParameterSpecs( plasticParams ), makeBoundParameterSpecs( plasticParams ) },
    { PbrtMaterialKind::UBER, makeBoundParameterSpecs( uberParams ), makeBoundParameterSpecs( namedUberParams ) },
    { PbrtMaterialKind::MIRROR, makeBoundParameterSpecs( mirrorParams ), makeBoundParameterSpecs( mirrorParams ) },
    { PbrtMaterialKind::GLASS, makeBoundParameterSpecs( glassParams ), makeBoundParameterSpecs( glassParams ) },
    { PbrtMaterialKind::METAL, makeBoundParameterSpecs( metalParams ), makeBoundParameterSpecs( metalParams ) },
    { PbrtMaterialKind::SUBSTRATE, makeBoundParameterSpecs( substrateParams ), makeBoundParameterSpecs( substrateParams ) },
    { PbrtMaterialKind::TRANSLUCENT, makeBoundParameterSpecs( translucentParams ),
      makeBoundParameterSpecs( translucentParams ) },
    { PbrtMaterialKind::SUBSURFACE, makeBoundParameterSpecs( subsurfaceParams ), {} },
    { PbrtMaterialKind::KD_SUBSURFACE, makeBoundParameterSpecs( kdSubsurfaceParams ), {} },
    { PbrtMaterialKind::MIX, makeBoundParameterSpecs( mixParams ), {} },
};

BoundParameterSpecs boundParameterSpecs( PbrtMaterialKind materialKind, BoundMaterialKind boundKind )
{
    for( const MaterialBoundParameterSpecs& specs : materialBoundParameterSpecs )
    {
        if( materialKind == specs.kind )
        {
            return boundKind == BoundMaterialKind::ROOT ? specs.root : specs.named;
        }
    }
    return {};
}

void appendRootMaterialBoundParameters( std::vector<MdlBoundMaterialParameter>& result,
                                         const otk::pbrt::PbrtMaterial&           material,
                                         PbrtMaterialKind                         kind )
{
    const BoundParameterSpecs specs{ boundParameterSpecs( kind, BoundMaterialKind::ROOT ) };
    appendMaterialBoundParameters( result, material, specs.begin, specs.end );
}

void appendNamedMaterialBoundParameters( std::vector<MdlBoundMaterialParameter>& result,
                                         const otk::pbrt::PbrtMaterial&           material,
                                         const std::string&                       paramName,
                                         unsigned int                             index )
{
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

    const PbrtMaterialKind   kind{ pbrtMaterialKind( namedMaterialType( namedMaterial->second ) ) };
    const BoundParameterSpecs specs{ boundParameterSpecs( kind, BoundMaterialKind::NAMED ) };
    appendNamedBoundParameters( result, namedMaterial->second.params, index, specs.begin, specs.end );
}

void appendNamedMaterialBoundParameters( std::vector<MdlBoundMaterialParameter>& result,
                                          const otk::pbrt::PbrtMaterial&           material,
                                          PbrtMaterialKind                         kind )
{
    if( kind != PbrtMaterialKind::MIX )
    {
        return;
    }

    appendNamedMaterialBoundParameters( result, material, "namedmaterial1", 0U );
    appendNamedMaterialBoundParameters( result, material, "namedmaterial2", 1U );
}

}  // namespace

std::vector<MdlBoundMaterialParameter> makeMdlBoundMaterialParameters( const otk::pbrt::PbrtMaterial& material )
{
    std::vector<MdlBoundMaterialParameter> result;
    const PbrtMaterialKind                 kind{ pbrtMaterialKind( material.type ) };
    appendRootMaterialBoundParameters( result, material, kind );
    appendNamedMaterialBoundParameters( result, material, kind );
    return result;
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL

