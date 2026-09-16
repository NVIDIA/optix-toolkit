// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/PbrtCheckerboardImageSource.h"

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace demandPbrtScene {

struct PbrtDemandTextureBinding
{
    std::string fileName;
    float3      scale;
    float3      bias;
    bool        transformed;
    bool        gamma;
};

inline float3 pbrtTextureScale( float value )
{
    return make_float3( value, value, value );
}

inline float3 pbrtTextureBias( float value )
{
    return make_float3( value, value, value );
}

inline ::pbrt::Point3f toPbrtPoint3f( const float3& value )
{
    return ::pbrt::Point3f{ value.x, value.y, value.z };
}

inline PbrtDemandTextureBinding pbrtDemandTextureBinding( const std::string& fileName, const float3& scale, const float3& bias, bool transformed, bool gamma )
{
    return PbrtDemandTextureBinding{ fileName, scale, bias, transformed, gamma };
}

inline PbrtDemandTextureBinding pbrtDemandTextureBinding()
{
    return pbrtDemandTextureBinding( std::string{}, pbrtTextureScale( 1.0f ), pbrtTextureBias( 0.0f ), false, false );
}

inline bool hasPbrtDemandTextureBinding( const PbrtDemandTextureBinding& binding )
{
    return !binding.fileName.empty();
}

inline MaterialFlags plasticMaterialFlags( const otk::pbrt::PlasticMaterial& material )
{
    MaterialFlags flags{};
    if( !material.alphaMapFileName.empty() )
        flags |= MaterialFlags::ALPHA_MAP;
    if( !material.diffuseMapFileName.empty() )
        flags |= MaterialFlags::DIFFUSE_MAP;
    return flags;
}

inline std::string pbrtTextureGraphKey( const std::string& valueType, const std::string& textureName )
{
    return valueType + ":" + textureName;
}

inline bool pbrtTextureStackContains( const std::vector<std::string>& textureStack, const std::string& graphKey )
{
    return std::find( textureStack.begin(), textureStack.end(), graphKey ) != textureStack.end();
}

inline const otk::pbrt::PbrtTexture* findPbrtTexture( const otk::pbrt::PbrtMaterial&     material,
                                                      const std::string&                 textureName,
                                                      std::initializer_list<const char*> valueTypes )
{
    if( textureName.empty() )
    {
        return nullptr;
    }

    for( const char* valueType : valueTypes )
    {
        const auto it = material.graph.textures.find( pbrtTextureGraphKey( valueType, textureName ) );
        if( it != material.graph.textures.end() )
        {
            return &it->second;
        }
    }
    return nullptr;
}

inline const otk::pbrt::PbrtTexture* findPbrtGraphTexture( const otk::pbrt::PbrtMaterialGraph& graph,
                                                           const std::string&                  textureName,
                                                           std::initializer_list<const char*>  valueTypes,
                                                           std::string&                        graphKey )
{
    if( textureName.empty() )
    {
        return nullptr;
    }

    for( const char* valueType : valueTypes )
    {
        graphKey      = pbrtTextureGraphKey( valueType, textureName );
        const auto it = graph.textures.find( graphKey );
        if( it != graph.textures.end() )
        {
            return &it->second;
        }
    }
    graphKey.clear();
    return nullptr;
}

inline std::string pbrtTextureMapName( const otk::pbrt::PbrtTexture* texture )
{
    if( texture == nullptr )
    {
        return {};
    }
    if( texture->type == "imagemap" )
    {
        return texture->params.FindOneFilename( "filename", "" );
    }
    if( texture->type == "checkerboard" )
    {
        return pbrtCheckerboardTextureKey( *texture );
    }
    return {};
}

inline bool pbrtTextureGamma( const otk::pbrt::PbrtTexture* texture )
{
    if( texture == nullptr || texture->type != "imagemap" )
    {
        return false;
    }
    const std::string fileName{ pbrtTextureMapName( texture ) };
    const bool defaultGamma{ ::pbrt::HasExtension( fileName, ".tga" ) || ::pbrt::HasExtension( fileName, ".png" ) };
    return texture->params.FindOneBool( "gamma", defaultGamma );
}

inline float3 multiplyPbrtTextureScale( const float3& lhs, const float3& rhs )
{
    return make_float3( lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z );
}

inline float3 addPbrtTextureColor( const float3& lhs, const float3& rhs )
{
    return make_float3( lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z );
}

inline float3 scalePbrtTextureColor( const float3& value, float scale )
{
    return make_float3( value.x * scale, value.y * scale, value.z * scale );
}

inline bool findPbrtConstantColor( const ::pbrt::ParamSet& params, const char* name, float3& value )
{
    if( !params.FindTexture( name ).empty() )
    {
        return false;
    }

    int                     count{};
    const ::pbrt::Spectrum* values{ params.FindSpectrum( name, &count ) };
    if( count <= 0 || values == nullptr )
    {
        return false;
    }

    float rgb[3]{};
    values[0].ToRGB( rgb );
    value = make_float3( rgb[0], rgb[1], rgb[2] );
    return true;
}

inline bool findPbrtConstantFloat( const ::pbrt::ParamSet& params, const char* name, float& value )
{
    if( !params.FindTexture( name ).empty() )
    {
        return false;
    }

    int          count{};
    const float* values{ params.FindFloat( name, &count ) };
    if( count <= 0 || values == nullptr )
    {
        return false;
    }

    value = values[0];
    return true;
}

inline bool findPbrtScalar( const ::pbrt::ParamSet& params, const char* name, float& value )
{
    if( findPbrtConstantFloat( params, name, value ) )
    {
        return true;
    }

    float3 color{};
    if( findPbrtConstantColor( params, name, color ) )
    {
        value = ( color.x + color.y + color.z ) / 3.0f;
        return true;
    }

    return false;
}

struct PbrtColorTextureTraits
{
    using Value = float3;

    static Value one() { return pbrtTextureScale( 1.0f ); }
    static bool  foldsMix() { return false; }

    static const otk::pbrt::PbrtTexture* findTexture( const otk::pbrt::PbrtMaterialGraph& graph,
                                                      const std::string&                  textureName,
                                                      std::string&                        graphKey )
    {
        return findPbrtGraphTexture( graph, textureName, { "spectrum", "color" }, graphKey );
    }

    static bool findConstant( const ::pbrt::ParamSet& params,
                              const char*             name,
                              const Value&            defaultValue,
                              Value&                  value )
    {
        if( findPbrtConstantColor( params, name, value ) )
        {
            return true;
        }

        float scalar{};
        if( findPbrtConstantFloat( params, name, scalar ) )
        {
            value = pbrtTextureScale( scalar );
            return true;
        }

        value = defaultValue;
        return true;
    }

    static Value multiply( const Value& lhs, const Value& rhs ) { return multiplyPbrtTextureScale( lhs, rhs ); }

    static Value mix( const Value& lhs, const Value& rhs, float amount )
    {
        return addPbrtTextureColor( scalePbrtTextureColor( lhs, 1.0f - amount ),
                                    scalePbrtTextureColor( rhs, amount ) );
    }

    static float3 color( const Value& value ) { return value; }
};

struct PbrtFloatTextureTraits
{
    using Value = float;

    static Value one() { return 1.0f; }
    static bool  foldsMix() { return true; }

    static const otk::pbrt::PbrtTexture* findTexture( const otk::pbrt::PbrtMaterialGraph& graph,
                                                      const std::string&                  textureName,
                                                      std::string&                        graphKey )
    {
        return findPbrtGraphTexture( graph, textureName, { "float" }, graphKey );
    }

    static bool findConstant( const ::pbrt::ParamSet& params,
                              const char*             name,
                              Value                   defaultValue,
                              Value&                  value )
    {
        if( findPbrtScalar( params, name, value ) )
        {
            return true;
        }
        value = defaultValue;
        return true;
    }

    static Value multiply( Value lhs, Value rhs ) { return lhs * rhs; }
    static Value mix( Value lhs, Value rhs, float amount ) { return lhs * ( 1.0f - amount ) + rhs * amount; }
    static float3 color( Value value ) { return pbrtTextureScale( value ); }
};

template <typename Traits>
inline bool foldPbrtTextureConstant( const otk::pbrt::PbrtMaterialGraph& graph,
                                     const std::string&                  textureName,
                                     std::vector<std::string>&           textureStack,
                                     typename Traits::Value&             value );

template <typename Traits>
inline bool foldPbrtTextureInputConstant( const otk::pbrt::PbrtMaterialGraph& graph,
                                          const otk::pbrt::PbrtTexture&       texture,
                                          const char*                         name,
                                          const typename Traits::Value&       defaultValue,
                                          std::vector<std::string>&           textureStack,
                                          typename Traits::Value&             value )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( !inputTextureName.empty() )
    {
        return foldPbrtTextureConstant<Traits>( graph, inputTextureName, textureStack, value );
    }
    return Traits::findConstant( texture.params, name, defaultValue, value );
}

template <typename Traits>
inline bool foldPbrtTextureConstant( const otk::pbrt::PbrtMaterialGraph& graph,
                                     const std::string&                  textureName,
                                     std::vector<std::string>&           textureStack,
                                     typename Traits::Value&             value )
{
    using Value = typename Traits::Value;

    std::string graphKey;
    const otk::pbrt::PbrtTexture* texture{ Traits::findTexture( graph, textureName, graphKey ) };
    if( texture == nullptr || pbrtTextureStackContains( textureStack, graphKey ) )
    {
        return false;
    }

    textureStack.push_back( graphKey );
    bool folded{ false };
    if( texture->type == "constant" )
    {
        folded = Traits::findConstant( texture->params, "value", Traits::one(), value );
    }
    else if( texture->type == "scale" )
    {
        Value tex1{};
        Value tex2{};
        folded = foldPbrtTextureInputConstant<Traits>( graph, *texture, "tex1", Traits::one(), textureStack, tex1 )
                 && foldPbrtTextureInputConstant<Traits>( graph, *texture, "tex2", Traits::one(), textureStack, tex2 );
        if( folded )
        {
            value = Traits::multiply( tex1, tex2 );
        }
    }
    else if( texture->type == "mix" && Traits::foldsMix() )
    {
        Value tex1{};
        Value tex2{};
        float amount{};
        folded = foldPbrtTextureInputConstant<Traits>( graph, *texture, "tex1", Traits::one(), textureStack, tex1 )
                 && foldPbrtTextureInputConstant<Traits>( graph, *texture, "tex2", Traits::one(), textureStack, tex2 )
                 && foldPbrtTextureInputConstant<PbrtFloatTextureTraits>( graph, *texture, "amount", 0.5f,
                                                                          textureStack, amount );
        if( folded )
        {
            value = Traits::mix( tex1, tex2, amount );
        }
    }

    textureStack.pop_back();
    return folded;
}

enum class PbrtTextureTraversalPolicy
{
    SCALE_ONLY,
    SCALE_AND_MIX,
};

template <typename Traits>
inline PbrtDemandTextureBinding findPbrtDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                              const std::string&                  textureName,
                                                              std::vector<std::string>&           textureStack,
                                                              PbrtTextureTraversalPolicy          policy );

template <typename Traits>
inline PbrtDemandTextureBinding findPbrtDirectDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                                    const std::string&                  textureName,
                                                                    std::vector<std::string>&           textureStack )
{
    std::string graphKey;
    const otk::pbrt::PbrtTexture* texture{ Traits::findTexture( graph, textureName, graphKey ) };
    if( texture == nullptr || pbrtTextureStackContains( textureStack, graphKey ) )
    {
        return pbrtDemandTextureBinding();
    }
    if( texture->type != "imagemap" && texture->type != "checkerboard" )
    {
        return pbrtDemandTextureBinding();
    }
    return pbrtDemandTextureBinding( pbrtTextureMapName( texture ), pbrtTextureScale( 1.0f ), pbrtTextureBias( 0.0f ),
                                     false, pbrtTextureGamma( texture ) );
}

template <typename Traits>
inline bool findPbrtDemandTextureInputBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                               const otk::pbrt::PbrtTexture&       texture,
                                               const char*                         name,
                                               std::vector<std::string>&           textureStack,
                                               PbrtDemandTextureBinding&           binding,
                                               PbrtTextureTraversalPolicy          policy )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( inputTextureName.empty() )
    {
        return false;
    }
    binding = findPbrtDemandTextureBinding<Traits>( graph, inputTextureName, textureStack, policy );
    return hasPbrtDemandTextureBinding( binding );
}

template <typename Traits>
inline bool findPbrtDirectDemandTextureInputBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                     const otk::pbrt::PbrtTexture&       texture,
                                                     const char*                         name,
                                                     std::vector<std::string>&           textureStack,
                                                     PbrtDemandTextureBinding&           binding )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( inputTextureName.empty() )
    {
        return false;
    }
    binding = findPbrtDirectDemandTextureBinding<Traits>( graph, inputTextureName, textureStack );
    return hasPbrtDemandTextureBinding( binding );
}

template <typename Traits>
inline PbrtDemandTextureBinding findPbrtScaleDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                                   const otk::pbrt::PbrtTexture&       texture,
                                                                   std::vector<std::string>&           textureStack )
{
    using Value = typename Traits::Value;

    PbrtDemandTextureBinding tex1Binding{};
    PbrtDemandTextureBinding tex2Binding{};
    const bool hasTex1Demand{ findPbrtDemandTextureInputBinding<Traits>(
        graph, texture, "tex1", textureStack, tex1Binding, PbrtTextureTraversalPolicy::SCALE_ONLY ) };
    const bool hasTex2Demand{ findPbrtDemandTextureInputBinding<Traits>(
        graph, texture, "tex2", textureStack, tex2Binding, PbrtTextureTraversalPolicy::SCALE_ONLY ) };
    if( hasTex1Demand == hasTex2Demand )
    {
        return pbrtDemandTextureBinding();
    }

    if( hasTex1Demand )
    {
        Value tex2Scale{};
        if( foldPbrtTextureInputConstant<Traits>( graph, texture, "tex2", Traits::one(), textureStack, tex2Scale ) )
        {
            const float3 scale{ Traits::color( tex2Scale ) };
            return pbrtDemandTextureBinding( tex1Binding.fileName, multiplyPbrtTextureScale( tex1Binding.scale, scale ),
                                             multiplyPbrtTextureScale( tex1Binding.bias, scale ), true, tex1Binding.gamma );
        }
        return pbrtDemandTextureBinding();
    }

    Value tex1Scale{};
    if( foldPbrtTextureInputConstant<Traits>( graph, texture, "tex1", Traits::one(), textureStack, tex1Scale ) )
    {
        const float3 scale{ Traits::color( tex1Scale ) };
        return pbrtDemandTextureBinding( tex2Binding.fileName, multiplyPbrtTextureScale( tex2Binding.scale, scale ),
                                         multiplyPbrtTextureScale( tex2Binding.bias, scale ), true, tex2Binding.gamma );
    }
    return pbrtDemandTextureBinding();
}

template <typename Traits>
inline PbrtDemandTextureBinding findPbrtMixDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                                 const otk::pbrt::PbrtTexture&       texture,
                                                                 std::vector<std::string>&           textureStack )
{
    using Value = typename Traits::Value;

    PbrtDemandTextureBinding tex1Binding{};
    PbrtDemandTextureBinding tex2Binding{};
    const bool hasTex1Demand{
        findPbrtDirectDemandTextureInputBinding<Traits>( graph, texture, "tex1", textureStack, tex1Binding ) };
    const bool hasTex2Demand{
        findPbrtDirectDemandTextureInputBinding<Traits>( graph, texture, "tex2", textureStack, tex2Binding ) };
    if( hasTex1Demand == hasTex2Demand )
    {
        return pbrtDemandTextureBinding();
    }

    float amount{};
    if( !foldPbrtTextureInputConstant<PbrtFloatTextureTraits>( graph, texture, "amount", 0.5f, textureStack, amount ) )
    {
        return pbrtDemandTextureBinding();
    }

    if( hasTex1Demand )
    {
        Value tex2{};
        if( foldPbrtTextureInputConstant<Traits>( graph, texture, "tex2", Traits::one(), textureStack, tex2 ) )
        {
            return pbrtDemandTextureBinding(
                tex1Binding.fileName, scalePbrtTextureColor( tex1Binding.scale, 1.0f - amount ),
                addPbrtTextureColor( scalePbrtTextureColor( tex1Binding.bias, 1.0f - amount ),
                                     scalePbrtTextureColor( Traits::color( tex2 ), amount ) ),
                true, tex1Binding.gamma );
        }
        return pbrtDemandTextureBinding();
    }

    Value tex1{};
    if( foldPbrtTextureInputConstant<Traits>( graph, texture, "tex1", Traits::one(), textureStack, tex1 ) )
    {
        return pbrtDemandTextureBinding(
            tex2Binding.fileName, scalePbrtTextureColor( tex2Binding.scale, amount ),
            addPbrtTextureColor( scalePbrtTextureColor( Traits::color( tex1 ), 1.0f - amount ),
                                 scalePbrtTextureColor( tex2Binding.bias, amount ) ),
            true, tex2Binding.gamma );
    }
    return pbrtDemandTextureBinding();
}

template <typename Traits>
inline PbrtDemandTextureBinding findPbrtDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                              const std::string&                  textureName,
                                                              std::vector<std::string>&           textureStack,
                                                              PbrtTextureTraversalPolicy          policy )
{
    std::string graphKey;
    const otk::pbrt::PbrtTexture* texture{ Traits::findTexture( graph, textureName, graphKey ) };
    if( texture == nullptr || pbrtTextureStackContains( textureStack, graphKey ) )
    {
        return pbrtDemandTextureBinding();
    }

    textureStack.push_back( graphKey );
    PbrtDemandTextureBinding result{};
    if( texture->type == "imagemap" || texture->type == "checkerboard" )
    {
        result = pbrtDemandTextureBinding( pbrtTextureMapName( texture ), pbrtTextureScale( 1.0f ),
                                           pbrtTextureBias( 0.0f ), false, pbrtTextureGamma( texture ) );
    }
    else if( texture->type == "scale" )
    {
        result = findPbrtScaleDemandTextureBinding<Traits>( graph, *texture, textureStack );
    }
    else if( texture->type == "mix" && policy == PbrtTextureTraversalPolicy::SCALE_AND_MIX )
    {
        result = findPbrtMixDemandTextureBinding<Traits>( graph, *texture, textureStack );
    }
    else
    {
        result = pbrtDemandTextureBinding();
    }
    textureStack.pop_back();
    return result;
}

inline PbrtDemandTextureBinding pbrtColorTextureBinding( const otk::pbrt::PbrtMaterial& material, const char* paramName )
{
    const std::string textureName{ material.params.FindTexture( paramName ) };
    if( textureName.empty() )
    {
        return pbrtDemandTextureBinding();
    }

    std::vector<std::string> textureStack;
    const PbrtTextureTraversalPolicy policy{ std::string{ paramName } == "Kd" ?
                                                 PbrtTextureTraversalPolicy::SCALE_AND_MIX :
                                                 PbrtTextureTraversalPolicy::SCALE_ONLY };
    return findPbrtDemandTextureBinding<PbrtColorTextureTraits>( material.graph, textureName, textureStack, policy );
}

inline PbrtDemandTextureBinding pbrtFloatTextureBinding( const otk::pbrt::PbrtMaterial& material, const char* paramName )
{
    const std::string textureName{ material.params.FindTexture( paramName ) };
    if( textureName.empty() )
    {
        return pbrtDemandTextureBinding();
    }

    std::vector<std::string> textureStack;
    const PbrtTextureTraversalPolicy policy{ std::string{ paramName } == "bumpmap" ?
                                                 PbrtTextureTraversalPolicy::SCALE_AND_MIX :
                                                 PbrtTextureTraversalPolicy::SCALE_ONLY };
    return findPbrtDemandTextureBinding<PbrtFloatTextureTraits>( material.graph, textureName, textureStack, policy );
}

inline std::string pbrtColorMapFileName( const otk::pbrt::PbrtMaterial& material, const char* paramName )
{
    return pbrtColorTextureBinding( material, paramName ).fileName;
}

inline std::string pbrtDiffuseMapFileName( const otk::pbrt::PbrtMaterial& material )
{
    return pbrtColorMapFileName( material, "Kd" );
}

inline std::string pbrtMirrorReflectanceMapFileName( const otk::pbrt::PbrtMaterial& material )
{
    return material.type == "mirror" ? pbrtColorMapFileName( material, "Kr" ) : std::string{};
}

inline std::string pbrtAlphaMapFileName( const otk::pbrt::PbrtMaterial& material )
{
    for( const char* paramName : { "alpha", "shadowalpha", "opacity" } )
    {
        const std::string textureName{ material.params.FindTexture( paramName ) };
        const std::string fileName{ pbrtTextureMapName( findPbrtTexture( material, textureName, { "float" } ) ) };
        if( !fileName.empty() )
        {
            return fileName;
        }
    }
    return {};
}

inline otk::pbrt::PlasticMaterial fallbackMaterialForShape( const otk::pbrt::ShapeDefinition& shape )
{
    otk::pbrt::PlasticMaterial result{ shape.material };

    const PbrtDemandTextureBinding diffuseTexture{ pbrtColorTextureBinding( shape.pbrtMaterial, "Kd" ) };
    if( hasPbrtDemandTextureBinding( diffuseTexture ) )
    {
        result.diffuseMapFileName = diffuseTexture.fileName;
        if( diffuseTexture.transformed )
        {
            result.Kd = toPbrtPoint3f( diffuseTexture.scale );
        }
    }
    else
    {
        const PbrtDemandTextureBinding reflectanceTexture{ shape.pbrtMaterial.type == "mirror" ?
                                                               pbrtColorTextureBinding( shape.pbrtMaterial, "Kr" ) :
                                                               pbrtDemandTextureBinding() };
        if( hasPbrtDemandTextureBinding( reflectanceTexture ) )
        {
            result.diffuseMapFileName = reflectanceTexture.fileName;
            if( reflectanceTexture.transformed )
            {
                result.Kd = toPbrtPoint3f( reflectanceTexture.scale );
            }
        }
    }

    const std::string alphaMapFileName{ pbrtAlphaMapFileName( shape.pbrtMaterial ) };
    if( !alphaMapFileName.empty() )
    {
        result.alphaMapFileName = alphaMapFileName;
    }

    return result;
}

inline MaterialFlags shapeMaterialFlags( const otk::pbrt::ShapeDefinition& shape )
{
    return plasticMaterialFlags( fallbackMaterialForShape( shape ) );
}

}  // namespace demandPbrtScene
