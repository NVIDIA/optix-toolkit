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

inline PbrtDemandTextureBinding pbrtDemandTextureBinding( const std::string& fileName, const float3& scale, const float3& bias, bool transformed )
{
    return PbrtDemandTextureBinding{ fileName, scale, bias, transformed };
}

inline PbrtDemandTextureBinding pbrtDemandTextureBinding()
{
    return pbrtDemandTextureBinding( std::string{}, pbrtTextureScale( 1.0f ), pbrtTextureBias( 0.0f ), false );
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

inline bool findPbrtTextureInputConstantColor( const ::pbrt::ParamSet& params, const char* name, const float3& defaultValue, float3& value )
{
    if( findPbrtConstantColor( params, name, value ) )
    {
        return true;
    }

    float floatValue{};
    if( findPbrtConstantFloat( params, name, floatValue ) )
    {
        value = pbrtTextureScale( floatValue );
        return true;
    }

    value = defaultValue;
    return true;
}

inline bool foldPbrtTextureConstantColor( const otk::pbrt::PbrtMaterialGraph& graph,
                                          const std::string&                  textureName,
                                          std::vector<std::string>&           textureStack,
                                          float3&                             value );

inline bool foldPbrtTextureConstantFloat( const otk::pbrt::PbrtMaterialGraph& graph,
                                          const std::string&                  textureName,
                                          std::vector<std::string>&           textureStack,
                                          float&                              value );

inline bool foldPbrtTextureInputConstantColor( const otk::pbrt::PbrtMaterialGraph& graph,
                                               const otk::pbrt::PbrtTexture&       texture,
                                               const char*                         name,
                                               const float3&                       defaultValue,
                                               std::vector<std::string>&           textureStack,
                                               float3&                             value )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( !inputTextureName.empty() )
    {
        return foldPbrtTextureConstantColor( graph, inputTextureName, textureStack, value );
    }
    return findPbrtTextureInputConstantColor( texture.params, name, defaultValue, value );
}

inline bool foldPbrtTextureInputConstantFloat( const otk::pbrt::PbrtMaterialGraph& graph,
                                               const otk::pbrt::PbrtTexture&       texture,
                                               const char*                         name,
                                               float                               defaultValue,
                                               std::vector<std::string>&           textureStack,
                                               float&                              value )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( !inputTextureName.empty() )
    {
        return foldPbrtTextureConstantFloat( graph, inputTextureName, textureStack, value );
    }
    if( findPbrtScalar( texture.params, name, value ) )
    {
        return true;
    }
    value = defaultValue;
    return true;
}

inline bool foldPbrtTextureConstantColor( const otk::pbrt::PbrtMaterialGraph& graph,
                                          const std::string&                  textureName,
                                          std::vector<std::string>&           textureStack,
                                          float3&                             value )
{
    std::string graphKey;
    const otk::pbrt::PbrtTexture* texture{ findPbrtGraphTexture( graph, textureName, { "spectrum", "color" }, graphKey ) };
    if( texture == nullptr || pbrtTextureStackContains( textureStack, graphKey ) )
    {
        return false;
    }

    textureStack.push_back( graphKey );
    bool folded{ false };
    if( texture->type == "constant" )
    {
        folded = findPbrtTextureInputConstantColor( texture->params, "value", pbrtTextureScale( 1.0f ), value );
    }
    else if( texture->type == "scale" )
    {
        float3 tex1{};
        float3 tex2{};
        folded = foldPbrtTextureInputConstantColor( graph, *texture, "tex1", pbrtTextureScale( 1.0f ), textureStack, tex1 )
                 && foldPbrtTextureInputConstantColor( graph, *texture, "tex2", pbrtTextureScale( 1.0f ), textureStack, tex2 );
        if( folded )
        {
            value = multiplyPbrtTextureScale( tex1, tex2 );
        }
    }

    textureStack.pop_back();
    return folded;
}

inline bool foldPbrtTextureConstantFloat( const otk::pbrt::PbrtMaterialGraph& graph,
                                          const std::string&                  textureName,
                                          std::vector<std::string>&           textureStack,
                                          float&                              value )
{
    std::string                   graphKey;
    const otk::pbrt::PbrtTexture* texture{ findPbrtGraphTexture( graph, textureName, { "float" }, graphKey ) };
    if( texture == nullptr || pbrtTextureStackContains( textureStack, graphKey ) )
    {
        return false;
    }

    textureStack.push_back( graphKey );
    bool folded{ false };
    if( texture->type == "constant" )
    {
        folded = foldPbrtTextureInputConstantFloat( graph, *texture, "value", 1.0f, textureStack, value );
    }
    else if( texture->type == "scale" )
    {
        float tex1{};
        float tex2{};
        folded = foldPbrtTextureInputConstantFloat( graph, *texture, "tex1", 1.0f, textureStack, tex1 )
                 && foldPbrtTextureInputConstantFloat( graph, *texture, "tex2", 1.0f, textureStack, tex2 );
        if( folded )
        {
            value = tex1 * tex2;
        }
    }
    else if( texture->type == "mix" )
    {
        float tex1{};
        float tex2{};
        float amount{};
        folded = foldPbrtTextureInputConstantFloat( graph, *texture, "tex1", 1.0f, textureStack, tex1 )
                 && foldPbrtTextureInputConstantFloat( graph, *texture, "tex2", 1.0f, textureStack, tex2 )
                 && foldPbrtTextureInputConstantFloat( graph, *texture, "amount", 0.5f, textureStack, amount );
        if( folded )
        {
            value = tex1 * ( 1.0f - amount ) + tex2 * amount;
        }
    }

    textureStack.pop_back();
    return folded;
}

inline PbrtDemandTextureBinding findPbrtDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                              const std::string&                  textureName,
                                                              std::vector<std::string>&           textureStack,
                                                              bool                                allowMix );

inline PbrtDemandTextureBinding findPbrtDirectDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                                    const std::string&                  textureName,
                                                                    std::vector<std::string>&           textureStack )
{
    std::string graphKey;
    const otk::pbrt::PbrtTexture* texture{ findPbrtGraphTexture( graph, textureName, { "spectrum", "color" }, graphKey ) };
    if( texture == nullptr || pbrtTextureStackContains( textureStack, graphKey ) )
    {
        return pbrtDemandTextureBinding();
    }
    if( texture->type != "imagemap" && texture->type != "checkerboard" )
    {
        return pbrtDemandTextureBinding();
    }
    return pbrtDemandTextureBinding( pbrtTextureMapName( texture ), pbrtTextureScale( 1.0f ), pbrtTextureBias( 0.0f ), false );
}

inline bool findPbrtDemandTextureInputBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                               const otk::pbrt::PbrtTexture&       texture,
                                               const char*                         name,
                                               std::vector<std::string>&           textureStack,
                                               PbrtDemandTextureBinding&           binding,
                                               bool                                allowMix )
{
    const std::string inputTextureName{ texture.params.FindTexture( name ) };
    if( inputTextureName.empty() )
    {
        return false;
    }
    binding = findPbrtDemandTextureBinding( graph, inputTextureName, textureStack, allowMix );
    return hasPbrtDemandTextureBinding( binding );
}

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
    binding = findPbrtDirectDemandTextureBinding( graph, inputTextureName, textureStack );
    return hasPbrtDemandTextureBinding( binding );
}

inline PbrtDemandTextureBinding findPbrtScaleDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                                   const otk::pbrt::PbrtTexture&       texture,
                                                                   std::vector<std::string>&           textureStack )
{
    PbrtDemandTextureBinding tex1Binding{};
    PbrtDemandTextureBinding tex2Binding{};
    const bool hasTex1Demand{ findPbrtDemandTextureInputBinding( graph, texture, "tex1", textureStack, tex1Binding, false ) };
    const bool hasTex2Demand{ findPbrtDemandTextureInputBinding( graph, texture, "tex2", textureStack, tex2Binding, false ) };
    if( hasTex1Demand == hasTex2Demand )
    {
        return pbrtDemandTextureBinding();
    }

    if( hasTex1Demand )
    {
        float3 tex2Scale{};
        if( foldPbrtTextureInputConstantColor( graph, texture, "tex2", pbrtTextureScale( 1.0f ), textureStack, tex2Scale ) )
        {
            return pbrtDemandTextureBinding( tex1Binding.fileName, multiplyPbrtTextureScale( tex1Binding.scale, tex2Scale ),
                                             multiplyPbrtTextureScale( tex1Binding.bias, tex2Scale ), true );
        }
        return pbrtDemandTextureBinding();
    }

    float3 tex1Scale{};
    if( foldPbrtTextureInputConstantColor( graph, texture, "tex1", pbrtTextureScale( 1.0f ), textureStack, tex1Scale ) )
    {
        return pbrtDemandTextureBinding( tex2Binding.fileName, multiplyPbrtTextureScale( tex2Binding.scale, tex1Scale ),
                                         multiplyPbrtTextureScale( tex2Binding.bias, tex1Scale ), true );
    }
    return pbrtDemandTextureBinding();
}

inline PbrtDemandTextureBinding findPbrtMixDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                                 const otk::pbrt::PbrtTexture&       texture,
                                                                 std::vector<std::string>&           textureStack )
{
    PbrtDemandTextureBinding tex1Binding{};
    PbrtDemandTextureBinding tex2Binding{};
    const bool hasTex1Demand{ findPbrtDirectDemandTextureInputBinding( graph, texture, "tex1", textureStack, tex1Binding ) };
    const bool hasTex2Demand{ findPbrtDirectDemandTextureInputBinding( graph, texture, "tex2", textureStack, tex2Binding ) };
    if( hasTex1Demand == hasTex2Demand )
    {
        return pbrtDemandTextureBinding();
    }

    float amount{};
    if( !foldPbrtTextureInputConstantFloat( graph, texture, "amount", 0.5f, textureStack, amount ) )
    {
        return pbrtDemandTextureBinding();
    }

    if( hasTex1Demand )
    {
        float3 tex2{};
        if( foldPbrtTextureInputConstantColor( graph, texture, "tex2", pbrtTextureScale( 1.0f ), textureStack, tex2 ) )
        {
            return pbrtDemandTextureBinding( tex1Binding.fileName, scalePbrtTextureColor( tex1Binding.scale, 1.0f - amount ),
                                             addPbrtTextureColor( scalePbrtTextureColor( tex1Binding.bias, 1.0f - amount ),
                                                                  scalePbrtTextureColor( tex2, amount ) ),
                                             true );
        }
        return pbrtDemandTextureBinding();
    }

    float3 tex1{};
    if( foldPbrtTextureInputConstantColor( graph, texture, "tex1", pbrtTextureScale( 1.0f ), textureStack, tex1 ) )
    {
        return pbrtDemandTextureBinding( tex2Binding.fileName, scalePbrtTextureColor( tex2Binding.scale, amount ),
                                         addPbrtTextureColor( scalePbrtTextureColor( tex1, 1.0f - amount ),
                                                              scalePbrtTextureColor( tex2Binding.bias, amount ) ),
                                         true );
    }
    return pbrtDemandTextureBinding();
}

inline PbrtDemandTextureBinding findPbrtDemandTextureBinding( const otk::pbrt::PbrtMaterialGraph& graph,
                                                              const std::string&                  textureName,
                                                              std::vector<std::string>&           textureStack,
                                                              bool                                allowMix )
{
    std::string graphKey;
    const otk::pbrt::PbrtTexture* texture{ findPbrtGraphTexture( graph, textureName, { "spectrum", "color" }, graphKey ) };
    if( texture == nullptr || pbrtTextureStackContains( textureStack, graphKey ) )
    {
        return pbrtDemandTextureBinding();
    }

    textureStack.push_back( graphKey );
    PbrtDemandTextureBinding result{};
    if( texture->type == "imagemap" || texture->type == "checkerboard" )
    {
        result = pbrtDemandTextureBinding( pbrtTextureMapName( texture ), pbrtTextureScale( 1.0f ), pbrtTextureBias( 0.0f ), false );
    }
    else if( texture->type == "scale" )
    {
        result = findPbrtScaleDemandTextureBinding( graph, *texture, textureStack );
    }
    else if( texture->type == "mix" && allowMix )
    {
        result = findPbrtMixDemandTextureBinding( graph, *texture, textureStack );
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
    return findPbrtDemandTextureBinding( material.graph, textureName, textureStack, std::string{ paramName } == "Kd" );
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
