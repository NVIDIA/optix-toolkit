// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/Config.h>

#ifdef OTK_USE_MDL

#include <gmock/gmock.h>
#include "DemandPbrtScene/MdlShaderCache.h"

#include <memory>
#include <string>
#include <utility>

using namespace demandPbrtScene;
using namespace otk::pbrt;

namespace {

void addRgbSpectrum( ::pbrt::ParamSet& params, const std::string& name, float r, float g, float b )
{
    std::unique_ptr<::pbrt::Float[]> values{ new ::pbrt::Float[3] };
    values[0] = r;
    values[1] = g;
    values[2] = b;
    params.AddRGBSpectrum( name, std::move( values ), 3 );
}

void addString( ::pbrt::ParamSet& params, const std::string& name, const std::string& value )
{
    std::unique_ptr<std::string[]> values{ new std::string[1] };
    values[0] = value;
    params.AddString( name, std::move( values ), 1 );
}

void addFloat( ::pbrt::ParamSet& params, const std::string& name, float value )
{
    std::unique_ptr<::pbrt::Float[]> values{ new ::pbrt::Float[1] };
    values[0] = value;
    params.AddFloat( name, std::move( values ), 1 );
}

PbrtMaterial matteMaterial( float kd )
{
    PbrtMaterial material;
    material.type = "matte";
    addRgbSpectrum( material.params, "Kd", kd, kd, kd );
    return material;
}

PbrtMaterial matteMaterialWithSigmaAndCutout()
{
    PbrtMaterial material{ matteMaterial( 0.2f ) };
    addFloat( material.params, "sigma", 20.0f );
    addFloat( material.params, "alpha", 0.8f );
    addFloat( material.params, "opacity", 0.7f );
    return material;
}

PbrtTexture imageMapTexture( const std::string& name,
                             const std::string& fileName,
                             const std::string& valueType = "spectrum" )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = valueType;
    texture.type      = "imagemap";
    addString( texture.params, "filename", fileName );
    return texture;
}

PbrtTexture checkerboardTexture( const std::string& name, float tex1, float tex2 )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "spectrum";
    texture.type      = "checkerboard";
    addRgbSpectrum( texture.params, "tex1", tex1, tex1, tex1 );
    addRgbSpectrum( texture.params, "tex2", tex2, tex2, tex2 );
    return texture;
}

PbrtTexture constantTexture( const std::string& name )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "color";
    texture.type      = "constant";
    addRgbSpectrum( texture.params, "value", 0.5f, 0.5f, 0.5f );
    return texture;
}

PbrtTexture constantFloatTexture( const std::string& name, float value )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "float";
    texture.type      = "constant";
    addFloat( texture.params, "value", value );
    return texture;
}

PbrtTexture scaleTexture( const std::string& name, const std::string& tex1 )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "color";
    texture.type      = "scale";
    texture.params.AddTexture( "tex1", tex1 );
    addRgbSpectrum( texture.params, "tex2", 0.5f, 0.5f, 0.5f );
    return texture;
}

PbrtTexture mixTexture( const std::string& name, const std::string& tex1, const std::string& tex2 )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "color";
    texture.type      = "mix";
    texture.params.AddTexture( "tex1", tex1 );
    texture.params.AddTexture( "tex2", tex2 );
    return texture;
}

PbrtTexture unsupportedTexture( const std::string& name, const std::string& valueType, const std::string& type )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = valueType;
    texture.type      = type;
    return texture;
}

PbrtMaterial materialWithKdTexture( const std::string& textureName )
{
    PbrtMaterial material;
    material.type = "matte";
    material.params.AddTexture( "Kd", textureName );
    return material;
}

PbrtMaterial materialOfType( const std::string& type )
{
    PbrtMaterial material;
    material.type = type;
    return material;
}

PbrtMaterial plasticMaterial()
{
    PbrtMaterial material;
    material.type = "plastic";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addFloat( material.params, "roughness", 0.25f );
    material.params.AddTexture( "bumpmap", "height" );
    material.graph.textures["float:height"] = imageMapTexture( "height", "height.exr", "float" );
    return material;
}

PbrtMaterial uberMaterial()
{
    PbrtMaterial material;
    material.type = "uber";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addRgbSpectrum( material.params, "Kr", 0.1f, 0.1f, 0.1f );
    addRgbSpectrum( material.params, "Kt", 0.0f, 0.1f, 0.2f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "index", 1.4f );
    addFloat( material.params, "alpha", 0.8f );
    addFloat( material.params, "opacity", 0.7f );
    material.params.AddTexture( "Kd", "albedo" );
    material.params.AddTexture( "bumpmap", "height" );
    material.graph.textures["color:albedo"] = imageMapTexture( "albedo", "albedo.exr", "color" );
    material.graph.textures["float:height"] = imageMapTexture( "height", "height.exr", "float" );
    return material;
}

PbrtMaterial constantUberMaterial()
{
    PbrtMaterial material;
    material.type = "uber";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addRgbSpectrum( material.params, "Kr", 0.1f, 0.2f, 0.3f );
    addRgbSpectrum( material.params, "Kt", 0.0f, 0.1f, 0.2f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "uroughness", 0.2f );
    addFloat( material.params, "vroughness", 0.3f );
    addFloat( material.params, "index", 1.4f );
    addFloat( material.params, "alpha", 0.8f );
    addFloat( material.params, "opacity", 0.7f );
    return material;
}

PbrtMaterial mirrorMaterial()
{
    PbrtMaterial material;
    material.type = "mirror";
    material.params.AddTexture( "Kr", "reflectance" );
    material.graph.textures["color:reflectance"] = imageMapTexture( "reflectance", "reflectance.exr", "color" );
    return material;
}

PbrtMaterial constantMirrorMaterial()
{
    PbrtMaterial material;
    material.type = "mirror";
    addRgbSpectrum( material.params, "Kr", 0.2f, 0.3f, 0.4f );
    return material;
}

PbrtMaterial glassMaterial()
{
    PbrtMaterial material;
    material.type = "glass";
    addRgbSpectrum( material.params, "Kr", 0.9f, 0.9f, 0.9f );
    addRgbSpectrum( material.params, "Kt", 0.8f, 0.9f, 1.0f );
    addFloat( material.params, "index", 1.5f );
    addFloat( material.params, "roughness", 0.05f );
    addFloat( material.params, "uroughness", 0.04f );
    addFloat( material.params, "vroughness", 0.06f );
    return material;
}

PbrtMaterial metalMaterial()
{
    PbrtMaterial material;
    material.type = "metal";
    addRgbSpectrum( material.params, "eta", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "k", 2.0f, 2.5f, 3.0f );
    addFloat( material.params, "roughness", 0.2f );
    addFloat( material.params, "uroughness", 0.15f );
    addFloat( material.params, "vroughness", 0.25f );
    material.params.AddTexture( "eta", "etaMap" );
    material.params.AddTexture( "k", "kMap" );
    material.graph.textures["color:etaMap"] = imageMapTexture( "etaMap", "eta.exr", "color" );
    material.graph.textures["color:kMap"]   = imageMapTexture( "kMap", "k.exr", "color" );
    return material;
}

PbrtMaterial constantMetalMaterial()
{
    PbrtMaterial material;
    material.type = "metal";
    addRgbSpectrum( material.params, "eta", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "k", 2.0f, 2.5f, 3.0f );
    addFloat( material.params, "roughness", 0.2f );
    addFloat( material.params, "uroughness", 0.15f );
    addFloat( material.params, "vroughness", 0.25f );
    return material;
}

PbrtMaterial substrateMaterial()
{
    PbrtMaterial material;
    material.type = "substrate";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "uroughness", 0.2f );
    addFloat( material.params, "vroughness", 0.3f );
    material.params.AddTexture( "bumpmap", "height" );
    material.graph.textures["float:height"] = imageMapTexture( "height", "height.exr", "float" );
    return material;
}

PbrtMaterial translucentMaterial()
{
    PbrtMaterial material;
    material.type = "translucent";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addRgbSpectrum( material.params, "reflect", 0.8f, 0.8f, 0.8f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "opacity", 0.7f );
    material.params.AddTexture( "transmit", "leafTransmit" );
    material.params.AddTexture( "opacity", "leafOpacity" );
    material.graph.textures["color:leafTransmit"] = imageMapTexture( "leafTransmit", "leaf-transmit.exr", "color" );
    material.graph.textures["float:leafOpacity"]  = imageMapTexture( "leafOpacity", "leaf-opacity.exr", "float" );
    return material;
}

PbrtMaterial constantTranslucentMaterial()
{
    PbrtMaterial material;
    material.type = "translucent";
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "Ks", 0.5f, 0.6f, 0.7f );
    addRgbSpectrum( material.params, "reflect", 0.8f, 0.7f, 0.6f );
    addRgbSpectrum( material.params, "transmit", 0.2f, 0.3f, 0.4f );
    addFloat( material.params, "roughness", 0.25f );
    addFloat( material.params, "opacity", 0.7f );
    return material;
}

PbrtMaterial texturedMatteMaterial( const std::string& textureType, const std::string& fileName )
{
    PbrtMaterial material;
    material.type = "matte";
    material.params.AddTexture( "Kd", "albedo" );

    if( textureType == "imagemap" )
    {
        material.graph.textures["spectrum:albedo"] = imageMapTexture( "albedo", fileName );
    }
    else
    {
        material.graph.textures["spectrum:albedo"] = checkerboardTexture( "albedo", 0.25f, 0.75f );
    }
    return material;
}

PbrtNamedMaterial namedMatte( const std::string& name, float kd )
{
    PbrtNamedMaterial material;
    material.name = name;
    material.type = "matte";
    addString( material.params, "type", "matte" );
    addRgbSpectrum( material.params, "Kd", kd, kd, kd );
    return material;
}

PbrtMaterial mixMaterial( const std::string& firstName, const std::string& secondName )
{
    PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", firstName );
    addString( material.params, "namedmaterial2", secondName );
    material.graph.namedMaterials[firstName]  = namedMatte( firstName, 0.2f );
    material.graph.namedMaterials[secondName] = namedMatte( secondName, 0.8f );
    return material;
}

PbrtMaterial constantMixMaterial()
{
    PbrtMaterial material{ mixMaterial( "front", "back" ) };
    addFloat( material.params, "amount", 0.25f );
    return material;
}

PbrtMaterial constantMixMaterialWithAmountTexture()
{
    PbrtMaterial material{ mixMaterial( "front", "back" ) };
    material.params.AddTexture( "amount", "weight" );
    material.graph.textures["float:weight"] = constantFloatTexture( "weight", 0.25f );
    return material;
}

PbrtNamedMaterial namedUberWithKdTexture( const std::string& name, const std::string& textureName )
{
    PbrtNamedMaterial material;
    material.name = name;
    material.type = "uber";
    addString( material.params, "type", "uber" );
    material.params.AddTexture( "Kd", textureName );
    addRgbSpectrum( material.params, "Ks", 0.3f, 0.3f, 0.3f );
    addRgbSpectrum( material.params, "Kr", 0.1f, 0.1f, 0.1f );
    addRgbSpectrum( material.params, "Kt", 0.0f, 0.0f, 0.0f );
    return material;
}

PbrtNamedMaterial namedTranslucentWithTransmitTexture( const std::string& name, const std::string& textureName )
{
    PbrtNamedMaterial material;
    material.name = name;
    material.type = "translucent";
    addString( material.params, "type", "translucent" );
    addRgbSpectrum( material.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( material.params, "reflect", 0.5f, 0.5f, 0.5f );
    material.params.AddTexture( "transmit", textureName );
    return material;
}

PbrtMaterial layeredMixMaterial()
{
    PbrtMaterial material;
    material.type = "mix";
    addString( material.params, "namedmaterial1", "front" );
    addString( material.params, "namedmaterial2", "back" );
    addFloat( material.params, "amount", 0.25f );
    material.graph.namedMaterials["front"]         = namedUberWithKdTexture( "front", "frontColor" );
    material.graph.namedMaterials["back"]          = namedTranslucentWithTransmitTexture( "back", "backTransmit" );
    material.graph.textures["spectrum:frontColor"] = imageMapTexture( "frontColor", "front.png" );
    material.graph.textures["color:backTransmit"]  = imageMapTexture( "backTransmit", "back-transmit.exr", "color" );
    return material;
}

const MdlBoundMaterialParameter* findBoundParameter( const std::vector<MdlBoundMaterialParameter>& parameters, const std::string& name )
{
    for( std::vector<MdlBoundMaterialParameter>::const_iterator it = parameters.begin(); it != parameters.end(); ++it )
    {
        if( it->name == name )
        {
            return &*it;
        }
    }
    return nullptr;
}

void expectBoundColor( const std::vector<MdlBoundMaterialParameter>& parameters, const std::string& name, float red, float green, float blue )
{
    const MdlBoundMaterialParameter* parameter{ findBoundParameter( parameters, name ) };
    ASSERT_NE( nullptr, parameter ) << name;
    EXPECT_EQ( MdlBoundParameterType::COLOR, parameter->type );
    EXPECT_FLOAT_EQ( red, parameter->red );
    EXPECT_FLOAT_EQ( green, parameter->green );
    EXPECT_FLOAT_EQ( blue, parameter->blue );
}

void expectBoundFloat( const std::vector<MdlBoundMaterialParameter>& parameters, const std::string& name, float value )
{
    const MdlBoundMaterialParameter* parameter{ findBoundParameter( parameters, name ) };
    ASSERT_NE( nullptr, parameter ) << name;
    EXPECT_EQ( MdlBoundParameterType::FLOAT, parameter->type );
    EXPECT_FLOAT_EQ( value, parameter->value );
}

}  // namespace

TEST( TestMdlShaderKey, parameterOnlyMaterialChangesProduceSameKey )
{
    EXPECT_EQ( makeMdlShaderKey( matteMaterial( 0.1f ) ), makeMdlShaderKey( matteMaterial( 0.9f ) ) );
}

TEST( TestMdlShaderKey, parameterOnlyTextureChangesProduceSameKey )
{
    EXPECT_EQ( makeMdlShaderKey( texturedMatteMaterial( "imagemap", "first.png" ) ),
               makeMdlShaderKey( texturedMatteMaterial( "imagemap", "second.png" ) ) );
}

TEST( TestMdlMaterialInstanceKey, parameterOnlyMaterialChangesProduceDifferentInstanceKeys )
{
    const PbrtMaterial first{ matteMaterial( 0.1f ) };
    const PbrtMaterial second{ matteMaterial( 0.9f ) };

    const MdlMaterialInstanceKey firstKey{ makeMdlMaterialInstanceKey( first ) };
    const MdlMaterialInstanceKey secondKey{ makeMdlMaterialInstanceKey( second ) };

    EXPECT_EQ( firstKey.sourceKey, secondKey.sourceKey );
    EXPECT_NE( firstKey, secondKey );
    EXPECT_THAT( toString( firstKey ), testing::HasSubstr( "0.1" ) );
    EXPECT_THAT( toString( secondKey ), testing::HasSubstr( "0.899" ) );
}

TEST( TestMdlMaterialInstanceKey, textureParameterValuesProduceDifferentInstanceKeys )
{
    const PbrtMaterial first{ texturedMatteMaterial( "imagemap", "first.png" ) };
    const PbrtMaterial second{ texturedMatteMaterial( "imagemap", "second.png" ) };

    const MdlMaterialInstanceKey firstKey{ makeMdlMaterialInstanceKey( first ) };
    const MdlMaterialInstanceKey secondKey{ makeMdlMaterialInstanceKey( second ) };

    EXPECT_EQ( firstKey.sourceKey, secondKey.sourceKey );
    EXPECT_NE( firstKey, secondKey );
    EXPECT_THAT( toString( firstKey ), testing::HasSubstr( "first.png" ) );
    EXPECT_THAT( toString( secondKey ), testing::HasSubstr( "second.png" ) );
}
TEST( TestMdlShaderKey, textureStructureChangesProduceDifferentKeys )
{
    EXPECT_NE( makeMdlShaderKey( texturedMatteMaterial( "imagemap", "albedo.png" ) ),
               makeMdlShaderKey( texturedMatteMaterial( "checkerboard", "" ) ) );
}

TEST( TestMdlShaderKey, namedMaterialNamesDoNotChangeEquivalentGraphKey )
{
    EXPECT_EQ( makeMdlShaderKey( mixMaterial( "front", "back" ) ), makeMdlShaderKey( mixMaterial( "left", "right" ) ) );
}

TEST( TestMdlGeneratedSourceCache, cachesGeneratedSourceByShaderKey )
{
    MdlGeneratedSourceCache cache;

    const MdlShaderKey        firstKey{ makeMdlShaderKey( matteMaterial( 0.1f ) ) };
    const GeneratedMdlSource& first{ cache.getSource( firstKey ) };
    EXPECT_TRUE( cache.contains( firstKey ) );
    EXPECT_EQ( 1U, cache.size() );
    EXPECT_THAT( first.moduleName, testing::StartsWith( "::otk::demand_pbrt_scene::pbrt_" ) );
    EXPECT_THAT( first.materialName, testing::StartsWith( "material_" ) );
    EXPECT_THAT( first.source, testing::HasSubstr( "export material " + first.materialName ) );

    const MdlShaderKey        equivalentKey{ makeMdlShaderKey( matteMaterial( 0.9f ) ) };
    const GeneratedMdlSource& equivalent{ cache.getSource( equivalentKey ) };
    EXPECT_EQ( 1U, cache.size() );
    EXPECT_EQ( first.moduleName, equivalent.moduleName );
    EXPECT_EQ( first.materialName, equivalent.materialName );
    EXPECT_EQ( first.source, equivalent.source );

    cache.getSource( makeMdlShaderKey( texturedMatteMaterial( "imagemap", "albedo.png" ) ) );
    EXPECT_EQ( 2U, cache.size() );
}

TEST( TestMdlGeneratedSource, mapsImagemapTextureNode )
{
    const GeneratedMdlSource generated{ generateMdlSource( texturedMatteMaterial( "imagemap", "albedo.png" ) ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: spectrum:imagemap" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color pbrt_demand_texture_2d(int texture_id) = color(1.0, 1.0, "
                                                       "1.0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// demand texture parameter: texture_2d image_0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_0() = pbrt_demand_texture_2d(0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "albedo.png" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsCheckerboardTextureNodeToDemandTexture )
{
    const GeneratedMdlSource generated{ generateMdlSource( texturedMatteMaterial( "checkerboard", "" ) ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: spectrum:checkerboard" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color pbrt_demand_texture_2d(int texture_id) = color(1.0, 1.0, "
                                                       "1.0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// demand texture parameter: texture_2d image_0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_0() = pbrt_demand_texture_2d(0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: texture_0()" ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsMatteMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( matteMaterial( 0.2f ) ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "mdl 1.10" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: matte" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kd = color(0.8, 0.8, 0.8)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float sigma = 0.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float alpha = 1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float opacity = 1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input Kd: Kd" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input sigma: sigma" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material approximation: sigma degrees map to MDL "
                                                       "Oren-Nayar roughness sigma / 90" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input alpha: alpha; texture=none" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input shadowalpha: any-hit texture=none" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input opacity: opacity; texture=none" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float pbrt_matte_sigma_roughness(float sigma_degrees) = "
                                                       "::math::clamp(sigma_degrees / 90.0, 0.0, 1.0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: Kd" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "roughness: pbrt_matte_sigma_roughness(sigma)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "geometry: material_geometry" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "cutout_opacity: alpha * opacity" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "0.2" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsMatteSigmaAndCutoutParameters )
{
    const GeneratedMdlSource generated{ generateMdlSource( matteMaterialWithSigmaAndCutout() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "float sigma = 0.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float alpha = 1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float opacity = 1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "roughness: pbrt_matte_sigma_roughness(sigma)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "cutout_opacity: alpha * opacity" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "20.0" ) ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "0.7" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsPlasticMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( plasticMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: plastic" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kd = color(0.8, 0.8, 0.8)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Ks = color(0.0, 0.0, 0.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float roughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input bumpmap: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material gap: PBRT-exact roughness/remapping "
                                                       "behavior is approximated" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material approximation: diffuse and glossy reflection "
                                                       "use an MDL color-normalized mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_normalized_mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::color_bsdf_component[]" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "weight: Kd" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "weight: Ks" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::simple_glossy_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "roughness_u: roughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "roughness_v: roughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "component: ::df::diffuse_reflection_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: color(1.0, 1.0, 1.0)" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "pbrt_plastic_approximation_tint" ) ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: float:imagemap" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "height.exr" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsSimpleUberMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( uberMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: uber" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kd = color(0.8, 0.8, 0.8)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Ks = color(0.0, 0.0, 0.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kr = color(0.0, 0.0, 0.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kt = color(0.0, 0.0, 0.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float roughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float uroughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float vroughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float index = 1.5" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float alpha = 1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float opacity = 1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input Kd: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input bumpmap: texture_1()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input uroughness: uroughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input vroughness: vroughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_uber_approximation_tint(texture_0(), Ks, Kr, Kt, "
                                                       "roughness)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "ior: color(index, index, index)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "geometry: material_geometry" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "cutout_opacity: alpha * opacity" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "albedo.exr" ) ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "height.exr" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlBoundMaterialParameters, bindsPlasticConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( plasticMaterial() ) };

    EXPECT_EQ( 3U, parameters.size() );
    expectBoundColor( parameters, "Kd", 0.2f, 0.3f, 0.4f );
    expectBoundColor( parameters, "Ks", 0.5f, 0.6f, 0.7f );
    expectBoundFloat( parameters, "roughness", 0.25f );
}

TEST( TestMdlBoundMaterialParameters, bindsMatteSigmaAndCutoutConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( matteMaterialWithSigmaAndCutout() ) };

    EXPECT_EQ( 4U, parameters.size() );
    expectBoundColor( parameters, "Kd", 0.2f, 0.2f, 0.2f );
    expectBoundFloat( parameters, "sigma", 20.0f );
    expectBoundFloat( parameters, "alpha", 0.8f );
    expectBoundFloat( parameters, "opacity", 0.7f );
}

TEST( TestMdlBoundMaterialParameters, bindsUberConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( constantUberMaterial() ) };

    EXPECT_EQ( 10U, parameters.size() );
    expectBoundColor( parameters, "Kd", 0.2f, 0.3f, 0.4f );
    expectBoundColor( parameters, "Ks", 0.5f, 0.6f, 0.7f );
    expectBoundColor( parameters, "Kr", 0.1f, 0.2f, 0.3f );
    expectBoundColor( parameters, "Kt", 0.0f, 0.1f, 0.2f );
    expectBoundFloat( parameters, "roughness", 0.25f );
    expectBoundFloat( parameters, "uroughness", 0.2f );
    expectBoundFloat( parameters, "vroughness", 0.3f );
    expectBoundFloat( parameters, "index", 1.4f );
    expectBoundFloat( parameters, "alpha", 0.8f );
    expectBoundFloat( parameters, "opacity", 0.7f );
}

TEST( TestMdlBoundMaterialParameters, bindsSubstrateConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( substrateMaterial() ) };

    EXPECT_EQ( 5U, parameters.size() );
    expectBoundColor( parameters, "Kd", 0.2f, 0.3f, 0.4f );
    expectBoundColor( parameters, "Ks", 0.5f, 0.6f, 0.7f );
    expectBoundFloat( parameters, "roughness", 0.25f );
    expectBoundFloat( parameters, "uroughness", 0.2f );
    expectBoundFloat( parameters, "vroughness", 0.3f );
}

TEST( TestMdlBoundMaterialParameters, bindsMirrorConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( constantMirrorMaterial() ) };

    ASSERT_EQ( 1U, parameters.size() );
    expectBoundColor( parameters, "Kr", 0.2f, 0.3f, 0.4f );
}

TEST( TestMdlBoundMaterialParameters, bindsGlassConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( glassMaterial() ) };

    EXPECT_EQ( 6U, parameters.size() );
    expectBoundColor( parameters, "Kr", 0.9f, 0.9f, 0.9f );
    expectBoundColor( parameters, "Kt", 0.8f, 0.9f, 1.0f );
    expectBoundFloat( parameters, "index", 1.5f );
    expectBoundFloat( parameters, "roughness", 0.05f );
    expectBoundFloat( parameters, "uroughness", 0.04f );
    expectBoundFloat( parameters, "vroughness", 0.06f );
}

TEST( TestMdlBoundMaterialParameters, bindsMetalConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( constantMetalMaterial() ) };

    EXPECT_EQ( 5U, parameters.size() );
    expectBoundColor( parameters, "eta", 0.2f, 0.3f, 0.4f );
    expectBoundColor( parameters, "k", 2.0f, 2.5f, 3.0f );
    expectBoundFloat( parameters, "roughness", 0.2f );
    expectBoundFloat( parameters, "uroughness", 0.15f );
    expectBoundFloat( parameters, "vroughness", 0.25f );
}

TEST( TestMdlBoundMaterialParameters, bindsTranslucentConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( constantTranslucentMaterial() ) };

    EXPECT_EQ( 6U, parameters.size() );
    expectBoundColor( parameters, "Kd", 0.2f, 0.3f, 0.4f );
    expectBoundColor( parameters, "Ks", 0.5f, 0.6f, 0.7f );
    expectBoundColor( parameters, "reflect", 0.8f, 0.7f, 0.6f );
    expectBoundColor( parameters, "transmit", 0.2f, 0.3f, 0.4f );
    expectBoundFloat( parameters, "roughness", 0.25f );
    expectBoundFloat( parameters, "opacity", 0.7f );
}

TEST( TestMdlBoundMaterialParameters, bindsMixConstantsAndNamedMaterialConstants )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( constantMixMaterial() ) };

    EXPECT_EQ( 3U, parameters.size() );
    expectBoundFloat( parameters, "amount", 0.25f );
    expectBoundColor( parameters, "named_0_Kd", 0.2f, 0.2f, 0.2f );
    expectBoundColor( parameters, "named_1_Kd", 0.8f, 0.8f, 0.8f );
}

TEST( TestMdlBoundMaterialParameters, bindsMixConstantAmountTexture )
{
    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( constantMixMaterialWithAmountTexture() ) };

    EXPECT_EQ( 3U, parameters.size() );
    expectBoundFloat( parameters, "amount", 0.25f );
    expectBoundColor( parameters, "named_0_Kd", 0.2f, 0.2f, 0.2f );
    expectBoundColor( parameters, "named_1_Kd", 0.8f, 0.8f, 0.8f );
}

TEST( TestMdlBoundMaterialParameters, skipsTextureBackedInputsUntilTextureBindingExists )
{
    PbrtMaterial material{ constantUberMaterial() };
    material.params.AddTexture( "Kd", "albedo" );

    const std::vector<MdlBoundMaterialParameter> parameters{ makeMdlBoundMaterialParameters( material ) };

    EXPECT_EQ( nullptr, findBoundParameter( parameters, "Kd" ) );
    expectBoundColor( parameters, "Ks", 0.5f, 0.6f, 0.7f );
}

TEST( TestMdlGeneratedSource, mapsMirrorMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( mirrorMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: mirror" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kr = color(1.0, 1.0, 1.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input Kr: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "scattering: ::df::specular_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "mode: ::df::scatter_reflect" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:imagemap" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "reflectance.exr" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsGlassMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( glassMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: glass" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kr = color(1.0, 1.0, 1.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kt = color(1.0, 1.0, 1.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float index = 1.5" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float roughness = 0.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float uroughness = 0.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float vroughness = 0.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input Kr: Kr" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input Kt: Kt" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material gap: rough glass microfacet behavior is not "
                                                       "implemented; roughness inputs are bound but the MDL glass "
                                                       "lobe is specular" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "ior: color(index, index, index)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "scattering: ::df::tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "::df::specular_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "mode: ::df::scatter_reflect_transmit" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "pbrt_glass_approximation_tint" ) ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "diffuse_reflection_bsdf" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsMetalMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( metalMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: metal" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color eta = color(0.2, 0.2, 0.2)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color k = color(3.0, 3.0, 3.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float roughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float uroughness = -1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float vroughness = -1.0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input eta: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input k: texture_1()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material gap: PBRT-exact spectral conductor behavior "
                                                       "is approximated" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material gap: PBRT-exact roughness/remapping "
                                                       "behavior is approximated" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_metal_resolved_roughness" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_metal_conductor_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "scattering: ::df::microfacet_ggx_smith_bsdf" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "roughness_u: pbrt_metal_resolved_roughness(roughness, "
                                                       "uroughness)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "roughness_v: pbrt_metal_resolved_roughness(roughness, "
                                                       "vroughness)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: pbrt_metal_conductor_tint(texture_0(), texture_1())" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "mode: ::df::scatter_reflect" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "pbrt_metal_approximation_tint" ) ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "diffuse_reflection_bsdf" ) ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:imagemap" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "eta.exr" ) ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "k.exr" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsSubstrateMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( substrateMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: substrate" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kd = color(0.5, 0.5, 0.5)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Ks = color(0.5, 0.5, 0.5)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float roughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float uroughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float vroughness = 0.1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input bumpmap: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color pbrt_substrate_approximation_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: pbrt_substrate_approximation_tint(Kd, Ks, roughness)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: float:imagemap" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "height.exr" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsTranslucentMaterialModel )
{
    const GeneratedMdlSource generated{ generateMdlSource( translucentMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: translucent" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Kd = color(0.8, 0.8, 0.8)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color Ks = color(0.0, 0.0, 0.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color reflect = color(0.5, 0.5, 0.5)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color transmit = color(0.5, 0.5, 0.5)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input transmit: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input opacity: opacity; "
                                                       "texture=texture_1()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color pbrt_translucent_approximation_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_translucent_approximation_tint(Kd, Ks, reflect, "
                                                       "texture_0(), roughness)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "cutout_opacity: opacity" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:imagemap" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: float:imagemap" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "leaf-transmit.exr" ) ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "leaf-opacity.exr" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsMixMaterialModelWithNamedReferences )
{
    const GeneratedMdlSource generated{ generateMdlSource( layeredMixMaterial() ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "float amount = 0.5" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input namedmaterial1: named material 0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material input namedmaterial2: named material 1" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt named material 0 model: uber" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color named_0_Kd = color(0.8, 0.8, 0.8)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt named material 0 input Kd: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt named material 1 model: translucent" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color named_1_transmit = color(0.5, 0.5, 0.5)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt named material 1 input transmit: texture_1()" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color pbrt_named_material_0_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color pbrt_named_material_1_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color pbrt_mix_approximation_tint" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_named_material_0_tint((texture_0() + named_0_Ks" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "pbrt_named_material_1_tint((named_1_Kd + "
                                                       "named_1_reflect + texture_1()) / 3.0)" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: spectrum:imagemap" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:imagemap" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "front.png" ) ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "back-transmit.exr" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, recordsExplicitUnsupportedMaterialGapPolicies )
{
    struct ExplicitGapExpectation
    {
        const char* type;
        const char* coverageReason;
    };

    const ExplicitGapExpectation gapExpectations[] = {
        { "fourier",
          "low-frequency PBRT corpus material; no current target scene or reference fixture requires approximation or "
          "baking" },
        { "hair",
          "low-frequency PBRT corpus material; no current target scene or reference fixture requires approximation" },
        { "subsurface",
          "low-frequency PBRT corpus material; no current target scene or reference fixture requires approximation or "
          "baking" },
        { "kdsubsurface",
          "distinct low-frequency subsurface parameterization; no current target scene or reference fixture requires "
          "support" },
        { "measured",
          "PBRT parity completeness gap; current corpus sample did not find a target scene requiring support" },
    };

    for( const ExplicitGapExpectation& gap : gapExpectations )
    {
        SCOPED_TRACE( gap.type );

        const GeneratedMdlSource generated{ generateMdlSource( materialOfType( gap.type ) ) };
        const std::string        reason{ std::string( "Explicit PBRT material gap " ) + gap.type
                                         + ": unsupported with visible fallback" };

        EXPECT_THAT( generated.unsupportedReasons, testing::ElementsAre( reason ) );
        EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material model: " + std::string( gap.type ) ) );
        EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt material gap policy: unsupported with visible "
                                                           "fallback" ) );
        const std::string coveragePrefix{ "// pbrt material gap coverage: " };
        const std::string coverageComment{ coveragePrefix + gap.coverageReason };
        EXPECT_THAT( generated.source, testing::HasSubstr( coverageComment ) );
        EXPECT_THAT( generated.source, testing::HasSubstr( "// unsupported: " + reason ) );
        EXPECT_THAT( generated.source, testing::HasSubstr( "tint: color(1.0, 0.0, 1.0)" ) );
    }
}

TEST( TestMdlGeneratedSource, mapsScaleTextureNode )
{
    PbrtMaterial material{ materialWithKdTexture( "scaled" ) };
    material.graph.textures["color:scaled"]    = scaleTexture( "scaled", "albedo" );
    material.graph.textures["spectrum:albedo"] = imageMapTexture( "albedo", "base.png" );

    const GeneratedMdlSource generated{ generateMdlSource( material ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: spectrum:imagemap" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:scale" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_1() = pbrt_demand_texture_2d(0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_0() = texture_1() * color(1.0, 1.0, 1.0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "base.png" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, mapsMixAndCheckerboardTextureNodes )
{
    PbrtMaterial material{ materialWithKdTexture( "mixed" ) };
    material.graph.textures["color:mixed"]    = mixTexture( "mixed", "constant", "checker" );
    material.graph.textures["color:constant"] = constantTexture( "constant" );
    material.graph.textures["color:checker"]  = checkerboardTexture( "checker", 0.25f, 0.75f );

    const GeneratedMdlSource generated{ generateMdlSource( material ) };

    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:constant" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: spectrum:checkerboard" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:mix" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_2() = pbrt_demand_texture_2d(0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_0() = texture_1() * (1.0 - 0.5) + texture_2() * "
                                                       "0.5;" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: texture_0()" ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
}

TEST( TestMdlGeneratedSource, recordsUnsupportedProceduralTextureNodes )
{
    PbrtMaterial material{ materialWithKdTexture( "marble" ) };
    material.graph.textures["color:marble"] = unsupportedTexture( "marble", "color", "marble" );

    const GeneratedMdlSource generated{ generateMdlSource( material ) };

    EXPECT_THAT( generated.unsupportedReasons, testing::ElementsAre( "Unsupported PBRT texture type color:marble" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// unsupported: Unsupported PBRT texture type color:marble" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "// pbrt texture node: color:marble" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_0() = pbrt_unsupported_texture();" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: texture_0()" ) );
}

TEST( TestMdlGeneratedSourceCache, materialSourceCacheIgnoresTextureParameterValues )
{
    MdlGeneratedSourceCache cache;

    const GeneratedMdlSource& first{ cache.getSource( texturedMatteMaterial( "imagemap", "first.png" ) ) };
    const GeneratedMdlSource& second{ cache.getSource( texturedMatteMaterial( "imagemap", "second.png" ) ) };

    EXPECT_EQ( 1U, cache.size() );
    EXPECT_EQ( first.moduleName, second.moduleName );
    EXPECT_EQ( first.materialName, second.materialName );
    EXPECT_EQ( first.source, second.source );
    EXPECT_THAT( first.source, testing::Not( testing::HasSubstr( "first.png" ) ) );
    EXPECT_THAT( first.source, testing::Not( testing::HasSubstr( "second.png" ) ) );
}

TEST( TestMdlShaderCompileCache, lookupCreatesMissingRecordWithoutQueueingCompile )
{
    MdlShaderCompileCache        cache;
    const PbrtMaterial           material{ matteMaterial( 0.1f ) };
    const MdlMaterialInstanceKey key{ makeMdlMaterialInstanceKey( material ) };

    const MdlShaderCompileRecord& record{ cache.getRecord( key ) };
    EXPECT_EQ( MdlShaderCompileState::MISSING, record.state );
    EXPECT_EQ( makeMdlShaderKey( material ), record.sourceKey );
    EXPECT_EQ( 1U, record.shaderKeyId );
    EXPECT_EQ( MdlShaderCompileState::MISSING, cache.state( key ) );
    EXPECT_EQ( 1U, cache.shaderKeyId( key ) );
    EXPECT_EQ( 1U, cache.size() );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 1U, stats.numShaderRequests );
    EXPECT_EQ( 0U, stats.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.numMaterialInstanceCacheHits );
    EXPECT_EQ( 0U, stats.numCompileRequests );
    EXPECT_EQ( 0U, stats.numCompletedCompiles );
    EXPECT_EQ( 1U, stats.numMissingShaders );
    EXPECT_EQ( 0U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 0U, stats.numFailedShaders );
}

TEST( TestMdlShaderCompileCache, duplicateMaterialInstanceRequestsDoNotQueueDuplicateCompiles )
{
    MdlShaderCompileCache        cache;
    const MdlMaterialInstanceKey key{ makeMdlMaterialInstanceKey( matteMaterial( 0.1f ) ) };
    const MdlMaterialInstanceKey duplicateKey{ makeMdlMaterialInstanceKey( matteMaterial( 0.1f ) ) };

    EXPECT_TRUE( cache.requestCompile( key ) );
    EXPECT_FALSE( cache.requestCompile( duplicateKey ) );
    EXPECT_EQ( MdlShaderCompileState::QUEUED, cache.state( key ) );
    EXPECT_EQ( 1U, cache.shaderKeyId( duplicateKey ) );
    EXPECT_EQ( 1U, cache.size() );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 0U, stats.numShaderRequests );
    EXPECT_EQ( 1U, stats.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.numSourceCacheHits );
    EXPECT_EQ( 1U, stats.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.numCompileRequests );
    EXPECT_EQ( 0U, stats.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.numMissingShaders );
    EXPECT_EQ( 1U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 0U, stats.numFailedShaders );
}

TEST( TestMdlShaderCompileCache, differentBoundValuesReuseSourceButQueueDistinctCompiles )
{
    MdlShaderCompileCache        cache;
    const MdlMaterialInstanceKey firstKey{ makeMdlMaterialInstanceKey( matteMaterial( 0.1f ) ) };
    const MdlMaterialInstanceKey secondKey{ makeMdlMaterialInstanceKey( matteMaterial( 0.9f ) ) };

    ASSERT_EQ( firstKey.sourceKey, secondKey.sourceKey );
    ASSERT_NE( firstKey, secondKey );

    EXPECT_TRUE( cache.requestCompile( firstKey ) );
    EXPECT_TRUE( cache.requestCompile( secondKey ) );
    EXPECT_EQ( MdlShaderCompileState::QUEUED, cache.state( firstKey ) );
    EXPECT_EQ( MdlShaderCompileState::QUEUED, cache.state( secondKey ) );
    EXPECT_NE( cache.shaderKeyId( firstKey ), cache.shaderKeyId( secondKey ) );
    EXPECT_EQ( 2U, cache.size() );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 0U, stats.numShaderRequests );
    EXPECT_EQ( 1U, stats.numShaderCacheHits );
    EXPECT_EQ( 1U, stats.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.numMaterialInstanceCacheHits );
    EXPECT_EQ( 2U, stats.numCompileRequests );
    EXPECT_EQ( 0U, stats.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.numMissingShaders );
    EXPECT_EQ( 2U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 0U, stats.numFailedShaders );
}

TEST( TestMdlShaderCompileCache, tracksCompileStateTransitions )
{
    MdlShaderCompileCache        cache;
    const MdlMaterialInstanceKey firstKey{ makeMdlMaterialInstanceKey( matteMaterial( 0.1f ) ) };
    const MdlMaterialInstanceKey secondKey{ makeMdlMaterialInstanceKey( texturedMatteMaterial( "checkerboard", "" ) ) };

    EXPECT_TRUE( cache.requestCompile( firstKey ) );
    EXPECT_TRUE( cache.requestCompile( secondKey ) );
    cache.markCompiling( firstKey );
    cache.markReady( firstKey );

    EXPECT_EQ( MdlShaderCompileState::READY, cache.state( firstKey ) );
    EXPECT_EQ( MdlShaderCompileState::QUEUED, cache.state( secondKey ) );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 0U, stats.numShaderRequests );
    EXPECT_EQ( 0U, stats.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.numMaterialInstanceCacheHits );
    EXPECT_EQ( 2U, stats.numCompileRequests );
    EXPECT_EQ( 1U, stats.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.numMissingShaders );
    EXPECT_EQ( 1U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 1U, stats.numReadyShaders );
    EXPECT_EQ( 0U, stats.numFailedShaders );
}

TEST( TestMdlShaderCompileCache, failedCompilesStayCachedWithDiagnostics )
{
    MdlShaderCompileCache        cache;
    const MdlMaterialInstanceKey key{ makeMdlMaterialInstanceKey( matteMaterial( 0.1f ) ) };

    EXPECT_TRUE( cache.requestCompile( key ) );
    cache.markCompiling( key );
    cache.markFailed( key, "mdl compile failed" );

    EXPECT_FALSE( cache.requestCompile( key ) );
    EXPECT_EQ( MdlShaderCompileState::FAILED, cache.state( key ) );
    EXPECT_EQ( "mdl compile failed", cache.diagnostics( key ) );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 0U, stats.numShaderRequests );
    EXPECT_EQ( 1U, stats.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.numSourceCacheHits );
    EXPECT_EQ( 1U, stats.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.numCompileRequests );
    EXPECT_EQ( 0U, stats.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.numMissingShaders );
    EXPECT_EQ( 0U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 1U, stats.numFailedShaders );
}

#endif  // OTK_USE_MDL
