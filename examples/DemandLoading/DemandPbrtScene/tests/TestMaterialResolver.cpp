// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/Config.h>
#include <DemandPbrtScene/MaterialResolver.h>

#include <DemandPbrtScene/Testing/GeometryInstancePrinter.h>
#include <DemandPbrtScene/Testing/MockDemandTextureCache.h>
#include <DemandPbrtScene/Testing/MockProgramGroups.h>
#include <DemandPbrtScene/Testing/ParamsPrinters.h>

#include <DemandPbrtScene/DemandTextureCache.h>
#include <DemandPbrtScene/FrameStopwatch.h>
#include <DemandPbrtScene/MaterialAdapters.h>
#include <DemandPbrtScene/Options.h>
#include <DemandPbrtScene/Primitive.h>
#include <DemandPbrtScene/ProgramGroups.h>
#include <DemandPbrtScene/Scene.h>
#include <DemandPbrtScene/SceneGeometry.h>
#include <DemandPbrtScene/SceneProxy.h>
#include <DemandPbrtScene/SceneSyncState.h>

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/DemandMaterial/Testing/MockMaterialLoader.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/Testing/Matchers.h>

#include <vector_types.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef OTK_USE_MDL
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#endif
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

constexpr const char* ALPHA_MAP_PATH{ "alphaMap.png" };
constexpr const char* DIFFUSE_MAP_PATH{ "diffuseMap.png" };
#ifdef OTK_USE_MDL
constexpr const char* BUMP_MAP_PATH{ "bumpMap.png" };
#endif
constexpr const char* SPECULAR_MAP_PATH{ "specularMap.png" };
constexpr const char* REFLECTANCE_MAP_PATH{ "reflectanceMap.png" };
constexpr int         ARBITRARY_PRIMITIVE_INDEX_END{ 654 };
constexpr int         ARBITRARY_PRIMITIVE_INDEX_END2{ 765 };

using namespace testing;
using namespace otk::testing;
using namespace demandPbrtScene;
using namespace demandPbrtScene::testing;

namespace {

PhongMaterial arbitraryPhongMaterial()
{
    PhongMaterial result{};
    result.Ka       = make_float3( 1.0f, 2.0f, 3.0f );
    result.Kd       = make_float3( 4.0f, 5.0f, 6.0f );
    result.Ks       = make_float3( 7.0f, 8.0f, 9.0f );
    result.Kr       = make_float3( 10.0f, 11.0f, 12.0f );
    result.phongExp = 13.4f;
    return result;
}

PhongMaterial arbitraryOtherPhongMaterial()
{
    PhongMaterial result{ arbitraryPhongMaterial() };
    result.Kd = make_float3( 3.0f, 2.0f, 1.0f );
    return result;
}

PhongMaterial arbitraryThirdPhongMaterial()
{
    PhongMaterial result{ arbitraryPhongMaterial() };
    result.Ka = make_float3( 3.0f, 2.0f, 1.0f );
    return result;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMaterialOfType( const std::string& type )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = type;
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtTexturedMaterialOfType( const std::string& type, const std::string& textureParam )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = type;
    material->params.AddTexture( textureParam, "pbrt-texture" );
    return material;
}

void addString( ::pbrt::ParamSet& params, const std::string& name, const std::string& value )
{
    std::unique_ptr<std::string[]> values{ new std::string[1] };
    values[0] = value;
    params.AddString( name, std::move( values ), 1 );
}

#ifdef OTK_USE_MDL
std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtFourierMaterialWithBsdfFile( const std::string& bsdfFile )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "fourier";
    addString( material->params, "bsdffile", bsdfFile );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtNamedFourierMaterialWithBsdfFile( const std::string& bsdfFile )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type              = "fourier";
    material->namedMaterialName = "measuredGold";
    addString( material->params, "type", "fourier" );
    addString( material->params, "bsdffile", bsdfFile );

    otk::pbrt::PbrtNamedMaterial namedMaterial{};
    namedMaterial.name = "measuredGold";
    namedMaterial.type = "fourier";
    addString( namedMaterial.params, "type", "fourier" );
    addString( namedMaterial.params, "bsdffile", bsdfFile );
    material->graph.namedMaterials[namedMaterial.name] = std::move( namedMaterial );
    return material;
}
#endif

void addFloat( ::pbrt::ParamSet& params, const std::string& name, float value )
{
    std::unique_ptr<::pbrt::Float[]> values{ new ::pbrt::Float[1] };
    values[0] = value;
    params.AddFloat( name, std::move( values ), 1 );
}

void addRgbSpectrum( ::pbrt::ParamSet& params, const std::string& name, float red, float green, float blue )
{
    std::unique_ptr<::pbrt::Float[]> values{ new ::pbrt::Float[3] };
    values[0] = red;
    values[1] = green;
    values[2] = blue;
    params.AddRGBSpectrum( name, std::move( values ), 3 );
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtImagemapMaterialOfType( const std::string& type,
                                                                           const std::string& textureParam,
                                                                           const std::string& valueType,
                                                                           const std::string& fileName )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = type;
    material->params.AddTexture( textureParam, "pbrt-texture" );
    otk::pbrt::PbrtTexture texture{};
    texture.name      = "pbrt-texture";
    texture.valueType = valueType;
    texture.type      = "imagemap";
    addString( texture.params, "filename", fileName );
    material->graph.textures[valueType + ":pbrt-texture"] = std::move( texture );
    return material;
}

void addPbrtImagemapTexture( otk::pbrt::PbrtMaterial& material,
                             const std::string&       textureParam,
                             const std::string&       valueType,
                             const std::string&       textureName,
                             const std::string&       fileName )
{
    material.params.AddTexture( textureParam, textureName );
    otk::pbrt::PbrtTexture texture{};
    texture.name      = textureName;
    texture.valueType = valueType;
    texture.type      = "imagemap";
    addString( texture.params, "filename", fileName );
    material.graph.textures[valueType + ":" + textureName] = std::move( texture );
}

#ifdef OTK_USE_MDL
void addPbrtNamedImagemapTexture( otk::pbrt::PbrtMaterial&      owner,
                                  otk::pbrt::PbrtNamedMaterial& material,
                                  const std::string&            textureParam,
                                  const std::string&            valueType,
                                  const std::string&            textureName,
                                  const std::string&            fileName )
{
    material.params.AddTexture( textureParam, textureName );
    otk::pbrt::PbrtTexture texture{};
    texture.name      = textureName;
    texture.valueType = valueType;
    texture.type      = "imagemap";
    addString( texture.params, "filename", fileName );
    owner.graph.textures[valueType + ":" + textureName] = std::move( texture );
}
#endif

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtUberSpecularImagemapMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "uber";
    addPbrtImagemapTexture( *material, "Kd", "spectrum", "albedo", DIFFUSE_MAP_PATH );
    addPbrtImagemapTexture( *material, "Ks", "spectrum", "specular", SPECULAR_MAP_PATH );
    addPbrtImagemapTexture( *material, "Kr", "spectrum", "reflectance", REFLECTANCE_MAP_PATH );
    addFloat( material->params, "roughness", 0.5f );
    addFloat( material->params, "index", 1.333f );
    return material;
}

#ifdef OTK_USE_MDL
std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtLandscapeMixNamedBranchImagemapMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addFloat( material->params, "amount", 0.4f );

    otk::pbrt::PbrtNamedMaterial front{};
    front.name = "front";
    front.type = "uber";
    addString( front.params, "type", "uber" );
    addFloat( front.params, "roughness", 0.5f );
    addFloat( front.params, "index", 1.333f );
    addRgbSpectrum( front.params, "Kt", 0.0f, 0.0f, 0.0f );
    addPbrtNamedImagemapTexture( *material, front, "Kd", "spectrum", "front-albedo", DIFFUSE_MAP_PATH );
    addPbrtNamedImagemapTexture( *material, front, "Ks", "spectrum", "front-specular", SPECULAR_MAP_PATH );
    addPbrtNamedImagemapTexture( *material, front, "Kr", "spectrum", "front-reflectance", REFLECTANCE_MAP_PATH );
    addPbrtNamedImagemapTexture( *material, front, "alpha", "float", "front-alpha", ALPHA_MAP_PATH );
    addPbrtNamedImagemapTexture( *material, front, "bumpmap", "float", "front-height", BUMP_MAP_PATH );

    otk::pbrt::PbrtNamedMaterial back{};
    back.name = "back";
    back.type = "translucent";
    addString( back.params, "type", "translucent" );
    addRgbSpectrum( back.params, "Ks", 0.0f, 0.0f, 0.0f );
    addRgbSpectrum( back.params, "reflect", 0.25f, 0.25f, 0.25f );
    addRgbSpectrum( back.params, "transmit", 0.75f, 0.75f, 0.75f );
    addFloat( back.params, "roughness", 0.5f );
    addPbrtNamedImagemapTexture( *material, back, "Kd", "spectrum", "back-albedo", "back-diffuse.png" );
    addPbrtNamedImagemapTexture( *material, back, "opacity", "float", "back-alpha", "back-alpha.png" );
    addPbrtNamedImagemapTexture( *material, back, "bumpmap", "float", "back-height", "back-bump.png" );

    material->graph.namedMaterials["front"] = std::move( front );
    material->graph.namedMaterials["back"]  = std::move( back );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtTranslucentDiffuseImagemapMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "translucent";
    addPbrtImagemapTexture( *material, "Kd", "spectrum", "albedo", DIFFUSE_MAP_PATH );
    addRgbSpectrum( material->params, "Ks", 0.0f, 0.0f, 0.0f );
    addRgbSpectrum( material->params, "reflect", 0.25f, 0.25f, 0.25f );
    addRgbSpectrum( material->params, "transmit", 0.75f, 0.75f, 0.75f );
    addFloat( material->params, "roughness", 0.5f );
    return material;
}
#endif

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtScaledImagemapMaterialOfType( const std::string& type,
                                                                                 const std::string& textureParam,
                                                                                 const std::string& valueType,
                                                                                 const std::string& fileName,
                                                                                 const float3&      scale )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = type;
    material->params.AddTexture( textureParam, "pbrt-scaled-texture" );
    otk::pbrt::PbrtTexture imageTexture{};
    imageTexture.name      = "pbrt-texture";
    imageTexture.valueType = valueType;
    imageTexture.type      = "imagemap";
    addString( imageTexture.params, "filename", fileName );
    material->graph.textures[valueType + ":pbrt-texture"] = std::move( imageTexture );
    otk::pbrt::PbrtTexture scaleTexture{};
    scaleTexture.name      = "pbrt-scaled-texture";
    scaleTexture.valueType = valueType;
    scaleTexture.type      = "scale";
    scaleTexture.params.AddTexture( "tex1", "pbrt-texture" );
    addRgbSpectrum( scaleTexture.params, "tex2", scale.x, scale.y, scale.z );
    material->graph.textures[valueType + ":pbrt-scaled-texture"] = std::move( scaleTexture );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMixedImagemapMaterialOfType( const std::string& type,
                                                                                const std::string& textureParam,
                                                                                const std::string& valueType,
                                                                                const std::string& fileName )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = type;
    material->params.AddTexture( textureParam, "pbrt-mixed-texture" );
    otk::pbrt::PbrtTexture imageTexture{};
    imageTexture.name      = "pbrt-texture";
    imageTexture.valueType = valueType;
    imageTexture.type      = "imagemap";
    addString( imageTexture.params, "filename", fileName );
    material->graph.textures[valueType + ":pbrt-texture"] = std::move( imageTexture );
    otk::pbrt::PbrtTexture constantTexture{};
    constantTexture.name      = "constant-texture";
    constantTexture.valueType = valueType;
    constantTexture.type      = "constant";
    addRgbSpectrum( constantTexture.params, "value", 0.25f, 0.5f, 0.75f );
    material->graph.textures[valueType + ":constant-texture"] = std::move( constantTexture );
    otk::pbrt::PbrtTexture mixTexture{};
    mixTexture.name      = "pbrt-mixed-texture";
    mixTexture.valueType = valueType;
    mixTexture.type      = "mix";
    mixTexture.params.AddTexture( "tex1", "pbrt-texture" );
    mixTexture.params.AddTexture( "tex2", "constant-texture" );
    addFloat( mixTexture.params, "amount", 0.5f );
    material->graph.textures[valueType + ":pbrt-mixed-texture"] = std::move( mixTexture );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMixedImagemapMaterialWithImagemapAmountOfType( const std::string& type,
                                                                                                  const std::string& textureParam,
                                                                                                  const std::string& valueType,
                                                                                                  const std::string& fileName )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = type;
    material->params.AddTexture( textureParam, "pbrt-mixed-texture" );
    otk::pbrt::PbrtTexture imageTexture{};
    imageTexture.name      = "pbrt-texture";
    imageTexture.valueType = valueType;
    imageTexture.type      = "imagemap";
    addString( imageTexture.params, "filename", fileName );
    material->graph.textures[valueType + ":pbrt-texture"] = std::move( imageTexture );
    otk::pbrt::PbrtTexture amountTexture{};
    amountTexture.name      = "amount-texture";
    amountTexture.valueType = "float";
    amountTexture.type      = "imagemap";
    addString( amountTexture.params, "filename", "amount.png" );
    material->graph.textures["float:amount-texture"] = std::move( amountTexture );
    otk::pbrt::PbrtTexture mixTexture{};
    mixTexture.name      = "pbrt-mixed-texture";
    mixTexture.valueType = valueType;
    mixTexture.type      = "mix";
    mixTexture.params.AddTexture( "tex1", "pbrt-texture" );
    addRgbSpectrum( mixTexture.params, "tex2", 0.25f, 0.5f, 0.75f );
    mixTexture.params.AddTexture( "amount", "amount-texture" );
    material->graph.textures[valueType + ":pbrt-mixed-texture"] = std::move( mixTexture );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMatteMaterial( float red )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    std::unique_ptr<::pbrt::Float[]>         kd{ new ::pbrt::Float[3] };
    kd[0]          = red;
    kd[1]          = 0.0f;
    kd[2]          = 0.0f;
    material->type = "matte";
    material->params.AddRGBSpectrum( "Kd", std::move( kd ), 3 );
    return material;
}

otk::pbrt::PbrtNamedMaterial pbrtNamedMatteMaterial( const std::string& name, float red )
{
    otk::pbrt::PbrtNamedMaterial material{};
    material.name = name;
    material.type = "matte";
    addString( material.params, "type", "matte" );
    std::unique_ptr<::pbrt::Float[]> kd{ new ::pbrt::Float[3] };
    kd[0] = red;
    kd[1] = 0.0f;
    kd[2] = 0.0f;
    material.params.AddRGBSpectrum( "Kd", std::move( kd ), 3 );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMixMaterial( float amount )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addFloat( material->params, "amount", amount );
    material->graph.namedMaterials["front"] = pbrtNamedMatteMaterial( "front", 0.2f );
    material->graph.namedMaterials["back"]  = pbrtNamedMatteMaterial( "back", 0.8f );
    return material;
}

#ifdef OTK_USE_MDL
std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMixMaterialWithRgbAmount( float red, float green, float blue )
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addRgbSpectrum( material->params, "amount", red, green, blue );
    material->graph.namedMaterials["front"] = pbrtNamedMatteMaterial( "front", 0.2f );
    material->graph.namedMaterials["back"]  = pbrtNamedMatteMaterial( "back", 0.8f );
    return material;
}
#endif

otk::pbrt::PbrtTexture pbrtConstantFloatTexture( const std::string& name, float value )
{
    otk::pbrt::PbrtTexture texture{};
    texture.name      = name;
    texture.valueType = "float";
    texture.type      = "constant";
    addFloat( texture.params, "value", value );
    return texture;
}

otk::pbrt::PbrtTexture pbrtConstantColorTexture( const std::string& name, float red, float green, float blue )
{
    otk::pbrt::PbrtTexture texture{};
    texture.name      = name;
    texture.valueType = "color";
    texture.type      = "constant";
    addRgbSpectrum( texture.params, "value", red, green, blue );
    return texture;
}

otk::pbrt::PbrtTexture pbrtImagemapFloatTexture( const std::string& name, const std::string& fileName )
{
    otk::pbrt::PbrtTexture texture{};
    texture.name      = name;
    texture.valueType = "float";
    texture.type      = "imagemap";
    addString( texture.params, "filename", fileName );
    return texture;
}

otk::pbrt::PbrtTexture pbrtScaleFloatTexture( const std::string& name, const std::string& tex1, float tex2 )
{
    otk::pbrt::PbrtTexture texture{};
    texture.name      = name;
    texture.valueType = "float";
    texture.type      = "scale";
    texture.params.AddTexture( "tex1", tex1 );
    addFloat( texture.params, "tex2", tex2 );
    return texture;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMixMaterialWithConstantAmountTexture()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    material->params.AddTexture( "amount", "weight" );
    material->graph.namedMaterials["front"]  = pbrtNamedMatteMaterial( "front", 0.2f );
    material->graph.namedMaterials["back"]   = pbrtNamedMatteMaterial( "back", 0.8f );
    material->graph.textures["float:weight"] = pbrtConstantFloatTexture( "weight", 0.25f );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMatteMaterialWithConstantKdTexture()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "matte";
    material->params.AddTexture( "Kd", "albedo" );
    material->graph.textures["color:albedo"] = pbrtConstantColorTexture( "albedo", 0.25f, 0.5f, 0.75f );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtMixMaterialWithImagemapAmountTexture()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    material->params.AddTexture( "amount", "weight" );
    material->graph.namedMaterials["front"]  = pbrtNamedMatteMaterial( "front", 0.2f );
    material->graph.namedMaterials["back"]   = pbrtNamedMatteMaterial( "back", 0.8f );
    material->graph.textures["float:weight"] = pbrtImagemapFloatTexture( "weight", "mix-weight.exr" );
    return material;
}

#ifdef OTK_USE_MDL
std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtUberBumpImagemapMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "uber";
    addPbrtImagemapTexture( *material, "bumpmap", "float", "height", BUMP_MAP_PATH );
    addFloat( material->params, "roughness", 0.5f );
    addFloat( material->params, "index", 1.333f );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtUberTwoLeafBumpImagemapMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "uber";
    material->params.AddTexture( "bumpmap", "mixed-height" );
    material->graph.textures["float:height-a"] = pbrtImagemapFloatTexture( "height-a", BUMP_MAP_PATH );
    material->graph.textures["float:height-b"] = pbrtImagemapFloatTexture( "height-b", "otherBump.png" );
    otk::pbrt::PbrtTexture mixTexture{};
    mixTexture.name      = "mixed-height";
    mixTexture.valueType = "float";
    mixTexture.type      = "mix";
    mixTexture.params.AddTexture( "tex1", "height-a" );
    mixTexture.params.AddTexture( "tex2", "height-b" );
    addFloat( mixTexture.params, "amount", 0.5f );
    material->graph.textures["float:mixed-height"] = std::move( mixTexture );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtLandscapeMixWithTexturedTransmitBranchMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addFloat( material->params, "amount", 0.4f );
    material->graph.namedMaterials["front"] = pbrtNamedMatteMaterial( "front", 0.2f );

    otk::pbrt::PbrtNamedMaterial back{};
    back.name = "back";
    back.type = "translucent";
    addString( back.params, "type", "translucent" );
    addRgbSpectrum( back.params, "Kd", 0.2f, 0.3f, 0.4f );
    addRgbSpectrum( back.params, "reflect", 0.25f, 0.25f, 0.25f );
    addPbrtNamedImagemapTexture( *material, back, "transmit", "color", "back-transmit", "back-transmit.png" );
    material->graph.namedMaterials["back"] = std::move( back );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtLandscapeMixWithScaledBranchBumpMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addFloat( material->params, "amount", 0.4f );
    material->graph.namedMaterials["back"] = pbrtNamedMatteMaterial( "back", 0.8f );

    otk::pbrt::PbrtNamedMaterial front{};
    front.name = "front";
    front.type = "uber";
    addString( front.params, "type", "uber" );
    addRgbSpectrum( front.params, "Kd", 0.2f, 0.3f, 0.4f );
    front.params.AddTexture( "bumpmap", "scaled-front-height" );
    material->graph.textures["float:front-height"] = pbrtImagemapFloatTexture( "front-height", BUMP_MAP_PATH );
    material->graph.textures["float:scaled-front-height"] = pbrtScaleFloatTexture( "scaled-front-height", "front-height", 0.5f );
    material->graph.namedMaterials["front"] = std::move( front );
    return material;
}

std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtLandscapeMixWithTwoLeafBranchTextureMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addFloat( material->params, "amount", 0.4f );
    material->graph.namedMaterials["back"] = pbrtNamedMatteMaterial( "back", 0.8f );

    otk::pbrt::PbrtNamedMaterial front{};
    front.name = "front";
    front.type = "uber";
    addString( front.params, "type", "uber" );
    front.params.AddTexture( "Kd", "mixed-front" );
    material->graph.textures["spectrum:front-a"]           = pbrtImagemapFloatTexture( "front-a", DIFFUSE_MAP_PATH );
    material->graph.textures["spectrum:front-a"].valueType = "spectrum";
    material->graph.textures["spectrum:front-b"]           = pbrtImagemapFloatTexture( "front-b", "other-diffuse.png" );
    material->graph.textures["spectrum:front-b"].valueType = "spectrum";
    otk::pbrt::PbrtTexture mixTexture{};
    mixTexture.name      = "mixed-front";
    mixTexture.valueType = "spectrum";
    mixTexture.type      = "mix";
    mixTexture.params.AddTexture( "tex1", "front-a" );
    mixTexture.params.AddTexture( "tex2", "front-b" );
    addFloat( mixTexture.params, "amount", 0.5f );
    material->graph.textures["spectrum:mixed-front"] = std::move( mixTexture );
    material->graph.namedMaterials["front"]          = std::move( front );
    return material;
}
#endif

#ifdef OTK_USE_MDL
std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtLandscapeMixWithTwoLeafBranchAlphaMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addFloat( material->params, "amount", 0.4f );
    material->graph.namedMaterials["back"] = pbrtNamedMatteMaterial( "back", 0.8f );

    otk::pbrt::PbrtNamedMaterial front{};
    front.name = "front";
    front.type = "uber";
    addString( front.params, "type", "uber" );
    front.params.AddTexture( "alpha", "mixed-front-alpha" );
    material->graph.textures["float:front-alpha-a"] = pbrtImagemapFloatTexture( "front-alpha-a", ALPHA_MAP_PATH );
    material->graph.textures["float:front-alpha-b"] = pbrtImagemapFloatTexture( "front-alpha-b", "other-alpha.png" );
    otk::pbrt::PbrtTexture mixTexture{};
    mixTexture.name      = "mixed-front-alpha";
    mixTexture.valueType = "float";
    mixTexture.type      = "mix";
    mixTexture.params.AddTexture( "tex1", "front-alpha-a" );
    mixTexture.params.AddTexture( "tex2", "front-alpha-b" );
    addFloat( mixTexture.params, "amount", 0.5f );
    material->graph.textures["float:mixed-front-alpha"] = std::move( mixTexture );
    material->graph.namedMaterials["front"]             = std::move( front );
    return material;
}
std::shared_ptr<const otk::pbrt::PbrtMaterial> pbrtLandscapeMixWithTwoLeafBranchBumpMaterial()
{
    std::shared_ptr<otk::pbrt::PbrtMaterial> material{ std::make_shared<otk::pbrt::PbrtMaterial>() };
    material->type = "mix";
    addString( material->params, "namedmaterial1", "front" );
    addString( material->params, "namedmaterial2", "back" );
    addFloat( material->params, "amount", 0.4f );
    material->graph.namedMaterials["back"] = pbrtNamedMatteMaterial( "back", 0.8f );

    otk::pbrt::PbrtNamedMaterial front{};
    front.name = "front";
    front.type = "uber";
    addString( front.params, "type", "uber" );
    front.params.AddTexture( "bumpmap", "mixed-front-height" );
    material->graph.textures["float:front-height-a"] = pbrtImagemapFloatTexture( "front-height-a", BUMP_MAP_PATH );
    material->graph.textures["float:front-height-b"] = pbrtImagemapFloatTexture( "front-height-b", "other-bump.png" );
    otk::pbrt::PbrtTexture mixTexture{};
    mixTexture.name      = "mixed-front-height";
    mixTexture.valueType = "float";
    mixTexture.type      = "mix";
    mixTexture.params.AddTexture( "tex1", "front-height-a" );
    mixTexture.params.AddTexture( "tex2", "front-height-b" );
    addFloat( mixTexture.params, "amount", 0.5f );
    material->graph.textures["float:mixed-front-height"] = std::move( mixTexture );
    material->graph.namedMaterials["front"]              = std::move( front );
    return material;
}
#endif

void usePbrtMaterialOfType( GeometryInstance& geom, const std::string& type )
{
    geom.groups[0].pbrtMaterial = pbrtMaterialOfType( type );
}

void usePbrtTexturedMaterialOfType( GeometryInstance& geom, const std::string& type, const std::string& textureParam )
{
    geom.groups[0].pbrtMaterial = pbrtTexturedMaterialOfType( type, textureParam );
}

#ifdef OTK_USE_MDL
void usePbrtFourierMaterialWithBsdfFile( GeometryInstance& geom, const std::string& bsdfFile )
{
    geom.groups[0].pbrtMaterial = pbrtFourierMaterialWithBsdfFile( bsdfFile );
}

void usePbrtNamedFourierMaterialWithBsdfFile( GeometryInstance& geom, const std::string& bsdfFile )
{
    geom.groups[0].pbrtMaterial = pbrtNamedFourierMaterialWithBsdfFile( bsdfFile );
}
#endif

void usePbrtDiffuseImagemapMaterial( GeometryInstance& geom, const std::string& type = "matte" )
{
    geom.groups[0].pbrtMaterial       = pbrtImagemapMaterialOfType( type, "Kd", "spectrum", DIFFUSE_MAP_PATH );
    geom.groups[0].material.flags     = MaterialFlags::DIFFUSE_MAP;
    geom.groups[0].diffuseMapFileName = pbrtColorTextureBinding( *geom.groups[0].pbrtMaterial, "Kd" ).fileName;
}

void usePbrtScaledDiffuseImagemapMaterial( GeometryInstance& geom, const float3& scale )
{
    geom.groups[0].pbrtMaterial   = pbrtScaledImagemapMaterialOfType( "matte", "Kd", "color", DIFFUSE_MAP_PATH, scale );
    geom.groups[0].material.Kd    = scale;
    geom.groups[0].material.flags = MaterialFlags::DIFFUSE_MAP;
    geom.groups[0].diffuseMapFileName = pbrtColorTextureBinding( *geom.groups[0].pbrtMaterial, "Kd" ).fileName;
}

void usePbrtMixedDiffuseImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial       = pbrtMixedImagemapMaterialOfType( "matte", "Kd", "color", DIFFUSE_MAP_PATH );
    geom.groups[0].material.Kd        = make_float3( 0.5f, 0.5f, 0.5f );
    geom.groups[0].material.flags     = MaterialFlags::DIFFUSE_MAP;
    geom.groups[0].diffuseMapFileName = pbrtColorTextureBinding( *geom.groups[0].pbrtMaterial, "Kd" ).fileName;
}

void usePbrtUberSpecularImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial       = pbrtUberSpecularImagemapMaterial();
    geom.groups[0].material.flags     = MaterialFlags::DIFFUSE_MAP;
    geom.groups[0].diffuseMapFileName = pbrtColorTextureBinding( *geom.groups[0].pbrtMaterial, "Kd" ).fileName;
}

#ifdef OTK_USE_MDL
void usePbrtTransmissionImagemapMaterial( GeometryInstance& geom, const std::string& type )
{
    geom.groups[0].pbrtMaterial = pbrtImagemapMaterialOfType( type, "Kt", "spectrum", DIFFUSE_MAP_PATH );
}

void usePbrtTranslucentDiffuseImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial       = pbrtTranslucentDiffuseImagemapMaterial();
    geom.groups[0].material.flags     = MaterialFlags::DIFFUSE_MAP;
    geom.groups[0].diffuseMapFileName = pbrtColorTextureBinding( *geom.groups[0].pbrtMaterial, "Kd" ).fileName;
}

void usePbrtUberBumpImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtUberBumpImagemapMaterial();
}

void usePbrtLandscapeMixNamedBranchImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtLandscapeMixNamedBranchImagemapMaterial();
}

void usePbrtLandscapeMixWithTexturedTransmitBranchMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtLandscapeMixWithTexturedTransmitBranchMaterial();
}

void usePbrtLandscapeMixWithScaledBranchBumpMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtLandscapeMixWithScaledBranchBumpMaterial();
}

void usePbrtLandscapeMixWithTwoLeafBranchTextureMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtLandscapeMixWithTwoLeafBranchTextureMaterial();
}

void usePbrtLandscapeMixWithTwoLeafBranchAlphaMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtLandscapeMixWithTwoLeafBranchAlphaMaterial();
}

void usePbrtLandscapeMixWithTwoLeafBranchBumpMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtLandscapeMixWithTwoLeafBranchBumpMaterial();
}

void usePbrtUberTwoLeafBumpImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtUberTwoLeafBumpImagemapMaterial();
}
#endif

void usePbrtDynamicAmountMixedDiffuseImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtMixedImagemapMaterialWithImagemapAmountOfType( "matte", "Kd", "color", DIFFUSE_MAP_PATH );
}

void usePbrtMirrorReflectanceImagemapMaterial( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial       = pbrtImagemapMaterialOfType( "mirror", "Kr", "color", DIFFUSE_MAP_PATH );
    geom.groups[0].material.flags     = MaterialFlags::DIFFUSE_MAP;
    geom.groups[0].diffuseMapFileName = pbrtColorTextureBinding( *geom.groups[0].pbrtMaterial, "Kr" ).fileName;
}

void usePbrtAlphaImagemapMaterial( GeometryInstance& geom, const std::string& type = "matte" )
{
    geom.groups[0].pbrtMaterial     = pbrtImagemapMaterialOfType( type, "opacity", "float", ALPHA_MAP_PATH );
    geom.groups[0].material.flags   = MaterialFlags::ALPHA_MAP;
    geom.groups[0].alphaMapFileName = ALPHA_MAP_PATH;
}

void usePbrtMatteMaterial( GeometryInstance& geom, float red )
{
    geom.groups[0].pbrtMaterial = pbrtMatteMaterial( red );
}

void usePbrtMixMaterial( GeometryInstance& geom, float amount )
{
    geom.groups[0].pbrtMaterial = pbrtMixMaterial( amount );
}

#ifdef OTK_USE_MDL
void usePbrtMixMaterialWithRgbAmount( GeometryInstance& geom, float red, float green, float blue )
{
    geom.groups[0].pbrtMaterial = pbrtMixMaterialWithRgbAmount( red, green, blue );
}
#endif

void usePbrtMixMaterialWithConstantAmountTexture( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtMixMaterialWithConstantAmountTexture();
}

void usePbrtMatteMaterialWithConstantKdTexture( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtMatteMaterialWithConstantKdTexture();
}

void usePbrtMixMaterialWithImagemapAmountTexture( GeometryInstance& geom )
{
    geom.groups[0].pbrtMaterial = pbrtMixMaterialWithImagemapAmountTexture();
}

MaterialState localFallbackState( uint_t materialId )
{
    return makeMaterialState( materialId, MaterialBackend::LOCAL_FALLBACK );
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

MaterialState mdlPendingState( uint_t materialId, uint_t shaderKeyId )
{
    return makeMaterialState( materialId, MaterialBackend::MDL_PENDING, shaderKeyId );
}

MaterialState mdlFailedState( uint_t materialId, uint_t shaderKeyId )
{
    return makeMaterialState( materialId, MaterialBackend::MDL_FAILED, shaderKeyId );
}

void expectMdlMaterialShader( const SceneSyncState& sync, uint_t materialId, const MdlMaterialShader& expected )
{
    ASSERT_LT( materialId, sync.mdlMaterialShaders.size() );
    EXPECT_EQ( expected, sync.mdlMaterialShaders[materialId] );
}
#endif

void setLocalFallbackState( SceneSyncState& sync, uint_t materialId )
{
    if( sync.materialStates.size() <= materialId )
    {
        sync.materialStates.resize( materialId + 1 );
    }
    sync.materialStates[materialId] = localFallbackState( materialId );
}

inline ListenerPredicate<GeometryInstance> hasMaterialFlags( MaterialFlags value )
{
    return [=]( MatchResultListener* listener, const GeometryInstance& arg ) {
        return hasEqualValues( listener, "flags", value, arg.groups[0].material.flags );
    };
}

inline ListenerPredicate<GeometryInstance> hasDiffuseTextureId( uint_t value )
{
    return [=]( MatchResultListener* listener, const GeometryInstance& arg ) {
        return hasEqualValues( listener, "diffuse texture id", value, arg.groups[0].material.diffuseTextureId );
    };
}

#ifdef OTK_USE_MDL
inline ListenerPredicate<GeometryInstance> hasMdlTextureBinding( uint_t index, uint_t textureId, const float3& scale, const float3& bias )
{
    return [=]( MatchResultListener* listener, const GeometryInstance& arg ) {
        if( arg.groups[0].mdlTextureBindings.size() <= index )
        {
            *listener << "mdl texture binding index " << index << " exceeds table size";
            return false;
        }
        const MdlMaterialTextureBinding& binding{ arg.groups[0].mdlTextureBindings[index].binding };
        return hasEqualValues( listener, "mdl texture id", textureId, binding.textureId )
               && hasEqualValues( listener, "mdl texture scale", scale, binding.scale )
               && hasEqualValues( listener, "mdl texture bias", bias, binding.bias );
    };
}
#endif

inline ListenerPredicate<GeometryInstance> hasAlphaTextureId( uint_t value )
{
    return [=]( MatchResultListener* listener, const GeometryInstance& arg ) {
        return hasEqualValues( listener, "alpha texture id", value, arg.groups[0].material.alphaTextureId );
    };
}

MATCHER_P( hasGeometryInstance, predicate, "" )
{
    return predicate( result_listener, arg );
}

// This was needed to satisfy gcc instead of constructing from a brace initializer list.
Options testOptions()
{
    Options options{};
    options.program   = "DemandPbrtScene";
    options.sceneFile = "test.pbrt";
    options.outFile   = "out.png";
#ifdef OTK_USE_MDL
    options.mdlSynchronousCompilation = true;
#endif
    return options;
}

#ifdef OTK_USE_MDL
std::filesystem::path makeFourierTestDirectory()
{
    const std::filesystem::path directory{ std::filesystem::temp_directory_path() / "DemandPbrtSceneFourierResolver" };
    std::filesystem::create_directories( directory / "bsdfs" );
    return directory;
}

void writeFourierUint32( std::ostream& output, std::uint32_t value )
{
    const unsigned char bytes[] = {
        static_cast<unsigned char>( value & 0xffU ),
        static_cast<unsigned char>( ( value >> 8 ) & 0xffU ),
        static_cast<unsigned char>( ( value >> 16 ) & 0xffU ),
        static_cast<unsigned char>( ( value >> 24 ) & 0xffU ),
    };
    output.write( reinterpret_cast<const char*>( bytes ), sizeof( bytes ) );
}

void writeFourierInt32( std::ostream& output, int value )
{
    writeFourierUint32( output, static_cast<std::uint32_t>( value ) );
}

void writeFourierFloat( std::ostream& output, float value )
{
    std::uint32_t bits{};
    std::memcpy( &bits, &value, sizeof( bits ) );
    writeFourierUint32( output, bits );
}

void writeMinimalFourierBsdfTable( const std::filesystem::path& fileName )
{
    constexpr char scatfunHeader[8] = { 'S', 'C', 'A', 'T', 'F', 'U', 'N', '\x01' };
    std::ofstream  output{ fileName, std::ios::binary };
    output.write( scatfunHeader, sizeof( scatfunHeader ) );
    writeFourierInt32( output, 1 );
    writeFourierInt32( output, 1 );
    writeFourierInt32( output, 3 );
    writeFourierInt32( output, 1 );
    writeFourierInt32( output, 3 );
    writeFourierInt32( output, 1 );
    for( int i = 0; i < 3; ++i )
    {
        writeFourierInt32( output, 0 );
    }
    writeFourierFloat( output, 1.0f );
    for( int i = 0; i < 4; ++i )
    {
        writeFourierInt32( output, 0 );
    }
    writeFourierFloat( output, 1.0f );
    writeFourierFloat( output, 1.0f );
    writeFourierInt32( output, 0 );
    writeFourierInt32( output, 1 );
    writeFourierFloat( output, 0.1f );
    writeFourierFloat( output, 0.2f );
    writeFourierFloat( output, 0.3f );
}

void writeInvalidFourierBsdfTable( const std::filesystem::path& fileName )
{
    std::ofstream output{ fileName, std::ios::binary };
    output.write( "NOTBSDF!", 8 );
}

void expectNoMdlShadersCompiled( const MaterialResolverStats& stats )
{
    EXPECT_EQ( 0U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 0U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.numFourierBsdfTableResourcesResolved );
    EXPECT_EQ( 0U, stats.numFourierBsdfTableResourcesMissing );
    EXPECT_EQ( 0U, stats.numFourierBsdfTableResourcesInvalid );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numMissingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numFailedShaders );
}

void expectGeneratedFourierFallbackStats( const MaterialResolverStats& stats,
                                          unsigned int                 expectedResolvedTables,
                                          unsigned int                 expectedMissingTables,
                                          unsigned int                 expectedInvalidTables = 0U )
{
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( expectedResolvedTables, stats.numFourierBsdfTableResourcesResolved );
    EXPECT_EQ( expectedMissingTables, stats.numFourierBsdfTableResourcesMissing );
    EXPECT_EQ( expectedInvalidTables, stats.numFourierBsdfTableResourcesInvalid );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}
#endif

using MockMaterialLoaderPtr = std::shared_ptr<MockMaterialLoader>;

class TestMaterialResolver : public Test
{
  public:
    ~TestMaterialResolver() override = default;

  protected:
    Options                   m_options{ testOptions() };
    MockMaterialLoaderPtr     m_loader{ std::make_shared<MockMaterialLoader>() };
    MockDemandTextureCachePtr m_demandTextureCache{ createMockDemandTextureCache() };
    MockProgramGroupsPtr      m_programGroups{ createMockProgramGroups() };
    MaterialResolverPtr m_resolver{ createMaterialResolver( m_options, m_loader, m_demandTextureCache, m_programGroups ) };
    SceneSyncState m_sync{};
};

class TestMaterialResolverForGeometry : public TestMaterialResolver
{
  public:
    ~TestMaterialResolverForGeometry() override = default;

  protected:
    void SetUp() override;

    GeometryInstance m_geom{};
};

void TestMaterialResolverForGeometry::SetUp()
{
    m_geom.primitive                  = GeometryPrimitive::TRIANGLE;
    m_geom.instance.traversableHandle = 0xbaadbeefU;
    m_geom.groups.push_back( MaterialGroup{ arbitraryPhongMaterial(), {}, {}, ARBITRARY_PRIMITIVE_INDEX_END } );
}

class TestMaterialResolverRequestedProxyIds : public TestMaterialResolverForGeometry
{
  public:
    ~TestMaterialResolverRequestedProxyIds() override = default;

  protected:
    void SetUp() override;

    CUstream       m_stream{};
    FrameStopwatch m_timer{ false };
};

void TestMaterialResolverRequestedProxyIds::SetUp()
{
    TestMaterialResolverForGeometry::SetUp();
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
}

class TestMaterialResolverRequestedProxyIdsGroups : public TestMaterialResolverRequestedProxyIds
{
  public:
    ~TestMaterialResolverRequestedProxyIdsGroups() override = default;

  protected:
    void SetUp() override;
};

void TestMaterialResolverRequestedProxyIdsGroups::SetUp()
{
    TestMaterialResolverRequestedProxyIds::SetUp();
    m_geom.groups.push_back( MaterialGroup{ arbitraryOtherPhongMaterial(), {}, {}, ARBITRARY_PRIMITIVE_INDEX_END2 } );
}

}  // namespace

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyPhongMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( 0U, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
    ASSERT_FALSE( m_sync.materialIndices.empty() );
    ASSERT_FALSE( m_sync.primitiveMaterials.empty() );
    EXPECT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( 1U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( MaterialIndex{ 1, 0 } ), m_sync.materialIndices[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId } ), m_sync.primitiveMaterials[0] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
}

#ifdef OTK_USE_MDL
TEST_F( TestMaterialResolverForGeometry, resolveNewProxyGeneratedMatteMaterialForGeometryDoesNotCompileShader )
{
    m_options.useMdlMaterials = true;
    usePbrtMaterialOfType( m_geom, "matte" );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    expectNoMdlShadersCompiled( m_resolver->getStatistics() );
}
#endif

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyPhongMaterialsForCoarseGeometry )
{
    const uint_t         proxyGeomId{ 1111U };
    const uint_t         proxyMaterialId1{ 4444U };
    const uint_t         proxyMaterialId2{ 5555U };
    const ExpectationSet firstMaterial{ EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ) };
    EXPECT_CALL( *m_loader, add() ).After( firstMaterial ).WillOnce( Return( proxyMaterialId2 ) );
    m_geom.groups.push_back( MaterialGroup{ arbitraryOtherPhongMaterial(), {}, {}, ARBITRARY_PRIMITIVE_INDEX_END2 } );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_FALSE( m_sync.topLevelInstances.empty() );
    EXPECT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
    ASSERT_FALSE( m_sync.materialIndices.empty() );
    EXPECT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 2, 0 } ), m_sync.materialIndices[0] );
    ASSERT_FALSE( m_sync.primitiveMaterials.empty() );
    EXPECT_EQ( 2U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId1 } ), m_sync.primitiveMaterials[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, proxyMaterialId2 } ), m_sync.primitiveMaterials[1] );
    ASSERT_LT( proxyMaterialId2, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId1 ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( localFallbackState( proxyMaterialId2 ), m_sync.materialStates[proxyMaterialId2] );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyDiffuseMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups[0].material.flags     = MaterialFlags::DIFFUSE_MAP;
    m_geom.groups[0].diffuseMapFileName = DIFFUSE_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( 0U, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
    ASSERT_FALSE( m_sync.materialIndices.empty() );
    ASSERT_FALSE( m_sync.primitiveMaterials.empty() );
    EXPECT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 1, 0 } ), m_sync.materialIndices[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId } ), m_sync.primitiveMaterials[0] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyAlphaCutOutMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups[0].material.flags   = MaterialFlags::ALPHA_MAP;
    m_geom.groups[0].alphaMapFileName = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( 0U, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
    ASSERT_FALSE( m_sync.materialIndices.empty() );
    ASSERT_FALSE( m_sync.primitiveMaterials.empty() );
    EXPECT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 1, 0 } ), m_sync.materialIndices[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId } ), m_sync.primitiveMaterials[0] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyDiffuseAlphaCutOutMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups[0].material.flags     = MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP;
    m_geom.groups[0].diffuseMapFileName = DIFFUSE_MAP_PATH;
    m_geom.groups[0].alphaMapFileName   = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( 0U, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
    ASSERT_FALSE( m_sync.materialIndices.empty() );
    ASSERT_FALSE( m_sync.primitiveMaterials.empty() );
    EXPECT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 1, 0 } ), m_sync.materialIndices[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId } ), m_sync.primitiveMaterials[0] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
}

TEST_F( TestMaterialResolverForGeometry, resolveSharedPhongMaterialForGeometry )
{
    const uint_t        proxyGeomId{ 1111 };
    const PhongMaterial existingMaterial{ arbitraryPhongMaterial() };
    const uint_t        existingMaterialId{ 1 };
    const PhongMaterial otherMaterial{ arbitraryOtherPhongMaterial() };
    const uint_t        otherMaterialId{ 0 };
    m_sync.realizedMaterials.push_back( otherMaterial );
    m_sync.realizedMaterials.push_back( existingMaterial );
    setLocalFallbackState( m_sync, otherMaterialId );
    setLocalFallbackState( m_sync, existingMaterialId );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, existingMaterialId } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, existingMaterialId } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, existingMaterialId } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, otherMaterialId } );
    m_sync.materialIndices.push_back( MaterialIndex{ 1U, 0U } );
    m_sync.materialIndices.push_back( MaterialIndex{ 1U, 1U } );
    m_sync.materialIndices.push_back( MaterialIndex{ 2U, 2U } );
    EXPECT_CALL( *m_loader, add() ).Times( 0 );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_TRUE( result );
    ASSERT_EQ( 4U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( 3U, m_sync.topLevelInstances.back().instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances.back().traversableHandle );
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, m_sync.topLevelInstances.back().sbtOffset );
    ASSERT_EQ( 4U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 1U, 4U } ), m_sync.materialIndices[3] );
    ASSERT_EQ( 5U, m_sync.primitiveMaterials.size() );
    ASSERT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, existingMaterialId } ), m_sync.primitiveMaterials[4] );
}

TEST_F( TestMaterialResolverForGeometry, resolveSharedSparseMaterialIdForGeometry )
{
    const uint_t        proxyGeomId{ 1111 };
    const PhongMaterial existingMaterial{ arbitraryPhongMaterial() };
    const uint_t        existingMaterialId{ 4444U };
    m_sync.realizedMaterials.resize( existingMaterialId + 1 );
    m_sync.realizedMaterials[existingMaterialId] = existingMaterial;
    setLocalFallbackState( m_sync, existingMaterialId );
    EXPECT_CALL( *m_loader, add() ).Times( 0 );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_TRUE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( 0U, m_sync.topLevelInstances.back().instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances.back().traversableHandle );
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, m_sync.topLevelInstances.back().sbtOffset );
    ASSERT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 1U, 0U } ), m_sync.materialIndices[0] );
    ASSERT_EQ( 1U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, existingMaterialId } ), m_sync.primitiveMaterials[0] );
}

TEST_F( TestMaterialResolverForGeometry, resolveOneProxyOneSharedMaterialForCoarseGeometry )
{
    const uint_t         proxyGeomId{ 1111U };
    const uint_t         proxyMaterialId1{ 4444U };
    const uint_t         proxyMaterialId2{ 5555U };
    const ExpectationSet first{ EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ) };
    EXPECT_CALL( *m_loader, add() ).After( first ).WillOnce( Return( proxyMaterialId2 ) );
    m_geom.groups.push_back( MaterialGroup{ arbitraryThirdPhongMaterial(), {}, {}, ARBITRARY_PRIMITIVE_INDEX_END2 } );
    const PhongMaterial existingMaterial{ arbitraryPhongMaterial() };
    const uint_t        existingMaterialId{ 1 };
    const PhongMaterial otherExistingMaterial{ arbitraryOtherPhongMaterial() };
    const uint_t        otherMaterialId{ 0 };
    m_sync.realizedMaterials.push_back( otherExistingMaterial );
    m_sync.realizedMaterials.push_back( existingMaterial );
    setLocalFallbackState( m_sync, otherMaterialId );
    setLocalFallbackState( m_sync, existingMaterialId );
    m_sync.materialIndices.push_back( MaterialIndex{ 1, 0 } );
    m_sync.materialIndices.push_back( MaterialIndex{ 1, 1 } );
    m_sync.materialIndices.push_back( MaterialIndex{ 1, 2 } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, otherMaterialId } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, otherMaterialId } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, existingMaterialId } );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).Times( 0 );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 4U, m_sync.topLevelInstances.size() );
    // TODO: instance id is always index into per-GAS data
    //EXPECT_EQ( 3U, m_sync.topLevelInstances.back().instanceId );
    ASSERT_EQ( 4U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 2, 3 } ), m_sync.materialIndices[3] );
    ASSERT_EQ( 5U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId1 } ), m_sync.primitiveMaterials[3] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, proxyMaterialId2 } ), m_sync.primitiveMaterials[4] );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances.back().traversableHandle );
}

TEST_F( TestMaterialResolverForGeometry, resolveSharedMaterialsForCoarseGeometry )
{
    const uint_t proxyGeomId{ 1111U };
    EXPECT_CALL( *m_loader, add() ).Times( 0 );
    m_geom.groups.push_back( MaterialGroup{ arbitraryOtherPhongMaterial(), {}, {}, ARBITRARY_PRIMITIVE_INDEX_END2 } );
    const PhongMaterial existingMaterial{ arbitraryPhongMaterial() };
    const uint_t        existingMaterialIndex{ 0U };
    const PhongMaterial otherExistingMaterial{ arbitraryOtherPhongMaterial() };
    const uint_t        otherMaterialIndex{ 1U };
    m_sync.realizedMaterials.push_back( existingMaterial );
    m_sync.realizedMaterials.push_back( otherExistingMaterial );
    setLocalFallbackState( m_sync, existingMaterialIndex );
    setLocalFallbackState( m_sync, otherMaterialIndex );
    m_sync.materialIndices.push_back( MaterialIndex{ 1, 0 } );
    m_sync.materialIndices.push_back( MaterialIndex{ 1, 1 } );
    m_sync.materialIndices.push_back( MaterialIndex{ 1, 2 } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, otherMaterialIndex } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, otherMaterialIndex } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, existingMaterialIndex } );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_TRUE( result );
    ASSERT_EQ( 4U, m_sync.topLevelInstances.size() );
    // TODO: instance id is always index into per-GAS data
    //EXPECT_EQ( 3U, m_sync.topLevelInstances.back().instanceId );
    ASSERT_EQ( 4U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 2, 3 } ), m_sync.materialIndices[3] );
    ASSERT_EQ( 5U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, existingMaterialIndex } ), m_sync.primitiveMaterials[3] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, otherMaterialIndex } ), m_sync.primitiveMaterials[4] );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances.back().traversableHandle );
}

TEST_F( TestMaterialResolverForGeometry, coarseGeometryRetainsTextureIdentityForEqualPhongMaterials )
{
    const uint_t        proxyGeomId{ 1111U };
    const uint_t        firstMaterialIndex{ 0U };
    const uint_t        secondMaterialIndex{ 1U };
    const uint_t        firstDiffuseTextureId{ 101U };
    const uint_t        secondDiffuseTextureId{ 102U };
    const uint_t        firstAlphaTextureId{ 201U };
    const uint_t        secondAlphaTextureId{ 202U };
    const char* const   secondDiffuseMapPath{ "diffuseMap2.png" };
    const char* const   secondAlphaMapPath{ "alphaMap2.png" };
    const MaterialFlags mapFlags{ MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP };
    const MaterialFlags realizedMapFlags{ mapFlags | MaterialFlags::ALPHA_MAP_ALLOCATED | MaterialFlags::DIFFUSE_MAP_ALLOCATED };

    m_geom.groups[0].material.flags     = mapFlags;
    m_geom.groups[0].diffuseMapFileName = DIFFUSE_MAP_PATH;
    m_geom.groups[0].alphaMapFileName   = ALPHA_MAP_PATH;
    m_geom.groups.push_back( MaterialGroup{ arbitraryPhongMaterial(), secondDiffuseMapPath, secondAlphaMapPath,
                                            ARBITRARY_PRIMITIVE_INDEX_END2 } );
    m_geom.groups[1].material.flags = mapFlags;

    PhongMaterial firstMaterial{ arbitraryPhongMaterial() };
    firstMaterial.flags            = realizedMapFlags;
    firstMaterial.diffuseTextureId = firstDiffuseTextureId;
    firstMaterial.alphaTextureId   = firstAlphaTextureId;
    PhongMaterial secondMaterial{ arbitraryPhongMaterial() };
    secondMaterial.flags            = realizedMapFlags;
    secondMaterial.diffuseTextureId = secondDiffuseTextureId;
    secondMaterial.alphaTextureId   = secondAlphaTextureId;
    m_sync.realizedMaterials.push_back( firstMaterial );
    m_sync.realizedMaterials.push_back( secondMaterial );
    setLocalFallbackState( m_sync, firstMaterialIndex );
    setLocalFallbackState( m_sync, secondMaterialIndex );
    m_sync.materialIndices.push_back( MaterialIndex{ 1U, 0U } );
    m_sync.materialIndices.push_back( MaterialIndex{ 1U, 1U } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, firstMaterialIndex } );
    m_sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, secondMaterialIndex } );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );

    EXPECT_CALL( *m_loader, add() ).Times( 0 );
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( secondDiffuseMapPath ) ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( secondAlphaMapPath ) ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( DIFFUSE_MAP_PATH ) ) )
        .Times( AnyNumber() )
        .WillRepeatedly( Return( firstDiffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( StrEq( ALPHA_MAP_PATH ) ) ).Times( AnyNumber() ).WillRepeatedly( Return( firstAlphaTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( secondDiffuseMapPath ) ) )
        .Times( AnyNumber() )
        .WillRepeatedly( Return( secondDiffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( StrEq( secondAlphaMapPath ) ) )
        .Times( AnyNumber() )
        .WillRepeatedly( Return( secondAlphaTextureId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_TRUE( result );
    ASSERT_EQ( 3U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 2U, 2U } ), m_sync.materialIndices.back() );
    ASSERT_EQ( 4U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, firstMaterialIndex } ), m_sync.primitiveMaterials[2] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, secondMaterialIndex } ), m_sync.primitiveMaterials[3] );
}

TEST_F( TestMaterialResolverRequestedProxyIds, noRequestedProxyMaterials )
{
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{} ) );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result );
}

#ifdef OTK_USE_MDL
TEST_F( TestMaterialResolverRequestedProxyIds, unrequestedGeneratedMatteMaterialDoesNotCompileShader )
{
    m_options.useMdlMaterials = true;
    usePbrtMaterialOfType( m_geom, "matte" );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{} ) );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result );
    expectNoMdlShadersCompiled( m_resolver->getStatistics() );
}
#endif

TEST_F( TestMaterialResolverRequestedProxyIds, resolvePhongMaterial )
{
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_EQ( proxyMaterialId + 1, m_sync.realizedMaterials.size() );
    EXPECT_EQ( arbitraryPhongMaterial(), m_sync.realizedMaterials[proxyMaterialId] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    EXPECT_EQ( 1U, m_sync.realizedNormals.size() );
    EXPECT_EQ( 1U, m_sync.realizedUVs.size() );
    const OptixInstance& instance{ m_sync.topLevelInstances.back() };
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, instance.sbtOffset );
    EXPECT_EQ( 0U, instance.instanceId );
    ASSERT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 1U, 0U } ), m_sync.materialIndices[0] );
    ASSERT_EQ( 1U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId } ), m_sync.primitiveMaterials[0] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numMaterialsRealized );
}

#ifdef OTK_USE_MDL
TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMatteMaterialCompilesShaderOnce )
{
    m_options.useMdlMaterials = true;
    usePbrtMaterialOfType( m_geom, "matte" );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 11U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, MdlMaterialShader{ 8U, 1U } );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 0U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numMissingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numFailedShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMatteMaterialFoldsConstantKdTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtMatteMaterialWithConstantKdTexture( m_geom );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 11U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( hasGeometryInstance( hasMaterialFlags( MaterialFlags::NONE ) ) ) )
        .WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( hasGeometryInstance( hasMaterialFlags( MaterialFlags::NONE ) ), 1U ) )
        .WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( MaterialFlags::NONE, m_sync.realizedMaterials[proxyMaterialId].flags );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, MdlMaterialShader{ 8U, 1U } );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 0U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedOpaqueMaterialFamiliesCompileShaders )
{
    m_options.useMdlMaterials = true;
    GeometryInstance plasticGeom{ m_geom };
    GeometryInstance uberGeom{ m_geom };
    GeometryInstance substrateGeom{ m_geom };
    usePbrtMaterialOfType( plasticGeom, "plastic" );
    usePbrtMaterialOfType( uberGeom, "uber" );
    usePbrtMaterialOfType( substrateGeom, "substrate" );
    const uint_t proxyGeomId1{ 1111U };
    const uint_t proxyGeomId2{ 2222U };
    const uint_t proxyGeomId3{ 3333U };
    const uint_t proxyMaterialId1{ 4444U };
    const uint_t proxyMaterialId2{ 5555U };
    const uint_t proxyMaterialId3{ 6666U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 11U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ).WillOnce( Return( proxyMaterialId2 ) ).WillOnce( Return( proxyMaterialId3 ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId1, plasticGeom, m_sync ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId2, uberGeom, m_sync ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId3, substrateGeom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId1, proxyMaterialId2, proxyMaterialId3 } ) );
    {
        InSequence sequence;
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId1 ) ).Times( 1 );
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 2U ) ).WillOnce( Return( MdlMaterialShader{ 9U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId2 ) ).Times( 1 );
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 3U ) ).WillOnce( Return( MdlMaterialShader{ 10U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId3 ) ).Times( 1 );
    }
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId1, m_sync.materialStates.size() );
    ASSERT_LT( proxyMaterialId2, m_sync.materialStates.size() );
    ASSERT_LT( proxyMaterialId3, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId1, 1U ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( mdlReadyState( proxyMaterialId2, 2U ), m_sync.materialStates[proxyMaterialId2] );
    EXPECT_EQ( mdlReadyState( proxyMaterialId3, 3U ), m_sync.materialStates[proxyMaterialId3] );
    expectMdlMaterialShader( m_sync, proxyMaterialId1, MdlMaterialShader{ 8U, 1U } );
    expectMdlMaterialShader( m_sync, proxyMaterialId2, MdlMaterialShader{ 9U, 1U } );
    expectMdlMaterialShader( m_sync, proxyMaterialId3, MdlMaterialShader{ 10U, 1U } );
    ASSERT_EQ( 3U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[0].sbtOffset );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[1].sbtOffset );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[2].sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 3U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 3U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 3U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 3U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 3U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedSpecularMaterialFamiliesCompileShaders )
{
    m_options.useMdlMaterials = true;
    GeometryInstance mirrorGeom{ m_geom };
    GeometryInstance glassGeom{ m_geom };
    GeometryInstance metalGeom{ m_geom };
    usePbrtMaterialOfType( mirrorGeom, "mirror" );
    usePbrtMaterialOfType( glassGeom, "glass" );
    usePbrtMaterialOfType( metalGeom, "metal" );
    const uint_t proxyGeomId1{ 1111U };
    const uint_t proxyGeomId2{ 2222U };
    const uint_t proxyGeomId3{ 3333U };
    const uint_t proxyMaterialId1{ 4444U };
    const uint_t proxyMaterialId2{ 5555U };
    const uint_t proxyMaterialId3{ 6666U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 11U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ).WillOnce( Return( proxyMaterialId2 ) ).WillOnce( Return( proxyMaterialId3 ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId1, mirrorGeom, m_sync ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId2, glassGeom, m_sync ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId3, metalGeom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId1, proxyMaterialId2, proxyMaterialId3 } ) );
    {
        InSequence sequence;
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId1 ) ).Times( 1 );
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 2U ) ).WillOnce( Return( MdlMaterialShader{ 9U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId2 ) ).Times( 1 );
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 3U ) ).WillOnce( Return( MdlMaterialShader{ 10U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId3 ) ).Times( 1 );
    }
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId1, m_sync.materialStates.size() );
    ASSERT_LT( proxyMaterialId2, m_sync.materialStates.size() );
    ASSERT_LT( proxyMaterialId3, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId1, 1U ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( mdlReadyState( proxyMaterialId2, 2U ), m_sync.materialStates[proxyMaterialId2] );
    EXPECT_EQ( mdlReadyState( proxyMaterialId3, 3U ), m_sync.materialStates[proxyMaterialId3] );
    expectMdlMaterialShader( m_sync, proxyMaterialId1, MdlMaterialShader{ 8U, 1U } );
    expectMdlMaterialShader( m_sync, proxyMaterialId2, MdlMaterialShader{ 9U, 1U } );
    expectMdlMaterialShader( m_sync, proxyMaterialId3, MdlMaterialShader{ 10U, 1U } );
    ASSERT_EQ( 3U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[0].sbtOffset );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[1].sbtOffset );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[2].sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 3U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 3U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 3U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 3U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 3U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMixAndTranslucentMaterialsCompileShaders )
{
    m_options.useMdlMaterials = true;
    GeometryInstance mixGeom{ m_geom };
    GeometryInstance translucentGeom{ m_geom };
    usePbrtMixMaterial( mixGeom, 0.25f );
    usePbrtMaterialOfType( translucentGeom, "translucent" );
    const uint_t proxyGeomId1{ 1111U };
    const uint_t proxyGeomId2{ 2222U };
    const uint_t proxyMaterialId1{ 4444U };
    const uint_t proxyMaterialId2{ 5555U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 11U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ).WillOnce( Return( proxyMaterialId2 ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId1, mixGeom, m_sync ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId2, translucentGeom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId1, proxyMaterialId2 } ) );
    {
        InSequence sequence;
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId1 ) ).Times( 1 );
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 2U ) ).WillOnce( Return( MdlMaterialShader{ 9U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId2 ) ).Times( 1 );
    }
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId1, m_sync.materialStates.size() );
    ASSERT_LT( proxyMaterialId2, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId1, 1U ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( mdlReadyState( proxyMaterialId2, 2U ), m_sync.materialStates[proxyMaterialId2] );
    expectMdlMaterialShader( m_sync, proxyMaterialId1, MdlMaterialShader{ 8U, 1U } );
    expectMdlMaterialShader( m_sync, proxyMaterialId2, MdlMaterialShader{ 9U, 1U } );
    ASSERT_EQ( 2U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[0].sbtOffset );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[1].sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 2U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 2U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 2U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 2U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 2U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedLandscapeMixMaterialBindsNamedBranchDemandTextures )
{
    m_options.useMdlMaterials = true;
    usePbrtLandscapeMixNamedBranchImagemapMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t frontDiffuseTextureId{ 331U };
    const uint_t frontSpecularTextureId{ 442U };
    const uint_t frontReflectanceTextureId{ 553U };
    const uint_t frontAlphaCutoutTextureId{ 664U };
    const uint_t frontAlphaTextureId{ 775U };
    const uint_t frontBumpTextureId{ 886U };
    const uint_t backDiffuseTextureId{ 997U };
    const uint_t backAlphaTextureId{ 1108U };
    const uint_t backBumpTextureId{ 1219U };
    const uint_t stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( EndsWith( ALPHA_MAP_PATH ) ) )
        .WillOnce( Return( frontAlphaCutoutTextureId ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution partialResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::PARTIAL, partialResult );
    EXPECT_EQ( frontAlphaCutoutTextureId, m_sync.minAlphaTextureId );
    EXPECT_EQ( frontAlphaCutoutTextureId, m_sync.maxAlphaTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.partialMaterials.size() );
    ASSERT_LT( proxyMaterialId, m_sync.partialUVs.size() );
    EXPECT_EQ( frontAlphaCutoutTextureId, m_sync.partialMaterials[proxyMaterialId].alphaTextureId );
    EXPECT_EQ( fakeUVs, m_sync.partialUVs[proxyMaterialId] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );

    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( DIFFUSE_MAP_PATH ), true ) )
        .WillOnce( Return( frontDiffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( SPECULAR_MAP_PATH ), true ) )
        .WillOnce( Return( frontSpecularTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( REFLECTANCE_MAP_PATH ), true ) )
        .WillOnce( Return( frontReflectanceTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( ALPHA_MAP_PATH ), true ) )
        .WillOnce( Return( frontAlphaTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( BUMP_MAP_PATH ), true ) )
        .WillOnce( Return( frontBumpTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( "back-diffuse.png" ), true ) )
        .WillOnce( Return( backDiffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( "back-alpha.png" ), true ) )
        .WillOnce( Return( backAlphaTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( "back-bump.png" ), true ) )
        .WillOnce( Return( backBumpTextureId ) );
    MdlMaterialShader expectedShader{ 8U, 1U };
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX, frontDiffuseTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_0_KS_TEXTURE_BINDING_INDEX, frontSpecularTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_0_KR_TEXTURE_BINDING_INDEX, frontReflectanceTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_0_ALPHA_TEXTURE_BINDING_INDEX, frontAlphaTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX, frontBumpTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX, backDiffuseTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_1_ALPHA_TEXTURE_BINDING_INDEX, backAlphaTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX, backBumpTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) );
    EXPECT_CALL( *m_programGroups,
                 getMdlMaterialSbtOffset( hasGeometryInstance( hasAll(
                     hasMaterialFlags( MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED ), hasAlphaTextureId( frontAlphaCutoutTextureId ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX, frontDiffuseTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_KS_TEXTURE_BINDING_INDEX, frontSpecularTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_KR_TEXTURE_BINDING_INDEX, frontReflectanceTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_ALPHA_TEXTURE_BINDING_INDEX, frontAlphaTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX, frontBumpTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX, backDiffuseTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_1_ALPHA_TEXTURE_BINDING_INDEX, backAlphaTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                     hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX, backBumpTextureId,
                                           make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) ) ) ) )
        .WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader(
                     hasGeometryInstance( hasAll(
                         hasMaterialFlags( MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED ),
                         hasAlphaTextureId( frontAlphaCutoutTextureId ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_KD_TEXTURE_BINDING_INDEX, frontDiffuseTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_KS_TEXTURE_BINDING_INDEX, frontSpecularTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_KR_TEXTURE_BINDING_INDEX, frontReflectanceTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_ALPHA_TEXTURE_BINDING_INDEX, frontAlphaTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX, frontBumpTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_1_KD_TEXTURE_BINDING_INDEX, backDiffuseTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_1_ALPHA_TEXTURE_BINDING_INDEX, backAlphaTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ),
                         hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_1_BUMPMAP_TEXTURE_BINDING_INDEX, backBumpTextureId,
                                               make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) ) ),
                     1U ) )
        .WillOnce( Return( expectedShader ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( frontDiffuseTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( backBumpTextureId, m_sync.maxDiffuseTextureId );
    EXPECT_EQ( frontAlphaCutoutTextureId, m_sync.minAlphaTextureId );
    EXPECT_EQ( frontAlphaCutoutTextureId, m_sync.maxAlphaTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( frontAlphaCutoutTextureId, m_sync.realizedMaterials[proxyMaterialId].alphaTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) );
    EXPECT_EQ( 0U, m_sync.partialMaterials[proxyMaterialId].alphaTextureId );
    EXPECT_EQ( nullptr, m_sync.partialUVs[proxyMaterialId] );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, expectedShader );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( stableSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMixMaterialBindsScaledBranchBumpmapDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtLandscapeMixWithScaledBranchBumpMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t            proxyGeomId{ 1111U };
    const uint_t            proxyMaterialId{ 4444U };
    const uint_t            bumpTextureId{ 333U };
    const uint_t            stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const MdlMaterialShader expectedShader{ 8U, 1U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( EndsWith( BUMP_MAP_PATH ), true ) )
        .WillOnce( Return( bumpTextureId ) );
    EXPECT_CALL( *m_programGroups,
                 getMdlMaterialSbtOffset( hasGeometryInstance(
                     hasAll( hasMaterialFlags( MaterialFlags::NONE ),
                             hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX, bumpTextureId,
                                                   make_float3( 0.5f, 0.5f, 0.5f ), make_float3( 0.0f, 0.0f, 0.0f ) ) ) ) ) )
        .WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader(
                     hasGeometryInstance( hasAll( hasMaterialFlags( MaterialFlags::NONE ),
                                                  hasMdlTextureBinding( MDL_MATERIAL_MIX_NAMED_0_BUMPMAP_TEXTURE_BINDING_INDEX,
                                                                        bumpTextureId, make_float3( 0.5f, 0.5f, 0.5f ),
                                                                        make_float3( 0.0f, 0.0f, 0.0f ) ) ) ),
                     1U ) )
        .WillOnce( Return( expectedShader ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( bumpTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( bumpTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, expectedShader );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( stableSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMixMaterialWithConstantAmountTextureCompilesShader )
{
    m_options.useMdlMaterials = true;
    usePbrtMixMaterialWithConstantAmountTexture( m_geom );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, MdlMaterialShader{ 8U, 1U } );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMixMaterialWithEqualRgbAmountCompilesShader )
{
    m_options.useMdlMaterials = true;
    usePbrtMixMaterialWithRgbAmount( m_geom, 0.4f, 0.4f, 0.4f );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, MdlMaterialShader{ 8U, 1U } );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMatteMaterialsShareShaderKeyButKeepDistinctMaterialShaders )
{
    m_options.useMdlMaterials = true;
    usePbrtMatteMaterial( m_geom, 0.25f );
    GeometryInstance secondGeom{ m_geom };
    usePbrtMatteMaterial( secondGeom, 0.75f );
    const uint_t proxyGeomId1{ 1111U };
    const uint_t proxyGeomId2{ 2222U };
    const uint_t proxyMaterialId1{ 4444U };
    const uint_t proxyMaterialId2{ 5555U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 11U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ).WillOnce( Return( proxyMaterialId2 ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId1, m_geom, m_sync ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId2, secondGeom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId1, proxyMaterialId2 } ) );
    {
        InSequence sequence;
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId1 ) ).Times( 1 );
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 9U, 1U } ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId2 ) ).Times( 1 );
    }
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId1, m_sync.materialStates.size() );
    ASSERT_LT( proxyMaterialId2, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId1, 1U ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( mdlReadyState( proxyMaterialId2, 1U ), m_sync.materialStates[proxyMaterialId2] );
    expectMdlMaterialShader( m_sync, proxyMaterialId1, MdlMaterialShader{ 8U, 1U } );
    expectMdlMaterialShader( m_sync, proxyMaterialId2, MdlMaterialShader{ 9U, 1U } );
    ASSERT_EQ( 2U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[0].sbtOffset );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[1].sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 2U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 2U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMatteMaterialsShareDelayedCompileButKeepDistinctMaterialShaders )
{
    m_options.useMdlMaterials            = true;
    m_options.mdlSynchronousCompilation = false;
    usePbrtMatteMaterial( m_geom, 0.25f );
    GeometryInstance secondGeom{ m_geom };
    usePbrtMatteMaterial( secondGeom, 0.75f );
    const uint_t proxyGeomId1{ 1111U };
    const uint_t proxyGeomId2{ 2222U };
    const uint_t proxyMaterialId1{ 4444U };
    const uint_t proxyMaterialId2{ 5555U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 11U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ).WillOnce( Return( proxyMaterialId2 ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId1, m_geom, m_sync ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId2, secondGeom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId1, proxyMaterialId2 } ) );
    {
        InSequence sequence;
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId1 ) ).Times( 1 );
        EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId2 ) ).Times( 1 );
    }
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution fallbackResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, fallbackResult );
    ASSERT_LT( proxyMaterialId1, m_sync.materialStates.size() );
    ASSERT_LT( proxyMaterialId2, m_sync.materialStates.size() );
    EXPECT_EQ( mdlPendingState( proxyMaterialId1, 1U ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( mdlPendingState( proxyMaterialId2, 1U ), m_sync.materialStates[proxyMaterialId2] );
    ASSERT_EQ( 2U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[0].sbtOffset );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances[1].sbtOffset );

    MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 2U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 2U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 2U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numReadyShaders );
    ASSERT_TRUE( Mock::VerifyAndClearExpectations( m_loader.get() ) );
    ASSERT_TRUE( Mock::VerifyAndClearExpectations( m_programGroups.get() ) );

    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) )
        .WillOnce( Throw( MdlMaterialBuildPending( "still building" ) ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution pendingResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, pendingResult );
    stats = m_resolver->getStatistics();
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numReadyShaders );
    ASSERT_TRUE( Mock::VerifyAndClearExpectations( m_loader.get() ) );
    ASSERT_TRUE( Mock::VerifyAndClearExpectations( m_programGroups.get() ) );

    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution firstReadyResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::SHADER_DATA_ONLY, firstReadyResult );
    EXPECT_EQ( mdlReadyState( proxyMaterialId1, 1U ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( mdlPendingState( proxyMaterialId2, 1U ), m_sync.materialStates[proxyMaterialId2] );
    expectMdlMaterialShader( m_sync, proxyMaterialId1, MdlMaterialShader{ 8U, 1U } );
    stats = m_resolver->getStatistics();
    EXPECT_EQ( 1U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
    ASSERT_TRUE( Mock::VerifyAndClearExpectations( m_loader.get() ) );
    ASSERT_TRUE( Mock::VerifyAndClearExpectations( m_programGroups.get() ) );

    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 9U, 1U } ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution secondReadyResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::SHADER_DATA_ONLY, secondReadyResult );
    EXPECT_EQ( mdlReadyState( proxyMaterialId1, 1U ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( mdlReadyState( proxyMaterialId2, 1U ), m_sync.materialStates[proxyMaterialId2] );
    expectMdlMaterialShader( m_sync, proxyMaterialId2, MdlMaterialShader{ 9U, 1U } );
    stats = m_resolver->getStatistics();
    EXPECT_EQ( 1U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numFailedShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksExplicitMaterialGapTypesOnFallback )
{
    m_options.useMdlMaterials = true;
    const char* const   gapTypes[]  = { "fourier", "hair", "subsurface", "kdsubsurface", "measured" };
    const uint_t        firstProxyGeomId{ 1111U };
    const uint_t        firstProxyMaterialId{ 4444U };
    std::vector<uint_t> requestedMaterialIds;

    EXPECT_CALL( *m_loader, add() )
        .WillOnce( Return( firstProxyMaterialId ) )
        .WillOnce( Return( firstProxyMaterialId + 1U ) )
        .WillOnce( Return( firstProxyMaterialId + 2U ) )
        .WillOnce( Return( firstProxyMaterialId + 3U ) )
        .WillOnce( Return( firstProxyMaterialId + 4U ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).Times( 5 ).WillRepeatedly( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    for( size_t index = 0; index < 5U; ++index )
    {
        GeometryInstance geom{ m_geom };
        usePbrtMaterialOfType( geom, gapTypes[index] );
        const uint_t proxyMaterialId{ firstProxyMaterialId + static_cast<uint_t>( index ) };

        SCOPED_TRACE( gapTypes[index] );
        ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( firstProxyGeomId + static_cast<uint_t>( index ), geom, m_sync ) );
        requestedMaterialIds.push_back( proxyMaterialId );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    }

    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( requestedMaterialIds ) );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    for( size_t index = 0; index < requestedMaterialIds.size(); ++index )
    {
        const uint_t proxyMaterialId{ requestedMaterialIds[index] };
        SCOPED_TRACE( gapTypes[index] );
        ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
        EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    }
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 5U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 5U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.numFourierBsdfTableResourcesResolved );
    EXPECT_EQ( 1U, stats.numFourierBsdfTableResourcesMissing );
    EXPECT_EQ( 0U, stats.numFourierBsdfTableResourcesInvalid );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeResolvesFourierBsdfTableResourceBeforeFallback )
{
    const std::filesystem::path sceneDirectory{ makeFourierTestDirectory() };
    const std::filesystem::path bsdfFile{ sceneDirectory / "direct-fourier.bsdf" };
    writeMinimalFourierBsdfTable( bsdfFile );
    ASSERT_TRUE( std::filesystem::exists( bsdfFile ) );

    m_options.useMdlMaterials = true;
    usePbrtFourierMaterialWithBsdfFile( m_geom, bsdfFile.string() );
    const uint_t                  proxyGeomId{ 1111 };
    const uint_t                  proxyMaterialId{ 4444U };
    const uint_t                  fourierSbtOffset{ +HitGroupIndex::REALIZED_MATERIAL_START };
    const uint_t                  fourierResourceId{ 55U };
    const FourierMaterialResource fourierResource{
        makeFourierMaterialResource( fourierResourceId, static_cast<CUdeviceptr>( 0x12340000U ) ) };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, realizeFourierMaterialResource( _, _ ) ).WillOnce( Return( fourierResource ) );
    EXPECT_CALL( *m_programGroups, getFourierMaterialSbtOffset( _ ) ).WillOnce( Return( fourierSbtOffset ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( fourierSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( fourierTableReadyState( proxyMaterialId, fourierResourceId ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_LT( proxyMaterialId, m_sync.fourierMaterialResources.size() );
    EXPECT_EQ( fourierResource, m_sync.fourierMaterialResources[proxyMaterialId] );
    expectGeneratedFourierFallbackStats( m_resolver->getStatistics(), 1U, 0U );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeResolvesNamedFourierRelativeBsdfTableBeforeFallback )
{
    const std::filesystem::path sceneDirectory{ makeFourierTestDirectory() };
    const std::filesystem::path bsdfFile{ sceneDirectory / "bsdfs" / "named-fourier.bsdf" };
    writeMinimalFourierBsdfTable( bsdfFile );
    ASSERT_TRUE( std::filesystem::exists( bsdfFile ) );

    m_options.sceneFile       = ( sceneDirectory / "scene.pbrt" ).string();
    m_options.useMdlMaterials = true;
    usePbrtNamedFourierMaterialWithBsdfFile( m_geom, "bsdfs/named-fourier.bsdf" );
    const uint_t                  proxyGeomId{ 1111 };
    const uint_t                  proxyMaterialId{ 4444U };
    const uint_t                  fourierSbtOffset{ +HitGroupIndex::REALIZED_MATERIAL_START };
    const uint_t                  fourierResourceId{ 56U };
    const FourierMaterialResource fourierResource{
        makeFourierMaterialResource( fourierResourceId, static_cast<CUdeviceptr>( 0x56780000U ) ) };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, realizeFourierMaterialResource( _, _ ) ).WillOnce( Return( fourierResource ) );
    EXPECT_CALL( *m_programGroups, getFourierMaterialSbtOffset( _ ) ).WillOnce( Return( fourierSbtOffset ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( fourierSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( fourierTableReadyState( proxyMaterialId, fourierResourceId ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_LT( proxyMaterialId, m_sync.fourierMaterialResources.size() );
    EXPECT_EQ( fourierResource, m_sync.fourierMaterialResources[proxyMaterialId] );
    expectGeneratedFourierFallbackStats( m_resolver->getStatistics(), 1U, 0U );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeKeepsMissingFourierBsdfTableOnFallbackWithoutCompile )
{
    const std::filesystem::path sceneDirectory{ makeFourierTestDirectory() };
    const std::filesystem::path missingFile{ sceneDirectory / "bsdfs" / "missing-fourier.bsdf" };
    std::filesystem::remove( missingFile );

    m_options.sceneFile       = ( sceneDirectory / "scene.pbrt" ).string();
    m_options.useMdlMaterials = true;
    usePbrtFourierMaterialWithBsdfFile( m_geom, "bsdfs/missing-fourier.bsdf" );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    expectGeneratedFourierFallbackStats( m_resolver->getStatistics(), 0U, 1U );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeKeepsInvalidFourierBsdfTableOnFallbackWithoutCompile )
{
    const std::filesystem::path sceneDirectory{ makeFourierTestDirectory() };
    const std::filesystem::path invalidFile{ sceneDirectory / "bsdfs" / "invalid-fourier.bsdf" };
    writeInvalidFourierBsdfTable( invalidFile );
    ASSERT_TRUE( std::filesystem::exists( invalidFile ) );

    m_options.sceneFile       = ( sceneDirectory / "scene.pbrt" ).string();
    m_options.useMdlMaterials = true;
    usePbrtFourierMaterialWithBsdfFile( m_geom, "bsdfs/invalid-fourier.bsdf" );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    expectGeneratedFourierFallbackStats( m_resolver->getStatistics(), 0U, 0U, 1U );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksTextureBackedNonMirrorSpecularMaterialUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtTexturedMaterialOfType( m_geom, "glass", "Kr" );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksTextureBackedTranslucentReflectTransmitUnsupported )
{
    m_options.useMdlMaterials = true;

    const char* const   textureParams[] = { "reflect", "transmit" };
    std::vector<uint_t> requestedMaterialIds;
    const uint_t        firstProxyGeomId{ 1111U };
    const uint_t        firstProxyMaterialId{ 4444U };

    for( size_t index = 0; index < sizeof( textureParams ) / sizeof( textureParams[0] ); ++index )
    {
        GeometryInstance geom{ m_geom };
        geom.groups[0].pbrtMaterial =
            pbrtImagemapMaterialOfType( "translucent", textureParams[index], "color", REFLECTANCE_MAP_PATH );
        const uint_t proxyMaterialId{ firstProxyMaterialId + static_cast<uint_t>( index ) };

        SCOPED_TRACE( textureParams[index] );
        EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
        ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( firstProxyGeomId + static_cast<uint_t>( index ), geom, m_sync ) );
        requestedMaterialIds.push_back( proxyMaterialId );
        EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    }

    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).Times( 2 ).WillRepeatedly( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( requestedMaterialIds ) );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    for( size_t index = 0; index < requestedMaterialIds.size(); ++index )
    {
        const uint_t proxyMaterialId{ requestedMaterialIds[index] };
        SCOPED_TRACE( textureParams[index] );
        ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
        EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    }
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 2U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 2U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksTransformedUberSpecularTextureUnsupported )
{
    m_options.useMdlMaterials = true;
    m_geom.groups[0].pbrtMaterial =
        pbrtScaledImagemapMaterialOfType( "uber", "Ks", "color", SPECULAR_MAP_PATH, make_float3( 0.25f, 0.5f, 0.75f ) );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksTwoLeafBumpTextureUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtUberTwoLeafBumpImagemapMaterial( m_geom );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksMixWithTexturedTransmitBranchUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtLandscapeMixWithTexturedTransmitBranchMaterial( m_geom );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksMixWithTwoLeafBranchTextureUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtLandscapeMixWithTwoLeafBranchTextureMaterial( m_geom );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksMixWithTwoLeafBranchAlphaUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtLandscapeMixWithTwoLeafBranchAlphaMaterial( m_geom );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksMixWithTwoLeafBranchBumpUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtLandscapeMixWithTwoLeafBranchBumpMaterial( m_geom );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksTextureBackedMixAmountUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtMixMaterialWithImagemapAmountTexture( m_geom );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksNonScalarRgbMixAmountUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtMixMaterialWithRgbAmount( m_geom, 0.2f, 0.4f, 0.6f );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, generatedMaterialModeMarksDynamicAmountMixedDiffuseTextureUnsupported )
{
    m_options.useMdlMaterials = true;
    usePbrtDynamicAmountMixedDiffuseImagemapMaterial( m_geom );
    const uint_t proxyGeomId{ 1111 };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( unsupportedFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 0U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompileRequests );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMixedDiffuseMaterialBindsDemandTexture )
{
    m_options.useMdlMaterials = true;
    const float3 textureScale{ make_float3( 0.5f, 0.5f, 0.5f ) };
    const float3 textureBias{ make_float3( 0.125f, 0.25f, 0.375f ) };
    usePbrtMixedDiffuseImagemapMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t      proxyGeomId{ 1111U };
    const uint_t      proxyMaterialId{ 4444U };
    const uint_t      diffuseTextureId{ 333U };
    const uint_t                   linearTextureId{ 334U };
    const uint_t      mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const std::string diffuseMapFileName{ m_geom.groups[0].diffuseMapFileName };
    const PbrtDemandTextureBinding binding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kd" ) };
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( diffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( binding.fileName ), binding.gamma ) ).WillOnce( Return( linearTextureId ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( hasGeometryInstance(
                                       hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                               hasDiffuseTextureId( diffuseTextureId ) ) ) ) )
        .WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader( hasGeometryInstance( hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                                                        hasDiffuseTextureId( diffuseTextureId ) ) ),
                                           1U ) )
        .WillOnce( Return( MdlMaterialShader{ 8U, 1U, linearTextureId, textureScale, textureBias } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( diffuseTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( linearTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( textureScale, m_sync.realizedMaterials[proxyMaterialId].Kd );
    EXPECT_EQ( diffuseTextureId, m_sync.realizedMaterials[proxyMaterialId].diffuseTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    expectMdlMaterialShader( m_sync, proxyMaterialId, MdlMaterialShader{ 8U, 1U, linearTextureId, textureScale, textureBias } );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedDiffuseMaterialBindsDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtDiffuseImagemapMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t      proxyGeomId{ 1111U };
    const uint_t      proxyMaterialId{ 4444U };
    const uint_t      diffuseTextureId{ 333U };
    const uint_t                   linearTextureId{ 334U };
    const uint_t      mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const std::string diffuseMapFileName{ m_geom.groups[0].diffuseMapFileName };
    const PbrtDemandTextureBinding binding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kd" ) };
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( diffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( binding.fileName ), binding.gamma ) ).WillOnce( Return( linearTextureId ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( hasGeometryInstance(
                                       hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                               hasDiffuseTextureId( diffuseTextureId ) ) ) ) )
        .WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader( hasGeometryInstance( hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                                                        hasDiffuseTextureId( diffuseTextureId ) ) ),
                                           1U ) )
        .WillOnce( Return( MdlMaterialShader{ 8U, 1U, linearTextureId, make_float3( 1.0f, 1.0f, 1.0f ),
                                              make_float3( 0.0f, 0.0f, 0.0f ) } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( diffuseTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( linearTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( diffuseTextureId, m_sync.realizedMaterials[proxyMaterialId].diffuseTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    expectMdlMaterialShader( m_sync, proxyMaterialId,
                             MdlMaterialShader{ 8U, 1U, linearTextureId, make_float3( 1.0f, 1.0f, 1.0f ),
                                                make_float3( 0.0f, 0.0f, 0.0f ) } );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedScaledDiffuseMaterialBindsDemandTexture )
{
    m_options.useMdlMaterials = true;
    const float3 textureScale{ make_float3( 0.25f, 0.5f, 0.75f ) };
    usePbrtScaledDiffuseImagemapMaterial( m_geom, textureScale );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t      proxyGeomId{ 1111U };
    const uint_t      proxyMaterialId{ 4444U };
    const uint_t      diffuseTextureId{ 333U };
    const uint_t                   linearTextureId{ 334U };
    const uint_t      mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const std::string diffuseMapFileName{ m_geom.groups[0].diffuseMapFileName };
    const PbrtDemandTextureBinding binding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kd" ) };
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( diffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( binding.fileName ), binding.gamma ) ).WillOnce( Return( linearTextureId ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( hasGeometryInstance(
                                       hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                               hasDiffuseTextureId( diffuseTextureId ) ) ) ) )
        .WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader( hasGeometryInstance( hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                                                        hasDiffuseTextureId( diffuseTextureId ) ) ),
                                           1U ) )
        .WillOnce( Return( MdlMaterialShader{ 8U, 1U, linearTextureId, textureScale, make_float3( 0.0f, 0.0f, 0.0f ) } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( diffuseTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( linearTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( textureScale, m_sync.realizedMaterials[proxyMaterialId].Kd );
    EXPECT_EQ( diffuseTextureId, m_sync.realizedMaterials[proxyMaterialId].diffuseTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    expectMdlMaterialShader( m_sync, proxyMaterialId,
                             MdlMaterialShader{ 8U, 1U, linearTextureId, textureScale, make_float3( 0.0f, 0.0f, 0.0f ) } );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMirrorMaterialBindsReflectanceDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtMirrorReflectanceImagemapMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t      proxyGeomId{ 1111U };
    const uint_t      proxyMaterialId{ 4444U };
    const uint_t      reflectanceTextureId{ 333U };
    const uint_t                   linearTextureId{ 334U };
    const uint_t      mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const std::string reflectanceMapFileName{ m_geom.groups[0].diffuseMapFileName };
    const PbrtDemandTextureBinding binding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kr" ) };
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( reflectanceMapFileName ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( reflectanceMapFileName ) ) ).WillOnce( Return( reflectanceTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( binding.fileName ), binding.gamma ) ).WillOnce( Return( linearTextureId ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( hasGeometryInstance(
                                       hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                               hasDiffuseTextureId( reflectanceTextureId ) ) ) ) )
        .WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader( hasGeometryInstance( hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                                                        hasDiffuseTextureId( reflectanceTextureId ) ) ),
                                           1U ) )
        .WillOnce( Return( MdlMaterialShader{ 8U, 1U, linearTextureId, make_float3( 1.0f, 1.0f, 1.0f ),
                                              make_float3( 0.0f, 0.0f, 0.0f ) } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( reflectanceTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( linearTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( reflectanceTextureId, m_sync.realizedMaterials[proxyMaterialId].diffuseTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    expectMdlMaterialShader( m_sync, proxyMaterialId,
                             MdlMaterialShader{ 8U, 1U, linearTextureId, make_float3( 1.0f, 1.0f, 1.0f ),
                                                make_float3( 0.0f, 0.0f, 0.0f ) } );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedUberMaterialBindsSpecularDemandTextures )
{
    m_options.useMdlMaterials = true;
    usePbrtUberSpecularImagemapMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t                   proxyGeomId{ 1111U };
    const uint_t                   proxyMaterialId{ 4444U };
    const uint_t                   diffuseTextureId{ 333U };
    const uint_t                   linearDiffuseTextureId{ 334U };
    const uint_t                   specularTextureId{ 444U };
    const uint_t                   reflectanceTextureId{ 555U };
    const uint_t                   stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const PbrtDemandTextureBinding diffuseBinding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kd" ) };
    const PbrtDemandTextureBinding specularBinding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Ks" ) };
    const PbrtDemandTextureBinding reflectanceBinding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial,
                                                                                "K"
                                                                                "r" ) };
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( diffuseBinding.fileName ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( diffuseBinding.fileName ) ) ).WillOnce( Return( diffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( diffuseBinding.fileName ), diffuseBinding.gamma ) )
        .WillOnce( Return( linearDiffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( specularBinding.fileName ), specularBinding.gamma ) )
        .WillOnce( Return( specularTextureId ) );
    EXPECT_CALL( *m_demandTextureCache,
                 createLinearTextureFromFile( StrEq( reflectanceBinding.fileName ), reflectanceBinding.gamma ) )
        .WillOnce( Return( reflectanceTextureId ) );
    const float3      textureScale{ make_float3( 1.0f, 1.0f, 1.0f ) };
    const float3      textureBias{ make_float3( 0.0f, 0.0f, 0.0f ) };
    MdlMaterialShader expectedShader{ 8U, 1U, linearDiffuseTextureId, textureScale, textureBias };
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX, specularTextureId,
                                               textureScale, textureBias ) );
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX,
                                               reflectanceTextureId, textureScale, textureBias ) );
    EXPECT_CALL(
        *m_programGroups,
        getMdlMaterialSbtOffset( hasGeometryInstance( hasAll(
            hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ), hasDiffuseTextureId( diffuseTextureId ),
            hasMdlTextureBinding( MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX, linearDiffuseTextureId, textureScale, textureBias ),
            hasMdlTextureBinding( MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX, specularTextureId, textureScale, textureBias ),
            hasMdlTextureBinding( MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX, reflectanceTextureId, textureScale, textureBias ) ) ) ) )
        .WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL(
        *m_programGroups,
        realizeMdlMaterialShader(
            hasGeometryInstance( hasAll(
                hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ), hasDiffuseTextureId( diffuseTextureId ),
                hasMdlTextureBinding( MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX, linearDiffuseTextureId, textureScale, textureBias ),
                hasMdlTextureBinding( MDL_MATERIAL_KS_TEXTURE_BINDING_INDEX, specularTextureId, textureScale, textureBias ),
                hasMdlTextureBinding( MDL_MATERIAL_KR_TEXTURE_BINDING_INDEX, reflectanceTextureId, textureScale, textureBias ) ) ),
            1U ) )
        .WillOnce( Return( expectedShader ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( diffuseTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( reflectanceTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( diffuseTextureId, m_sync.realizedMaterials[proxyMaterialId].diffuseTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( stableSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    ASSERT_LT( 1U, m_sync.mdlMaterialShaders.size() );
    expectMdlMaterialShader( m_sync, proxyMaterialId, expectedShader );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedUberMaterialBindsTransmissionDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtTransmissionImagemapMaterial( m_geom, "uber" );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t                   proxyGeomId{ 1111U };
    const uint_t                   proxyMaterialId{ 4444U };
    const uint_t                   transmissionTextureId{ 333U };
    const uint_t                   stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const MdlMaterialShader        expectedShader{ 8U, 1U };
    const PbrtDemandTextureBinding transmissionBinding{
        pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kt" ) };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( transmissionBinding.fileName ), true ) )
        .WillOnce( Return( transmissionTextureId ) );
    EXPECT_CALL( *m_programGroups,
                 getMdlMaterialSbtOffset( hasGeometryInstance(
                     hasAll( hasMaterialFlags( MaterialFlags::NONE ),
                             hasMdlTextureBinding( MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX, transmissionTextureId,
                                                   make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) ) ) ) )
        .WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader(
                     hasGeometryInstance(
                         hasAll( hasMaterialFlags( MaterialFlags::NONE ),
                                 hasMdlTextureBinding( MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX, transmissionTextureId,
                                                       make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) ) ),
                     1U ) )
        .WillOnce( Return( expectedShader ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( transmissionTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( transmissionTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_FALSE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, expectedShader );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( stableSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedGlassMaterialBindsTransmissionDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtTransmissionImagemapMaterial( m_geom, "glass" );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t                   proxyGeomId{ 1111U };
    const uint_t                   proxyMaterialId{ 4444U };
    const uint_t                   transmissionTextureId{ 333U };
    const uint_t                   stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const MdlMaterialShader        expectedShader{ 8U, 1U };
    const PbrtDemandTextureBinding transmissionBinding{
        pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kt" ) };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( transmissionBinding.fileName ), true ) )
        .WillOnce( Return( transmissionTextureId ) );
    EXPECT_CALL( *m_programGroups,
                 getMdlMaterialSbtOffset( hasGeometryInstance(
                     hasAll( hasMaterialFlags( MaterialFlags::NONE ),
                             hasMdlTextureBinding( MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX, transmissionTextureId,
                                                   make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) ) ) ) )
        .WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader(
                     hasGeometryInstance(
                         hasAll( hasMaterialFlags( MaterialFlags::NONE ),
                                 hasMdlTextureBinding( MDL_MATERIAL_KT_TEXTURE_BINDING_INDEX, transmissionTextureId,
                                                       make_float3( 1.0f, 1.0f, 1.0f ), make_float3( 0.0f, 0.0f, 0.0f ) ) ) ),
                     1U ) )
        .WillOnce( Return( expectedShader ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( transmissionTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( transmissionTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_FALSE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, expectedShader );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( stableSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedTranslucentMaterialBindsDiffuseDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtTranslucentDiffuseImagemapMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t      proxyGeomId{ 1111U };
    const uint_t      proxyMaterialId{ 4444U };
    const uint_t      diffuseTextureId{ 333U };
    const uint_t      linearTextureId{ 334U };
    const uint_t      stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const std::string diffuseMapFileName{ m_geom.groups[0].diffuseMapFileName };
    const PbrtDemandTextureBinding binding{ pbrtColorTextureBinding( *m_geom.groups[0].pbrtMaterial, "Kd" ) };
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( diffuseMapFileName ) ) ).WillOnce( Return( diffuseTextureId ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( binding.fileName ), binding.gamma ) )
        .WillOnce( Return( linearTextureId ) );
    const float3      textureScale{ make_float3( 1.0f, 1.0f, 1.0f ) };
    const float3      textureBias{ make_float3( 0.0f, 0.0f, 0.0f ) };
    MdlMaterialShader expectedShader{ 8U, 1U, linearTextureId, textureScale, textureBias };
    EXPECT_CALL( *m_programGroups,
                 getMdlMaterialSbtOffset( hasGeometryInstance( hasAll(
                     hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ), hasDiffuseTextureId( diffuseTextureId ),
                     hasMdlTextureBinding( MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX, linearTextureId, textureScale, textureBias ) ) ) ) )
        .WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader(
                     hasGeometryInstance( hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                                  hasDiffuseTextureId( diffuseTextureId ),
                                                  hasMdlTextureBinding( MDL_MATERIAL_KD_TEXTURE_BINDING_INDEX,
                                                                        linearTextureId, textureScale, textureBias ) ) ),
                     1U ) )
        .WillOnce( Return( expectedShader ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( diffuseTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( linearTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( diffuseTextureId, m_sync.realizedMaterials[proxyMaterialId].diffuseTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( stableSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    ASSERT_LT( 1U, m_sync.mdlMaterialShaders.size() );
    expectMdlMaterialShader( m_sync, proxyMaterialId, expectedShader );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedUberMaterialBindsBumpmapDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtUberBumpImagemapMaterial( m_geom );
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.devUVs     = fakeUVs;
    m_geom.devNormals = fakeNormals;
    const uint_t                   proxyGeomId{ 1111U };
    const uint_t                   proxyMaterialId{ 4444U };
    const uint_t                   bumpTextureId{ 333U };
    const uint_t                   stableSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    const PbrtDemandTextureBinding bumpBinding{ pbrtFloatTextureBinding( *m_geom.groups[0].pbrtMaterial, "bumpmap" ) };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createLinearTextureFromFile( StrEq( bumpBinding.fileName ), bumpBinding.gamma ) )
        .WillOnce( Return( bumpTextureId ) );
    const float3      textureScale{ make_float3( 1.0f, 1.0f, 1.0f ) };
    const float3      textureBias{ make_float3( 0.0f, 0.0f, 0.0f ) };
    MdlMaterialShader expectedShader{ 8U, 1U };
    EXPECT_TRUE( setMdlMaterialTextureBinding( expectedShader, MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX,
                                               bumpTextureId, textureScale, textureBias ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( hasGeometryInstance( hasMdlTextureBinding(
                                       MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX, bumpTextureId, textureScale, textureBias ) ) ) )
        .WillOnce( Return( stableSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader( hasGeometryInstance( hasMdlTextureBinding( MDL_MATERIAL_BUMPMAP_TEXTURE_BINDING_INDEX,
                                                                                      bumpTextureId, textureScale, textureBias ) ),
                                           1U ) )
        .WillOnce( Return( expectedShader ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( bumpTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( bumpTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( stableSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    ASSERT_LT( 1U, m_sync.mdlMaterialShaders.size() );
    expectMdlMaterialShader( m_sync, proxyMaterialId, expectedShader );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedAlphaMaterialBindsDemandTexture )
{
    m_options.useMdlMaterials = true;
    usePbrtAlphaImagemapMaterial( m_geom );
    TriangleUVs* fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    m_geom.devUVs = fakeUVs;
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t alphaTextureId{ 333U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 12U };
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );

    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( alphaTextureId ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution partialResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::PARTIAL, partialResult );
    EXPECT_EQ( alphaTextureId, m_sync.minAlphaTextureId );
    EXPECT_EQ( alphaTextureId, m_sync.maxAlphaTextureId );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );

    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( hasGeometryInstance(
                                       hasAll( hasMaterialFlags( MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED ),
                                               hasAlphaTextureId( alphaTextureId ) ) ) ) )
        .WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups,
                 realizeMdlMaterialShader( hasGeometryInstance( hasAll( hasMaterialFlags( MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED ),
                                                                        hasAlphaTextureId( alphaTextureId ) ) ),
                                           1U ) )
        .WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution fullResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, fullResult );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( alphaTextureId, m_sync.realizedMaterials[proxyMaterialId].alphaTextureId );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) );
    EXPECT_EQ( 0U, m_sync.partialMaterials[proxyMaterialId].alphaTextureId );
    EXPECT_EQ( nullptr, m_sync.partialUVs[proxyMaterialId] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );
    expectMdlMaterialShader( m_sync, proxyMaterialId, MdlMaterialShader{ 8U, 1U } );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
}

TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMdlMaterialFallsBackWhenCompileFails )
{
    m_options.useMdlMaterials = true;
    usePbrtMaterialOfType( m_geom, "matte" );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 3U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) )
        .WillOnce( Throw( std::runtime_error( "compile failed" ) ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlFailedState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numMissingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numFailedShaders );
}
#endif

#ifdef OTK_USE_MDL
TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMdlMaterialCanRenderFallbackDuringAsynchronousCompile )
{
    m_options.useMdlMaterials            = true;
    m_options.mdlSynchronousCompilation = false;
    usePbrtMaterialOfType( m_geom, "matte" );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 9U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution fallbackResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, fallbackResult );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlPendingState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    EXPECT_TRUE( m_sync.mdlMaterialShaders.empty() );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numMissingShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numFailedShaders );

    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) )
        .WillOnce( Throw( MdlMaterialBuildPending( "still building" ) ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution pendingResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, pendingResult );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlPendingState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );

    stats = m_resolver->getStatistics();
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numMissingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numFailedShaders );

    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) ).WillOnce( Return( MdlMaterialShader{ 8U, 1U } ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution mdlResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::SHADER_DATA_ONLY, mdlResult );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlReadyState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );
    expectMdlMaterialShader( m_sync, proxyMaterialId, MdlMaterialShader{ 8U, 1U } );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( mdlSbtOffset, m_sync.topLevelInstances.back().sbtOffset );

    stats = m_resolver->getStatistics();
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 1U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numSourceCacheHits );
    EXPECT_EQ( 0U, stats.mdlShaders.numMaterialInstanceCacheHits );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numMissingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numFailedShaders );
}
#endif

#ifdef OTK_USE_MDL
TEST_F( TestMaterialResolverRequestedProxyIds, requestedGeneratedMatteMaterialFallsBackWhenAsynchronousCompileFails )
{
    m_options.useMdlMaterials            = true;
    m_options.mdlSynchronousCompilation = false;
    usePbrtMaterialOfType( m_geom, "matte" );
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    const uint_t mdlSbtOffset{ +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + 9U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    EXPECT_CALL( *m_programGroups, getMdlMaterialSbtOffset( _ ) ).WillOnce( Return( mdlSbtOffset ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );
    ASSERT_EQ( MaterialResolution::FULL, m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) );

    EXPECT_CALL( *m_programGroups, realizeMdlMaterialShader( _, 1U ) )
        .WillOnce( Throw( std::runtime_error( "compile failed" ) ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution failedResult{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::SHADER_DATA_ONLY, failedResult );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( mdlFailedState( proxyMaterialId, 1U ), m_sync.materialStates[proxyMaterialId] );

    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numRequestedMaterialPages );
    EXPECT_EQ( 2U, stats.numMdlFallbackShaders );
    EXPECT_EQ( 1U, stats.numGeneratedMdlMaterialCompileRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numShaderRequests );
    EXPECT_EQ( 1U, stats.mdlShaders.numCompileRequests );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.mdlShaders.numQueuedShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numCompilingShaders );
    EXPECT_EQ( 0U, stats.mdlShaders.numReadyShaders );
    EXPECT_EQ( 1U, stats.mdlShaders.numFailedShaders );
}
#endif

TEST_F( TestMaterialResolverRequestedProxyIdsGroups, resolvePhongMaterialGroups )
{
    const uint_t         proxyGeomId{ 1111 };
    const uint_t         proxyMaterialId1{ 4444U };
    const ExpectationSet first{ EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId1 ) ) };
    const uint_t         proxyMaterialId2{ 5555U };
    EXPECT_CALL( *m_loader, add() ).After( first ).WillOnce( Return( proxyMaterialId2 ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillRepeatedly( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId1 } ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId1 ) ).Times( 1 );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId2 ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_EQ( proxyMaterialId2 + 1, m_sync.realizedMaterials.size() );
    EXPECT_EQ( arbitraryPhongMaterial(), m_sync.realizedMaterials[proxyMaterialId1] );
    EXPECT_EQ( arbitraryOtherPhongMaterial(), m_sync.realizedMaterials[proxyMaterialId2] );
    ASSERT_LT( proxyMaterialId2, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId1 ), m_sync.materialStates[proxyMaterialId1] );
    EXPECT_EQ( localFallbackState( proxyMaterialId2 ), m_sync.materialStates[proxyMaterialId2] );
    EXPECT_EQ( 1U, m_sync.realizedNormals.size() );
    EXPECT_EQ( 1U, m_sync.realizedUVs.size() );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    const OptixInstance& instance{ m_sync.topLevelInstances.back() };
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, instance.sbtOffset );
    EXPECT_EQ( 0U, instance.instanceId );
    ASSERT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 2U, 0U } ), m_sync.materialIndices[0] );
    ASSERT_EQ( 2U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, proxyMaterialId1 } ), m_sync.primitiveMaterials[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, proxyMaterialId2 } ), m_sync.primitiveMaterials[1] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 2U, stats.numMaterialsRealized );
}

TEST_F( TestMaterialResolverRequestedProxyIds, resolveAlphaCutOutMaterialPartial )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups[0].material.flags = MaterialFlags::ALPHA_MAP;
    TriangleUVs* fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    m_geom.devUVs                     = fakeUVs;
    m_geom.groups[0].alphaMapFileName = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( _ ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    const uint_t alphaTextureId{ 333 };
    EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( alphaTextureId ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::PARTIAL, result );
    EXPECT_EQ( alphaTextureId, m_sync.minAlphaTextureId );
    EXPECT_EQ( alphaTextureId, m_sync.maxAlphaTextureId );
    ASSERT_FALSE( m_sync.partialMaterials.empty() );
    ASSERT_FALSE( m_sync.partialUVs.empty() );
    EXPECT_EQ( alphaTextureId, m_sync.partialMaterials.back().alphaTextureId );
    EXPECT_EQ( fakeUVs, m_sync.partialUVs.back() );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA, m_sync.topLevelInstances.back().sbtOffset );
}

TEST_F( TestMaterialResolverRequestedProxyIds, resolveAlphaCutOutMaterialFull )
{
    const uint_t proxyGeomId{ 1111 };
    const uint_t alphaTextureId{ 333 };
    m_geom.groups[0].material.flags          = MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED;
    m_geom.groups[0].material.alphaTextureId = alphaTextureId;
    TriangleUVs* fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    m_geom.devUVs                     = fakeUVs;
    m_geom.groups[0].alphaMapFileName = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( _ ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    m_sync.partialMaterials.resize( proxyMaterialId + 1 );
    m_sync.partialUVs.resize( proxyMaterialId + 1 );
    m_sync.partialMaterials.back().alphaTextureId = alphaTextureId;
    m_sync.partialUVs.back()                      = fakeUVs;
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( hasGeometryInstance(
                                       hasAll( hasMaterialFlags( MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED ),
                                               hasAlphaTextureId( alphaTextureId ) ) ) ) )
        .WillOnce( Return( +HitGroupIndex::REALIZED_MATERIAL_START ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    ASSERT_FALSE( m_sync.partialMaterials.empty() );
    ASSERT_FALSE( m_sync.partialUVs.empty() );
    EXPECT_EQ( 0U, m_sync.partialMaterials.back().alphaTextureId );
    EXPECT_EQ( nullptr, m_sync.partialUVs.back() );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    const OptixInstance& topLevel{ m_sync.topLevelInstances.back() };
    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, topLevel.sbtOffset );
    EXPECT_EQ( 0U, topLevel.instanceId );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    EXPECT_EQ( m_geom.groups[0].material, m_sync.realizedMaterials[proxyMaterialId] );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
}

TEST_F( TestMaterialResolverRequestedProxyIds, resolveDiffuseMaterial )
{
    const uint_t     proxyGeomId{ 1111 };
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.groups[0].material.flags     = MaterialFlags::DIFFUSE_MAP;
    m_geom.devUVs                       = fakeUVs;
    m_geom.devNormals                   = fakeNormals;
    m_geom.groups[0].diffuseMapFileName = DIFFUSE_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( _ ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    const uint_t diffuseTextureId{ 333 };
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( diffuseTextureId ) );
    m_geom.groups[0].material.flags |= MaterialFlags::DIFFUSE_MAP_ALLOCATED;
    m_geom.groups[0].material.diffuseTextureId = diffuseTextureId;
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( hasGeometryInstance(
                                       hasAll( hasMaterialFlags( MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED ),
                                               hasDiffuseTextureId( diffuseTextureId ) ) ) ) )
        .WillOnce( Return( +HitGroupIndex::REALIZED_MATERIAL_START ) );
    EXPECT_CALL( *m_loader, remove( proxyMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::FULL, result );
    EXPECT_EQ( diffuseTextureId, m_sync.minDiffuseTextureId );
    EXPECT_EQ( diffuseTextureId, m_sync.maxDiffuseTextureId );
    ASSERT_TRUE( m_sync.partialMaterials.empty() );
    ASSERT_TRUE( m_sync.partialUVs.empty() );
    ASSERT_LT( proxyMaterialId, m_sync.realizedMaterials.size() );
    ASSERT_FALSE( m_sync.realizedNormals.empty() );
    ASSERT_FALSE( m_sync.realizedUVs.empty() );
    EXPECT_EQ( diffuseTextureId, m_sync.realizedMaterials[proxyMaterialId].diffuseTextureId );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, m_sync.topLevelInstances.back().sbtOffset );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials[proxyMaterialId].flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
    ASSERT_LT( proxyMaterialId, m_sync.materialStates.size() );
    EXPECT_EQ( localFallbackState( proxyMaterialId ), m_sync.materialStates[proxyMaterialId] );
}

TEST_F( TestMaterialResolverRequestedProxyIds, oneShotNotTriggeredDoesNothing )
{
    m_options.oneShotMaterial = true;
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution result1{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };
    const MaterialResolution result2{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result1 );
    EXPECT_EQ( MaterialResolution::NONE, result2 );
}

TEST_F( TestMaterialResolverRequestedProxyIds, oneShotTriggeredRequestsProxies )
{
    m_options.oneShotMaterial = true;
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{} ) );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result1{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };
    m_resolver->resolveOneMaterial();
    const MaterialResolution result2{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result1 );
    EXPECT_EQ( MaterialResolution::NONE, result2 );
}
