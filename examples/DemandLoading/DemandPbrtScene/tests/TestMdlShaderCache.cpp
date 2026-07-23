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

PbrtMaterial matteMaterial( float kd )
{
    PbrtMaterial material;
    material.type = "matte";
    addRgbSpectrum( material.params, "Kd", kd, kd, kd );
    return material;
}

PbrtTexture imageMapTexture( const std::string& name, const std::string& fileName )
{
    PbrtTexture texture;
    texture.name      = name;
    texture.valueType = "spectrum";
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
    EXPECT_THAT( generated.source, testing::HasSubstr( "// demand texture parameter: texture_2d image_0" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_0() = pbrt_demand_texture_2d(0);" ) );
    EXPECT_THAT( generated.source, testing::HasSubstr( "tint: texture_0()" ) );
    EXPECT_THAT( generated.source, testing::Not( testing::HasSubstr( "albedo.png" ) ) );
    EXPECT_TRUE( generated.unsupportedReasons.empty() );
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
    EXPECT_THAT( generated.source, testing::HasSubstr( "color texture_2() = pbrt_checkerboard_2d(color(1.0, 1.0, 1.0), "
                                                       "color(0.0, 0.0, 0.0));" ) );
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
    MdlShaderCompileCache cache;
    const MdlShaderKey    key{ makeMdlShaderKey( matteMaterial( 0.1f ) ) };

    const MdlShaderCompileRecord& record{ cache.getRecord( key ) };
    EXPECT_EQ( MdlShaderCompileState::MISSING, record.state );
    EXPECT_EQ( 1U, record.shaderKeyId );
    EXPECT_EQ( MdlShaderCompileState::MISSING, cache.state( key ) );
    EXPECT_EQ( 1U, cache.shaderKeyId( key ) );
    EXPECT_EQ( 1U, cache.size() );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 1U, stats.numShaderRequests );
    EXPECT_EQ( 0U, stats.numShaderCacheHits );
    EXPECT_EQ( 0U, stats.numCompileRequests );
    EXPECT_EQ( 0U, stats.numCompletedCompiles );
    EXPECT_EQ( 1U, stats.numMissingShaders );
    EXPECT_EQ( 0U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 0U, stats.numFailedShaders );
}

TEST( TestMdlShaderCompileCache, duplicateRequestsDoNotQueueDuplicateCompiles )
{
    MdlShaderCompileCache cache;
    const MdlShaderKey    key{ makeMdlShaderKey( matteMaterial( 0.1f ) ) };
    const MdlShaderKey    equivalentKey{ makeMdlShaderKey( matteMaterial( 0.9f ) ) };

    EXPECT_TRUE( cache.requestCompile( key ) );
    EXPECT_FALSE( cache.requestCompile( equivalentKey ) );
    EXPECT_EQ( MdlShaderCompileState::QUEUED, cache.state( key ) );
    EXPECT_EQ( 1U, cache.shaderKeyId( equivalentKey ) );
    EXPECT_EQ( 1U, cache.size() );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 0U, stats.numShaderRequests );
    EXPECT_EQ( 1U, stats.numShaderCacheHits );
    EXPECT_EQ( 1U, stats.numCompileRequests );
    EXPECT_EQ( 0U, stats.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.numMissingShaders );
    EXPECT_EQ( 1U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 0U, stats.numFailedShaders );
}

TEST( TestMdlShaderCompileCache, tracksCompileStateTransitions )
{
    MdlShaderCompileCache cache;
    const MdlShaderKey    firstKey{ makeMdlShaderKey( matteMaterial( 0.1f ) ) };
    const MdlShaderKey    secondKey{ makeMdlShaderKey( texturedMatteMaterial( "checkerboard", "" ) ) };

    EXPECT_TRUE( cache.requestCompile( firstKey ) );
    EXPECT_TRUE( cache.requestCompile( secondKey ) );
    cache.markCompiling( firstKey );
    cache.markReady( firstKey );

    EXPECT_EQ( MdlShaderCompileState::READY, cache.state( firstKey ) );
    EXPECT_EQ( MdlShaderCompileState::QUEUED, cache.state( secondKey ) );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 0U, stats.numShaderRequests );
    EXPECT_EQ( 0U, stats.numShaderCacheHits );
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
    MdlShaderCompileCache cache;
    const MdlShaderKey    key{ makeMdlShaderKey( matteMaterial( 0.1f ) ) };

    EXPECT_TRUE( cache.requestCompile( key ) );
    cache.markCompiling( key );
    cache.markFailed( key, "mdl compile failed" );

    EXPECT_FALSE( cache.requestCompile( key ) );
    EXPECT_EQ( MdlShaderCompileState::FAILED, cache.state( key ) );
    EXPECT_EQ( "mdl compile failed", cache.diagnostics( key ) );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
    EXPECT_EQ( 0U, stats.numShaderRequests );
    EXPECT_EQ( 1U, stats.numShaderCacheHits );
    EXPECT_EQ( 1U, stats.numCompileRequests );
    EXPECT_EQ( 0U, stats.numCompletedCompiles );
    EXPECT_EQ( 0U, stats.numMissingShaders );
    EXPECT_EQ( 0U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 1U, stats.numFailedShaders );
}

#endif  // OTK_USE_MDL
