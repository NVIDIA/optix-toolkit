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
    const GeneratedMdlSource& first{ cache.getOrCreate( firstKey ) };
    EXPECT_TRUE( cache.contains( firstKey ) );
    EXPECT_EQ( 1U, cache.size() );
    EXPECT_THAT( first.moduleName, testing::StartsWith( "::otk::demand_pbrt_scene::pbrt_" ) );
    EXPECT_THAT( first.materialName, testing::StartsWith( "material_" ) );
    EXPECT_THAT( first.source, testing::HasSubstr( "export material " + first.materialName ) );

    const MdlShaderKey        equivalentKey{ makeMdlShaderKey( matteMaterial( 0.9f ) ) };
    const GeneratedMdlSource& equivalent{ cache.getOrCreate( equivalentKey ) };
    EXPECT_EQ( 1U, cache.size() );
    EXPECT_EQ( first.moduleName, equivalent.moduleName );
    EXPECT_EQ( first.materialName, equivalent.materialName );
    EXPECT_EQ( first.source, equivalent.source );

    cache.getOrCreate( makeMdlShaderKey( texturedMatteMaterial( "imagemap", "albedo.png" ) ) );
    EXPECT_EQ( 2U, cache.size() );
}

TEST( TestMdlShaderCompileCache, lookupCreatesMissingRecordWithoutQueueingCompile )
{
    MdlShaderCompileCache cache;
    const MdlShaderKey    key{ makeMdlShaderKey( matteMaterial( 0.1f ) ) };

    const MdlShaderCompileRecord& record{ cache.getOrCreate( key ) };
    EXPECT_EQ( MdlShaderCompileState::MISSING, record.state );
    EXPECT_EQ( 1U, record.shaderKeyId );
    EXPECT_EQ( MdlShaderCompileState::MISSING, cache.state( key ) );
    EXPECT_EQ( 1U, cache.shaderKeyId( key ) );
    EXPECT_EQ( 1U, cache.size() );

    const MdlShaderCompileCacheStatistics stats{ cache.getStatistics() };
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
    EXPECT_EQ( 0U, stats.numMissingShaders );
    EXPECT_EQ( 0U, stats.numQueuedShaders );
    EXPECT_EQ( 0U, stats.numCompilingShaders );
    EXPECT_EQ( 0U, stats.numReadyShaders );
    EXPECT_EQ( 1U, stats.numFailedShaders );
}

#endif  // OTK_USE_MDL
