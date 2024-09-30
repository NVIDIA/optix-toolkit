// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/MaterialResolver.h>

#include "GeometryInstancePrinter.h"
#include "MockDemandTextureCache.h"
#include "MockMaterialLoader.h"
#include "MockProgramGroups.h"
#include "ParamsPrinters.h"

#include <DemandPbrtScene/DemandTextureCache.h>
#include <DemandPbrtScene/FrameStopwatch.h>
#include <DemandPbrtScene/Options.h>
#include <DemandPbrtScene/Primitive.h>
#include <DemandPbrtScene/ProgramGroups.h>
#include <DemandPbrtScene/Scene.h>
#include <DemandPbrtScene/SceneGeometry.h>
#include <DemandPbrtScene/SceneProxy.h>
#include <DemandPbrtScene/SceneSyncState.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <vector_types.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>

constexpr const char* ALPHA_MAP_PATH{ "alphaMap.png" };
constexpr const char* DIFFUSE_MAP_PATH{ "diffuseMap.png" };
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
    return options;
}

class TestMaterialResolver : public Test
{
  public:
    ~TestMaterialResolver() override = default;

  protected:
    Options                   m_options{ testOptions() };
    MockMaterialLoaderPtr     m_loader{ createMockMaterialLoader() };
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
}

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

TEST_F( TestMaterialResolverRequestedProxyIds, noRequestedProxyMaterials )
{
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{} ) );
    EXPECT_CALL( *m_loader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result );
}

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
    EXPECT_EQ( 1U, m_sync.realizedMaterials.size() );
    EXPECT_EQ( 1U, m_sync.realizedNormals.size() );
    EXPECT_EQ( 1U, m_sync.realizedUVs.size() );
    const OptixInstance& instance{ m_sync.topLevelInstances.back() };
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, instance.sbtOffset );
    EXPECT_EQ( 0U, instance.instanceId );
    ASSERT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 1U, 0U } ), m_sync.materialIndices[0] );
    ASSERT_EQ( 1U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, 0U } ), m_sync.primitiveMaterials[0] );
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numMaterialsRealized );
}

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
    ASSERT_EQ( 2U, m_sync.realizedMaterials.size() );
    EXPECT_EQ( arbitraryPhongMaterial(), m_sync.realizedMaterials[0] );
    EXPECT_EQ( arbitraryOtherPhongMaterial(), m_sync.realizedMaterials[1] );
    EXPECT_EQ( 2U, m_sync.realizedNormals.size() );
    EXPECT_EQ( 2U, m_sync.realizedUVs.size() );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    const OptixInstance& instance{ m_sync.topLevelInstances.back() };
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, instance.sbtOffset );
    EXPECT_EQ( 0U, instance.instanceId );
    ASSERT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ 2U, 0U } ), m_sync.materialIndices[0] );
    ASSERT_EQ( 2U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END, 0U } ), m_sync.primitiveMaterials[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ ARBITRARY_PRIMITIVE_INDEX_END2, 1U } ), m_sync.primitiveMaterials[1] );
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
    ASSERT_FALSE( m_sync.realizedMaterials.empty() );
    EXPECT_EQ( m_geom.groups[0].material, m_sync.realizedMaterials.back() );
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
    ASSERT_FALSE( m_sync.realizedMaterials.empty() );
    ASSERT_FALSE( m_sync.realizedNormals.empty() );
    ASSERT_FALSE( m_sync.realizedUVs.empty() );
    EXPECT_EQ( diffuseTextureId, m_sync.realizedMaterials.back().diffuseTextureId );
    EXPECT_EQ( fakeUVs, m_sync.realizedUVs.back() );
    EXPECT_EQ( fakeNormals, m_sync.realizedNormals.back() );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, m_sync.topLevelInstances.back().sbtOffset );
    EXPECT_TRUE( flagSet( m_sync.realizedMaterials.back().flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) );
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
