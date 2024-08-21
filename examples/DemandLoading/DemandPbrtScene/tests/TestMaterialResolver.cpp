// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <MaterialResolver.h>

#include "GeometryInstancePrinter.h"
#include "MockDemandTextureCache.h"
#include "MockMaterialLoader.h"
#include "MockProgramGroups.h"
#include "ParamsPrinters.h"

#include <DemandTextureCache.h>
#include <FrameStopwatch.h>
#include <MaterialBatch.h>
#include <Options.h>
#include <Primitive.h>
#include <ProgramGroups.h>
#include <Scene.h>
#include <SceneGeometry.h>
#include <SceneProxy.h>
#include <SceneSyncState.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <vector_types.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>

constexpr const char* ALPHA_MAP_PATH{ "alphaMap.png" };
constexpr const char* DIFFUSE_MAP_PATH{ "diffuseMap.png" };

using namespace testing;
using namespace otk::testing;
using namespace demandPbrtScene;
using namespace demandPbrtScene::testing;

namespace {

class MockMaterialBatch : public StrictMock<MaterialBatch>
{
  public:
    ~MockMaterialBatch() override = default;

    MOCK_METHOD( void, addMaterialIndex, ( uint_t, uint_t ), ( override ) );
    MOCK_METHOD( uint_t, addPrimitiveMaterialRange, ( uint_t, uint_t ), ( override ) );
    MOCK_METHOD( void, setLaunchParams, (CUstream, Params&), ( override ) );
};

using MockMaterialBatchPtr = std::shared_ptr<MockMaterialBatch>;

inline MockMaterialBatchPtr createMockMaterialBatch()
{
    return std::make_shared<MockMaterialBatch>();
}

inline ListenerPredicate<GeometryInstance> hasMaterialFlags( MaterialFlags value )
{
    return [=]( MatchResultListener* listener, const GeometryInstance& arg ) {
        return hasEqualValues( listener, "flags", value, arg.groups.material.flags );
    };
}

inline ListenerPredicate<GeometryInstance> hasDiffuseTextureId( uint_t value )
{
    return [=]( MatchResultListener* listener, const GeometryInstance& arg ) {
        return hasEqualValues( listener, "diffuse texture id", value, arg.groups.material.diffuseTextureId );
    };
}

inline ListenerPredicate<GeometryInstance> hasAlphaTextureId( uint_t value )
{
    return [=]( MatchResultListener* listener, const GeometryInstance& arg ) {
        return hasEqualValues( listener, "alpha texture id", value, arg.groups.material.alphaTextureId );
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
    MockMaterialBatchPtr      m_batch{ createMockMaterialBatch() };
    MockDemandTextureCachePtr m_demandTextureCache{ createMockDemandTextureCache() };
    MockProgramGroupsPtr      m_programGroups{ createMockProgramGroups() };
    MaterialResolverPtr m_resolver{ createMaterialResolver( m_options, m_loader, m_batch, m_demandTextureCache, m_programGroups ) };
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

}  // namespace

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyPhongMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111U };
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( proxyMaterialId, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyDiffuseMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups.material.flags     = MaterialFlags::DIFFUSE_MAP;
    m_geom.groups.diffuseMapFileName = DIFFUSE_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( proxyMaterialId, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyAlphaCutOutMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups.material.flags   = MaterialFlags::ALPHA_MAP;
    m_geom.groups.alphaMapFileName = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( proxyMaterialId, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyDiffuseAlphaCutOutMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups.material.flags     = MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP;
    m_geom.groups.diffuseMapFileName = DIFFUSE_MAP_PATH;
    m_geom.groups.alphaMapFileName   = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_FALSE( result );
    ASSERT_EQ( 1U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( proxyMaterialId, m_sync.topLevelInstances[0].instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances[0].traversableHandle );
}

TEST_F( TestMaterialResolverForGeometry, resolveSharedPhongMaterialForGeometry )
{
    const uint_t  proxyGeomId{ 1111 };
    PhongMaterial existingMaterial{};
    existingMaterial.Ka       = make_float3( 1.0f, 2.0f, 3.0f );
    existingMaterial.Kd       = make_float3( 4.0f, 5.0f, 6.0f );
    existingMaterial.Ks       = make_float3( 7.0f, 8.0f, 9.0f );
    existingMaterial.Kr       = make_float3( 10.0f, 11.0f, 12.0f );
    existingMaterial.phongExp = 13.4f;
    const uint_t  existingMaterialId{ 1 };
    PhongMaterial otherMaterial{ existingMaterial };
    otherMaterial.Kd = make_float3( 3.0f, 2.0f, 1.0f );
    const uint_t otherMaterialId{ 0 };
    m_sync.realizedMaterials.push_back( otherMaterial );
    m_sync.realizedMaterials.push_back( existingMaterial );
    m_sync.instanceMaterialIds.push_back( otherMaterialId );
    m_sync.instanceMaterialIds.push_back( otherMaterialId );
    m_sync.instanceMaterialIds.push_back( existingMaterialId );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_sync.topLevelInstances.push_back( OptixInstance{} );
    m_geom.groups.material = existingMaterial;
    EXPECT_CALL( *m_loader, add() ).Times( 0 );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( m_geom ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );

    const bool result{ m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) };

    EXPECT_TRUE( result );
    ASSERT_EQ( 4U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( 3U, m_sync.topLevelInstances.back().instanceId );
    EXPECT_EQ( m_geom.instance.traversableHandle, m_sync.topLevelInstances.back().traversableHandle );
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, m_sync.topLevelInstances.back().sbtOffset );
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
    const MaterialResolverStats stats{ m_resolver->getStatistics() };
    EXPECT_EQ( 1U, stats.numMaterialsRealized );
}

TEST_F( TestMaterialResolverRequestedProxyIds, resolveAlphaCutOutMaterialPartial )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.groups.material.flags = MaterialFlags::ALPHA_MAP;
    TriangleUVs* fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    m_geom.devUVs                  = fakeUVs;
    m_geom.groups.alphaMapFileName = ALPHA_MAP_PATH;
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
    m_geom.groups.material.flags          = MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED;
    m_geom.groups.material.alphaTextureId = alphaTextureId;
    TriangleUVs* fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    m_geom.devUVs                  = fakeUVs;
    m_geom.groups.alphaMapFileName = ALPHA_MAP_PATH;
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
    EXPECT_EQ( m_geom.groups.material, m_sync.realizedMaterials.back() );
}

TEST_F( TestMaterialResolverRequestedProxyIds, resolveDiffuseMaterial )
{
    const uint_t     proxyGeomId{ 1111 };
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.groups.material.flags     = MaterialFlags::DIFFUSE_MAP;
    m_geom.devUVs                    = fakeUVs;
    m_geom.devNormals                = fakeNormals;
    m_geom.groups.diffuseMapFileName = DIFFUSE_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( _ ) ).WillOnce( Return( false ) );
    const uint_t proxyMaterialId{ 4444U };
    EXPECT_CALL( *m_loader, add() ).WillOnce( Return( proxyMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_loader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ proxyMaterialId } ) );
    const uint_t diffuseTextureId{ 333 };
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( diffuseTextureId ) );
    m_geom.groups.material.flags |= MaterialFlags::DIFFUSE_MAP_ALLOCATED;
    m_geom.groups.material.diffuseTextureId = diffuseTextureId;
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
