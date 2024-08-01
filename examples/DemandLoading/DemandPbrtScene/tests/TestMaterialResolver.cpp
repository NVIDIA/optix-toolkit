// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <MaterialResolver.h>

#include "ParamsPrinters.h"

#include <DemandTextureCache.h>
#include <FrameStopwatch.h>
#include <Options.h>
#include <PbrtScene.h>
#include <ProgramGroups.h>

#include <OptiXToolkit/DemandGeometry/Mocks/OptixCompare.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <vector_types.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>

constexpr const char* ALPHA_MAP_PATH{ "alphaMap.png" };
constexpr const char* DIFFUSE_MAP_PATH{ "diffuseMap.png" };

using namespace testing;
using namespace demandPbrtScene;

namespace demandPbrtScene {

std::ostream& operator<<( std::ostream& str, GeometryPrimitive value )
{
    switch( value )
    {
        case GeometryPrimitive::NONE:
            return str << "NONE";
        case GeometryPrimitive::TRIANGLE:
            return str << "TRIANGLE";
        case GeometryPrimitive::SPHERE:
            return str << "SPHERE";
    }
    return str << "? (" << static_cast<int>( value ) << ')';
}

std::ostream& operator<<( std::ostream& str, const GeometryInstance& lhs )
{
    return str << "GeometryInstance{ " << lhs.accelBuffer  //
               << ", " << lhs.primitive                    //
               << ", " << lhs.instance                     //
               << ", " << lhs.material                     //
               << ", '" << lhs.diffuseMapFileName << "'"   //
               << ", '" << lhs.alphaMapFileName << "'"     //
               << ", " << lhs.normals                      //
               << ", " << lhs.uvs                          //
               << " }";
}

inline bool operator==( const GeometryInstance& lhs, const GeometryInstance& rhs )
{
    return lhs.accelBuffer == rhs.accelBuffer                   //
           && lhs.primitive == rhs.primitive                    //
           && lhs.instance == rhs.instance                      //
           && lhs.material == rhs.material                      //
           && lhs.diffuseMapFileName == rhs.diffuseMapFileName  //
           && lhs.alphaMapFileName == rhs.alphaMapFileName      //
           && lhs.normals == rhs.normals                        //
           && lhs.uvs == rhs.uvs;
}

}  // namespace demandPbrtScene

namespace {

class MockDemandTextureCache : public StrictMock<DemandTextureCache>
{
  public:
    ~MockDemandTextureCache() override = default;

    MOCK_METHOD( uint_t, createDiffuseTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasDiffuseTextureForFile, ( const std::string& path ), ( const override ) );
    MOCK_METHOD( uint_t, createAlphaTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasAlphaTextureForFile, (const std::string&), ( const, override ) );
    MOCK_METHOD( uint_t, createSkyboxTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasSkyboxTextureForFile, (const std::string&), ( const, override ) );
    MOCK_METHOD( DemandTextureCacheStatistics, getStatistics, (), ( const override ) );
};

class MockMaterialLoader : public StrictMock<demandMaterial::MaterialLoader>
{
  public:
    ~MockMaterialLoader() override = default;

    MOCK_METHOD( const char*, getCHFunctionName, (), ( const, override ) );
    MOCK_METHOD( uint_t, add, (), ( override ) );
    MOCK_METHOD( void, remove, ( uint_t ), ( override ) );
    MOCK_METHOD( std::vector<uint_t>, requestedMaterialIds, (), ( const, override ) );
    MOCK_METHOD( void, clearRequestedMaterialIds, (), ( override ) );
    MOCK_METHOD( bool, getRecycleProxyIds, (), ( const, override ) );
    MOCK_METHOD( void, setRecycleProxyIds, (bool), ( override ) );
};

class MockProgramGroups : public StrictMock<ProgramGroups>
{
  public:
    ~MockProgramGroups() override = default;

    MOCK_METHOD( void, cleanup, (), ( override ) );
    MOCK_METHOD( uint_t, getRealizedMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( void, initialize, (), ( override ) );
};

using MockDemandTextureCachePtr = std::shared_ptr<MockDemandTextureCache>;
using MockMaterialLoaderPtr     = std::shared_ptr<MockMaterialLoader>;
using MockProgramGroupsPtr      = std::shared_ptr<MockProgramGroups>;

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
    MockMaterialLoaderPtr     m_materialLoader{ std::make_shared<MockMaterialLoader>() };
    MockDemandTextureCachePtr m_demandTextureCache{ std::make_shared<MockDemandTextureCache>() };
    MockProgramGroupsPtr      m_programGroups{ std::make_shared<MockProgramGroups>() };
    MaterialResolverPtr m_resolver{ createMaterialResolver( m_options, m_materialLoader, m_demandTextureCache, m_programGroups ) };
    SceneSyncState m_sync{};
};

class TestMaterialResolverForGeometry : public TestMaterialResolver
{
  public:
    ~TestMaterialResolverForGeometry() override = default;

  protected:
    void SetUp() override;

    SceneGeometry m_geom{};
};

void TestMaterialResolverForGeometry::SetUp()
{
    const uint_t arbitraryNonZeroValue{ 2222 };
    m_geom.instanceIndex      = arbitraryNonZeroValue;
    m_geom.instance.primitive = GeometryPrimitive::TRIANGLE;
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
    const uint_t proxyGeomId{ 1111 };
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( 4444 ) );

    const bool result = m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync );

    EXPECT_FALSE( result );
    EXPECT_EQ( 4444, m_geom.materialId );
    EXPECT_EQ( 4444, m_geom.instance.instance.instanceId );
    EXPECT_EQ( 0, m_geom.instanceIndex );
    EXPECT_EQ( 1U, m_sync.topLevelInstances.size() );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyDiffuseMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.instance.material.flags     = MaterialFlags::DIFFUSE_MAP;
    m_geom.instance.diffuseMapFileName = DIFFUSE_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( 4444 ) );

    const bool result = m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync );

    EXPECT_FALSE( result );
    EXPECT_EQ( 4444, m_geom.materialId );
    EXPECT_EQ( 4444, m_geom.instance.instance.instanceId );
    EXPECT_EQ( 0, m_geom.instanceIndex );
    EXPECT_EQ( 1U, m_sync.topLevelInstances.size() );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyAlphaCutOutMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.instance.material.flags   = MaterialFlags::ALPHA_MAP;
    m_geom.instance.alphaMapFileName = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( 4444 ) );

    const bool result = m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync );

    EXPECT_FALSE( result );
    EXPECT_EQ( 4444, m_geom.materialId );
    EXPECT_EQ( 4444, m_geom.instance.instance.instanceId );
    EXPECT_EQ( 0, m_geom.instanceIndex );
    EXPECT_EQ( 1U, m_sync.topLevelInstances.size() );
}

TEST_F( TestMaterialResolverForGeometry, resolveNewProxyDiffuseAlphaCutOutMaterialForGeometry )
{
    const uint_t proxyGeomId{ 1111 };
    m_geom.instance.material.flags     = MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP;
    m_geom.instance.diffuseMapFileName = DIFFUSE_MAP_PATH;
    m_geom.instance.alphaMapFileName   = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( 4444 ) );

    const bool result = m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync );

    EXPECT_FALSE( result );
    EXPECT_EQ( 4444, m_geom.materialId );
    EXPECT_EQ( 4444, m_geom.instance.instance.instanceId );
    EXPECT_EQ( 0, m_geom.instanceIndex );
    EXPECT_EQ( 1U, m_sync.topLevelInstances.size() );
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
    m_geom.instance.material = existingMaterial;
    EXPECT_CALL( *m_materialLoader, add() ).Times( 0 );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( m_geom.instance ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );

    const bool result = m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync );

    EXPECT_TRUE( result );
    EXPECT_EQ( existingMaterialId, m_geom.materialId );
    EXPECT_EQ( 3U, m_geom.instanceIndex );
    EXPECT_EQ( 3U, m_geom.instance.instance.instanceId );
    EXPECT_EQ( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START, m_geom.instance.instance.sbtOffset );
    EXPECT_EQ( 4U, m_sync.topLevelInstances.size() );
    EXPECT_EQ( m_geom.instance.instance, m_sync.topLevelInstances.back() );
}

TEST_F( TestMaterialResolverRequestedProxyIds, noRequestedProxyMaterials )
{
    EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{} ) );
    EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result );
}

TEST_F( TestMaterialResolverRequestedProxyIds, resolvePhongMaterial )
{
    const uint_t fakeMaterialId{ 2222 };
    const uint_t proxyGeomId{ 1111 };
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( fakeMaterialId ) );
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( _ ) ).WillOnce( Return( +ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ fakeMaterialId } ) );
    EXPECT_CALL( *m_materialLoader, remove( fakeMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).Times( 1 );

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
    const uint_t fakeMaterialId{ 2222 };
    const uint_t proxyGeomId{ 1111 };
    m_geom.instance.material.flags = MaterialFlags::ALPHA_MAP;
    TriangleUVs* fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    m_geom.instance.uvs              = fakeUVs;
    m_geom.instance.alphaMapFileName = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( _ ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( fakeMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ fakeMaterialId } ) );
    const uint_t alphaTextureId{ 333 };
    EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( StrEq( ALPHA_MAP_PATH ) ) ).WillOnce( Return( alphaTextureId ) );
    EXPECT_CALL( *m_materialLoader, remove( fakeMaterialId ) ).Times( 0 );
    EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).Times( 1 );

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
    const uint_t fakeMaterialId{ 2222 };
    const uint_t proxyGeomId{ 1111 };
    m_geom.instance.material.flags = MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED;
    TriangleUVs* fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    m_geom.instance.uvs              = fakeUVs;
    m_geom.instance.alphaMapFileName = ALPHA_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasAlphaTextureForFile( _ ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( fakeMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ fakeMaterialId } ) );
    const uint_t alphaTextureId{ 333 };
    m_sync.partialMaterials.resize( fakeMaterialId + 1 );
    m_sync.partialUVs.resize( fakeMaterialId + 1 );
    m_sync.partialMaterials.back().alphaTextureId = alphaTextureId;
    m_sync.partialUVs.back()                      = fakeUVs;
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( m_geom.instance ) ).WillOnce( Return( +HitGroupIndex::REALIZED_MATERIAL_START ) );
    EXPECT_CALL( *m_materialLoader, remove( fakeMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).Times( 1 );

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
    EXPECT_EQ( m_geom.instance.material, m_sync.realizedMaterials.back() );
}

TEST_F( TestMaterialResolverRequestedProxyIds, resolveDiffuseMaterial )
{
    const uint_t     fakeMaterialId{ 2222 };
    const uint_t     proxyGeomId{ 1111 };
    TriangleUVs*     fakeUVs{ reinterpret_cast<TriangleUVs*>( 0xdeadbeefULL ) };
    TriangleNormals* fakeNormals{ reinterpret_cast<TriangleNormals*>( 0xbaadf00dULL ) };
    m_geom.instance.material.flags     = MaterialFlags::DIFFUSE_MAP;
    m_geom.instance.uvs                = fakeUVs;
    m_geom.instance.normals            = fakeNormals;
    m_geom.instance.diffuseMapFileName = DIFFUSE_MAP_PATH;
    EXPECT_CALL( *m_demandTextureCache, hasDiffuseTextureForFile( _ ) ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_materialLoader, add() ).WillOnce( Return( fakeMaterialId ) );
    ASSERT_FALSE( m_resolver->resolveMaterialForGeometry( proxyGeomId, m_geom, m_sync ) );
    EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{ fakeMaterialId } ) );
    const uint_t diffuseTextureId{ 333 };
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( DIFFUSE_MAP_PATH ) ) ).WillOnce( Return( diffuseTextureId ) );
    m_geom.instance.material.flags |= MaterialFlags::DIFFUSE_MAP_ALLOCATED;
    m_geom.instance.material.diffuseTextureId = diffuseTextureId;
    EXPECT_CALL( *m_programGroups, getRealizedMaterialSbtOffset( m_geom.instance ) ).WillOnce( Return( +HitGroupIndex::REALIZED_MATERIAL_START ) );
    EXPECT_CALL( *m_materialLoader, remove( fakeMaterialId ) ).Times( 1 );
    EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).Times( 1 );

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
    const uint_t fakeMaterialId{ 2222 };
    EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).Times( 0 );
    EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).Times( 0 );

    const MaterialResolution result1{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };
    const MaterialResolution result2{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result1 );
    EXPECT_EQ( MaterialResolution::NONE, result2 );
}

TEST_F( TestMaterialResolverRequestedProxyIds, oneShotTriggeredRequestsProxies )
{
    m_options.oneShotMaterial = true;
    const uint_t fakeMaterialId{ 2222 };
    EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).WillOnce( Return( std::vector<uint_t>{} ) );
    EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).Times( 1 );

    const MaterialResolution result1{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };
    m_resolver->resolveOneMaterial();
    const MaterialResolution result2{ m_resolver->resolveRequestedProxyMaterials( m_stream, m_timer, m_sync ) };

    EXPECT_EQ( MaterialResolution::NONE, result1 );
    EXPECT_EQ( MaterialResolution::NONE, result2 );
}
