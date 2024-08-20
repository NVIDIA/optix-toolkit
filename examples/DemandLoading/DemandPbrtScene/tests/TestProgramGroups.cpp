// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <ProgramGroups.h>

#include "Matchers.h"
#include "MockGeometryLoader.h"
#include "MockMaterialLoader.h"
#include "MockRenderer.h"
#include "ParamsPrinters.h"
#include "SceneAdapters.h"

#include <Renderer.h>
#include <SceneProxy.h>

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockOptix.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>

using namespace testing;
using namespace demandPbrtScene;
using namespace demandPbrtScene::testing;
using namespace otk::testing;

inline OptixDeviceContext fakeOptixDeviceContext()
{
    return reinterpret_cast<OptixDeviceContext>( static_cast<std::intptr_t>( 0xfeedfeed ) );
}

inline OptixProgramGroup PG( unsigned int id )
{
    return reinterpret_cast<OptixProgramGroup>( static_cast<std::intptr_t>( id ) );
}

MATCHER( hasModuleTypeTriangle, "" )
{
    const bool result = arg->builtinISModuleType == OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    if( !result )
    {
        *result_listener << "module has type " << arg->builtinISModuleType
                         << ", expected OPTIX_PRIMITIVE_TYPE_TRIANGLE (" << OPTIX_PRIMITIVE_TYPE_TRIANGLE << ")";
    }
    else
    {
        *result_listener << "module has type OPTIX_PRIMITIVE_TYPE_TRIANGLE (" << OPTIX_PRIMITIVE_TYPE_TRIANGLE << ")";
    }
    return result;
}

MATCHER( hasModuleTypeSphere, "" )
{
    const bool result = arg->builtinISModuleType == OPTIX_PRIMITIVE_TYPE_SPHERE;
    if( !result )
    {
        *result_listener << "module has type " << arg->builtinISModuleType << ", expected OPTIX_PRIMITIVE_TYPE_SPHERE ("
                         << OPTIX_PRIMITIVE_TYPE_SPHERE << ")";
    }
    else
    {
        *result_listener << "module has type OPTIX_PRIMITIVE_TYPE_SPHERE (" << OPTIX_PRIMITIVE_TYPE_SPHERE << ")";
    }
    return result;
}

MATCHER( allowsRandomVertexAccess, "" )
{
    const bool result = ( arg->buildFlags & OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ) == OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    const char fill = result_listener->stream()->fill();
    if( !result )
    {
        *result_listener << "builtin IS module options build flags " << std::dec << arg->buildFlags << " (0x"
                         << std::hex << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << arg->buildFlags << ") don't set OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS (0x" << std::hex
                         << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS << ")" << std::setfill( fill );
    }
    else
    {
        *result_listener << "builtin IS module options build flags " << std::dec << arg->buildFlags << " (0x"
                         << std::hex << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << arg->buildFlags << ") set OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS (0x" << std::hex
                         << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS << ")" << std::setfill( fill );
    }
    return result;
}

MATCHER_P( hasProgramGroupCount, count, "" )
{
    if( arg.size() != count )
    {
        *result_listener << "program group count " << arg.size() << ", expected " << count;
        return false;
    }

    *result_listener << "has program group count " << count;
    return true;
}

namespace demandLoading {

template <typename T>
std::ostream& operator<<( std::ostream& str, const DeviceArray<T>& val )
{
    return str << "DeviceArray{" << static_cast<void*>( val.data ) << ", " << val.capacity << "}";
}

template <typename T>
bool operator==( const DeviceArray<T>& lhs, const DeviceArray<T>& rhs )
{
    return lhs.data == rhs.data && lhs.capacity == rhs.capacity;
}

template <typename T>
bool operator!=( const DeviceArray<T>& lhs, const DeviceArray<T>& rhs )
{
    return !( lhs == rhs );
}

inline bool operator==( const DeviceContext& lhs, const DeviceContext& rhs )
{
    // clang-format off
    return lhs.pageTable         == rhs.pageTable
        && lhs.maxNumPages       == rhs.maxNumPages
        && lhs.referenceBits     == rhs.referenceBits
        && lhs.residenceBits     == rhs.residenceBits
        && lhs.lruTable          == rhs.lruTable
        && lhs.requestedPages    == rhs.requestedPages
        && lhs.stalePages        == rhs.stalePages
        && lhs.evictablePages    == rhs.evictablePages
        && lhs.arrayLengths      == rhs.arrayLengths
        && lhs.filledPages       == rhs.filledPages
        && lhs.invalidatedPages  == rhs.invalidatedPages
        && lhs.requestIfResident == rhs.requestIfResident 
        && lhs.poolIndex         == rhs.poolIndex;
    // clang-format on
}

inline bool operator!=( const DeviceContext& lhs, const DeviceContext& rhs )
{
    return !( lhs == rhs );
}

}  // namespace demandLoading

namespace {

using StrictMockOptix = StrictMock<MockOptix>;

using ProgramGroupDescMatcher = Matcher<const OptixProgramGroupDesc*>;

class TestProgramGroups : public Test
{
  public:
    ~TestProgramGroups() override = default;

  protected:
    void SetUp() override;

    Expectation    expectModuleCreated( OptixModule module );
    ExpectationSet expectInitialize();

    StrictMockOptix       m_optix{};
    MockGeometryLoaderPtr m_geometryLoader{ createMockGeometryLoader() };
    MockMaterialLoaderPtr m_materialLoader{ createMockMaterialLoader() };
    MockRendererPtr       m_renderer{ createMockRenderer() };
    ProgramGroupsPtr      m_programGroups{ createProgramGroups( m_geometryLoader, m_materialLoader, m_renderer ) };
    OptixPipelineCompileOptions    m_pipelineCompileOptions{};
    OptixDeviceContext             m_fakeContext{ fakeOptixDeviceContext() };
    OptixModule                    m_sceneModule{ reinterpret_cast<OptixModule>( 0x1111U ) };
    OptixModule                    m_builtinTriangleModule{ reinterpret_cast<OptixModule>( 0x4444U ) };
    OptixModule                    m_builtinSphereModule{ reinterpret_cast<OptixModule>( 0x5555U ) };
    const char*                    m_proxyGeomIS{ "__intersection__proxyGeometry" };
    const char*                    m_proxyGeomCH{ "__closesthit__proxyGeometry" };
    const char*                    m_proxyMatCH{ "__closesthit__proxyMaterial" };
    const char*                    m_proxyMatMeshAlphaAH{ "__anyhit__alphaCutOutPartialMesh" };
    const char*                    m_proxyMatSphereAlphaAH{ "__anyhit__sphere" };
    std::vector<OptixProgramGroup> m_fakeProgramGroups{ PG( 111100U ), PG( 2222000U ), PG( 333300U ), PG( 444400U ),
                                                        PG( 555500U ), PG( 666600U ),  PG( 777700U ) };
};

void TestProgramGroups::SetUp()
{
    Test::SetUp();
    m_pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
}

Expectation TestProgramGroups::expectModuleCreated( OptixModule module )
{
    Expectation expect;
#if OPTIX_VERSION < 70700
    expect = EXPECT_CALL( m_optix, moduleCreateFromPTX( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#else
    expect = EXPECT_CALL( m_optix, moduleCreate( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#endif
    return expect;
}

ExpectationSet TestProgramGroups::expectInitialize()
{
    ExpectationSet expect;
    expect += EXPECT_CALL( *m_renderer, getPipelineCompileOptions() ).Times( AtLeast( 1 ) ).WillRepeatedly( ReturnRef( m_pipelineCompileOptions ) );
    expect += EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_fakeContext ) );
    expect += expectModuleCreated( m_sceneModule );
    expect += EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, _, _, hasModuleTypeTriangle(), _ ) )
                  .WillOnce( DoAll( SetArgPointee<4>( m_builtinTriangleModule ), Return( OPTIX_SUCCESS ) ) );
    expect += EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, _, _, hasModuleTypeSphere(), _ ) )
                  .WillOnce( DoAll( SetArgPointee<4>( m_builtinSphereModule ), Return( OPTIX_SUCCESS ) ) );
    expect += EXPECT_CALL( *m_geometryLoader, getISFunctionName() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_proxyGeomIS ) );
    expect += EXPECT_CALL( *m_geometryLoader, getCHFunctionName() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_proxyGeomCH ) );
    expect += EXPECT_CALL( *m_materialLoader, getCHFunctionName() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_proxyMatCH ) );
    expect += EXPECT_CALL( m_optix, programGroupCreate( m_fakeContext, _, m_fakeProgramGroups.size(), _, _, _, _ ) )
                  .WillOnce( DoAll( SetArrayArgument<6>( m_fakeProgramGroups.begin(), m_fakeProgramGroups.end() ),
                                    Return( OPTIX_SUCCESS ) ) );
    expect += EXPECT_CALL( *m_renderer, setProgramGroups( _ ) ).Times( 1 );
    return expect;
}

class TestProgramGroupsInitialized : public TestProgramGroups
{
  public:
    ~TestProgramGroupsInitialized() override = default;

  protected:
    void SetUp() override;

    Expectation expectModuleCreatedAfter( OptixModule module, const ExpectationSet& before );
    ExpectationSet expectProgramGroupAddedAfter( const ProgramGroupDescMatcher& desc, OptixProgramGroup result, const ExpectationSet& before );

    ExpectationSet                 m_init;
    OptixModule                    m_phongModule{ reinterpret_cast<OptixModule>( 0x3333U ) };
    std::vector<OptixProgramGroup> m_updatedGroups{ m_fakeProgramGroups };
    OptixProgramGroup              m_fakePhongProgramGroup{ PG( 8888U ) };
    OptixProgramGroup              m_fakeAlphaPhongProgramGroup{ PG( 9999U ) };
};

void TestProgramGroupsInitialized::SetUp()
{
    TestProgramGroups::SetUp();
    m_init = expectInitialize();
    m_programGroups->initialize();
}

Expectation TestProgramGroupsInitialized::expectModuleCreatedAfter( OptixModule module, const ExpectationSet& before )
{
    Expectation expect;
#if OPTIX_VERSION < 70700
    expect = EXPECT_CALL( m_optix, moduleCreateFromPTX( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .After( before )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#else
    expect = EXPECT_CALL( m_optix, moduleCreate( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .After( before )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#endif
    return expect;
}

ExpectationSet TestProgramGroupsInitialized::expectProgramGroupAddedAfter( const ProgramGroupDescMatcher& desc,
                                                                           OptixProgramGroup              result,
                                                                           const ExpectationSet&          before )
{
    m_updatedGroups.push_back( result );
    ExpectationSet set;
    set += EXPECT_CALL( m_optix, programGroupCreate( m_fakeContext, desc, 1, _, _, _, _ ) )
               .After( before )
               .WillOnce( DoAll( SetArgPointee<6>( result ), Return( OPTIX_SUCCESS ) ) );
    set += EXPECT_CALL( *m_renderer, setProgramGroups( hasProgramGroupCount( m_updatedGroups.size() ) ) ).Times( 1 ).After( before );
    return set;
}

}  // namespace

TEST_F( TestProgramGroups, initializeCreatesOptixResourcesForLoadedScene )
{
    EXPECT_CALL( *m_renderer, getPipelineCompileOptions() ).Times( AtLeast( 1 ) ).WillRepeatedly( ReturnRef( m_pipelineCompileOptions ) );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_fakeContext ) );
    expectModuleCreated( m_sceneModule );
    EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, NotNull(), Pointee( Eq( m_pipelineCompileOptions ) ),
                                              AllOf( NotNull(), hasModuleTypeTriangle(), allowsRandomVertexAccess() ), NotNull() ) )
        .WillOnce( DoAll( SetArgPointee<4>( m_builtinTriangleModule ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, NotNull(), Pointee( Eq( m_pipelineCompileOptions ) ),
                                              AllOf( NotNull(), hasModuleTypeSphere(), allowsRandomVertexAccess() ), NotNull() ) )
        .WillOnce( DoAll( SetArgPointee<4>( m_builtinSphereModule ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( *m_geometryLoader, getISFunctionName() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_proxyGeomIS ) );
    EXPECT_CALL( *m_geometryLoader, getCHFunctionName() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_proxyGeomCH ) );
    EXPECT_CALL( *m_materialLoader, getCHFunctionName() ).Times( AtLeast( 1 ) ).WillRepeatedly( Return( m_proxyMatCH ) );
    const char* const proxyMatIS = nullptr;
    size_t            numGroups  = m_fakeProgramGroups.size();
    auto              expectedProgramGroupDescs =
        AllOf( NotNull(), hasRayGenDesc( numGroups, m_sceneModule, "__raygen__perspectiveCamera" ),
               hasMissDesc( numGroups, m_sceneModule, "__miss__backgroundColor" ),
               hasHitGroupISCHDesc( numGroups, m_sceneModule, m_proxyGeomIS, m_sceneModule, m_proxyGeomCH ),
               hasHitGroupISCHDesc( numGroups, m_builtinTriangleModule, proxyMatIS, m_sceneModule, m_proxyMatCH ),
               hasHitGroupISAHCHDesc( numGroups, m_builtinTriangleModule, proxyMatIS, m_sceneModule,
                                      m_proxyMatMeshAlphaAH, m_sceneModule, m_proxyMatCH ),
               hasHitGroupISCHDesc( numGroups, m_builtinSphereModule, proxyMatIS, m_sceneModule, m_proxyMatCH ),
               hasHitGroupISAHCHDesc( numGroups, m_builtinSphereModule, proxyMatIS, m_sceneModule,
                                      m_proxyMatSphereAlphaAH, m_sceneModule, m_proxyMatCH ) );
    EXPECT_CALL( m_optix, programGroupCreate( m_fakeContext, expectedProgramGroupDescs, m_fakeProgramGroups.size(),
                                              NotNull(), NotNull(), NotNull(), NotNull() ) )
        .WillOnce( DoAll( SetArrayArgument<6>( m_fakeProgramGroups.begin(), m_fakeProgramGroups.end() ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( *m_renderer, setProgramGroups( _ ) ).Times( 1 );

    m_programGroups->initialize();
}

TEST_F( TestProgramGroupsInitialized, cleanupDestroysOptixResources )
{
    for( OptixProgramGroup group : m_fakeProgramGroups )
    {
        EXPECT_CALL( m_optix, programGroupDestroy( group ) ).After( m_init ).WillOnce( Return( OPTIX_SUCCESS ) );
    }
    EXPECT_CALL( m_optix, moduleDestroy( m_sceneModule ) ).After( m_init ).WillOnce( Return( OPTIX_SUCCESS ) );

    m_programGroups->cleanup();
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForPhongSpheres )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const sphereIS = nullptr;
    const char* const sphereCH = "__closesthit__sphere";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinSphereModule, sphereIS, m_phongModule, sphereCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakePhongProgramGroup, m_init );
    GeometryInstance instance{};
    instance.primitive = GeometryPrimitive::SPHERE;

    const uint_t index = m_programGroups->getRealizedMaterialSbtOffset( instance );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForPhongSpheresShared )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const sphereIS = nullptr;
    const char* const sphereCH = "__closesthit__sphere";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinSphereModule, sphereIS, m_phongModule, sphereCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakePhongProgramGroup, m_init );
    GeometryInstance instance1{};
    instance1.primitive = GeometryPrimitive::SPHERE;
    GeometryInstance instance2{};
    instance2.primitive = GeometryPrimitive::SPHERE;

    const uint_t index1 = m_programGroups->getRealizedMaterialSbtOffset( instance1 );
    const uint_t index2 = m_programGroups->getRealizedMaterialSbtOffset( instance2 );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index1 );
    EXPECT_EQ( index2, index1 );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForPhongTriangles )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshCH = "__closesthit__mesh";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_phongModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakePhongProgramGroup, m_init );
    GeometryInstance instance{};
    instance.primitive = GeometryPrimitive::TRIANGLE;

    const uint_t index = m_programGroups->getRealizedMaterialSbtOffset( instance );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForPhongTrianglesShared )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshCH = "__closesthit__mesh";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_phongModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakePhongProgramGroup, m_init );
    GeometryInstance instance1{};
    instance1.primitive = GeometryPrimitive::TRIANGLE;
    GeometryInstance instance2{};
    instance2.primitive = GeometryPrimitive::TRIANGLE;

    const uint_t index1 = m_programGroups->getRealizedMaterialSbtOffset( instance1 );
    const uint_t index2 = m_programGroups->getRealizedMaterialSbtOffset( instance2 );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index1 );
    EXPECT_EQ( index2, index1 );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForAlphaCutOutTriangles )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshAH = "__anyhit__alphaCutOutMesh";
    const char* const triMeshCH = "__closesthit__mesh";
    auto expectedHitGroupDesc = AllOf( NotNull(), hasHitGroupISAHCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule,
                                                                         triMeshAH, m_phongModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, m_init );
    GeometryInstance instance{};
    instance.primitive             = GeometryPrimitive::TRIANGLE;
    instance.groups.material.flags = MaterialFlags::ALPHA_MAP;

    const uint_t index = m_programGroups->getRealizedMaterialSbtOffset( instance );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForAlphaCutOutTrianglesShared )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshAH = "__anyhit__alphaCutOutMesh";
    const char* const triMeshCH = "__closesthit__mesh";
    auto expectedHitGroupDesc = AllOf( NotNull(), hasHitGroupISAHCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule,
                                                                         triMeshAH, m_phongModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, m_init );
    GeometryInstance instance1{};
    instance1.primitive             = GeometryPrimitive::TRIANGLE;
    instance1.groups.material.flags = MaterialFlags::ALPHA_MAP;
    GeometryInstance instance2{};
    instance2.primitive             = GeometryPrimitive::TRIANGLE;
    instance2.groups.material.flags = MaterialFlags::ALPHA_MAP;

    const uint_t index1 = m_programGroups->getRealizedMaterialSbtOffset( instance1 );
    const uint_t index2 = m_programGroups->getRealizedMaterialSbtOffset( instance2 );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index1 );
    EXPECT_EQ( index2, index1 );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForTexturedTriangles )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshCH = "__closesthit__texturedMesh";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, m_init );
    GeometryInstance instance{};
    instance.primitive             = GeometryPrimitive::TRIANGLE;
    instance.groups.material.flags = MaterialFlags::DIFFUSE_MAP;

    const uint_t index = m_programGroups->getRealizedMaterialSbtOffset( instance );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForTexturedTrianglesShared )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshCH = "__closesthit__texturedMesh";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, m_init );
    GeometryInstance instance1{};
    instance1.primitive             = GeometryPrimitive::TRIANGLE;
    instance1.groups.material.flags = MaterialFlags::DIFFUSE_MAP;
    GeometryInstance instance2{};
    instance2.primitive             = GeometryPrimitive::TRIANGLE;
    instance2.groups.material.flags = MaterialFlags::DIFFUSE_MAP;

    const uint_t index1 = m_programGroups->getRealizedMaterialSbtOffset( instance1 );
    const uint_t index2 = m_programGroups->getRealizedMaterialSbtOffset( instance2 );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index1 );
    EXPECT_EQ( index2, index1 );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForAlphaCutOutTexturedTriangles )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshAH = "__anyhit__alphaCutOutMesh";
    const char* const triMeshCH = "__closesthit__texturedMesh";
    const auto expectedHitGroupDesc = AllOf( NotNull(), hasHitGroupISAHCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule,
                                                                               triMeshAH, m_sceneModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, m_init );
    GeometryInstance instance{};
    instance.primitive             = GeometryPrimitive::TRIANGLE;
    instance.groups.material.flags = MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP;

    const uint_t index = m_programGroups->getRealizedMaterialSbtOffset( instance );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index );
}

TEST_F( TestProgramGroupsInitialized, requestSbtIndexForAlphaCutOutTexturedTrianglesShared )
{
    expectModuleCreatedAfter( m_phongModule, m_init );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).Times( AtLeast( 1 ) ).After( m_init ).WillRepeatedly( Return( m_fakeContext ) );
    const char* const triMeshIS = nullptr;
    const char* const triMeshAH = "__anyhit__alphaCutOutMesh";
    const char* const triMeshCH = "__closesthit__texturedMesh";
    const auto expectedHitGroupDesc = AllOf( NotNull(), hasHitGroupISAHCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule,
                                                                               triMeshAH, m_sceneModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, m_init );
    GeometryInstance instance1{};
    instance1.primitive             = GeometryPrimitive::TRIANGLE;
    instance1.groups.material.flags = MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP;
    GeometryInstance instance2{};
    instance2.primitive             = GeometryPrimitive::TRIANGLE;
    instance2.groups.material.flags = MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP;

    const uint_t index1 = m_programGroups->getRealizedMaterialSbtOffset( instance1 );
    const uint_t index2 = m_programGroups->getRealizedMaterialSbtOffset( instance2 );

    EXPECT_EQ( +HitGroupIndex::REALIZED_MATERIAL_START, index1 );
    EXPECT_EQ( index2, index1 );
}
