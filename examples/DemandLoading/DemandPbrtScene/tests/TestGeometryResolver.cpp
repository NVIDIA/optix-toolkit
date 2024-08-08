#include <GeometryResolver.h>

#include "MockDemandTextureCache.h"
#include "MockGeometryLoader.h"
#include "MockMaterialResolver.h"

#include <DemandTextureCache.h>
#include <FrameStopwatch.h>
#include <MaterialResolver.h>
#include <Options.h>
#include <ProgramGroups.h>
#include <SceneProxy.h>
#include <SceneSyncState.h>

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

using namespace testing;
using namespace otk::testing;
using namespace demandPbrtScene;
using namespace demandPbrtScene::testing;

static OptixDeviceContext fakeOptixDeviceContext()
{
    return reinterpret_cast<OptixDeviceContext>( static_cast<std::intptr_t>( 0xfeedfeed ) );
}

namespace {

class MockProgramGroups : public StrictMock<ProgramGroups>
{
  public:
    ~MockProgramGroups() override = default;

    MOCK_METHOD( void, cleanup, (), ( override ) );
    MOCK_METHOD( uint_t, getRealizedMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( void, initialize, (), ( override ) );
};

class MockProxyFactory : public StrictMock<ProxyFactory>
{
  public:
    ~MockProxyFactory() override = default;

    MOCK_METHOD( SceneProxyPtr, scene, ( GeometryLoaderPtr, SceneDescriptionPtr ), ( override ) );
    MOCK_METHOD( SceneProxyPtr, sceneShape, ( GeometryLoaderPtr, SceneDescriptionPtr, uint_t ), ( override ) );
    MOCK_METHOD( SceneProxyPtr, sceneInstance, ( GeometryLoaderPtr, SceneDescriptionPtr, uint_t ), ( override ) );
    MOCK_METHOD( SceneProxyPtr, sceneInstanceShape, ( GeometryLoaderPtr, SceneDescriptionPtr, uint_t, uint_t ), ( override ) );
    MOCK_METHOD( ProxyFactoryStatistics, getStatistics, (), ( const override ) );
};

class MockSceneProxy : public StrictMock<SceneProxy>
{
  public:
    ~MockSceneProxy() override = default;

    MOCK_METHOD( uint_t, getPageId, (), ( const, override ) );
    MOCK_METHOD( OptixAabb, getBounds, (), ( const, override ) );
    MOCK_METHOD( bool, isDecomposable, (), ( const, override ) );
    MOCK_METHOD( GeometryInstance, createGeometry, ( OptixDeviceContext, CUstream ), ( override ) );
    MOCK_METHOD( std::vector<SceneProxyPtr>, decompose, ( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory ), ( override ) );
};

using MockProgramGroupsPtr      = std::shared_ptr<MockProgramGroups>;
using MockProxyFactoryPtr       = std::shared_ptr<MockProxyFactory>;
using MockSceneProxyPtr         = std::shared_ptr<MockSceneProxy>;

inline MockProxyFactoryPtr createMockProxyFactory()
{
    return std::make_shared<MockProxyFactory>();
}
inline MockSceneProxyPtr createMockSceneProxy()
{
    return std::make_shared<MockSceneProxy>();
}

class TestGeometryResolver : public Test
{
  public:
    ~TestGeometryResolver() override = default;

  protected:
    void SetUp() override;
    void TearDown() override;

    // Dependencies
    Options                   m_options{};
    MockProgramGroupsPtr      m_programGroups{ std::make_shared<MockProgramGroups>() };
    MockGeometryLoaderPtr     m_geometryLoader{ std::make_shared<MockGeometryLoader>() };
    MockProxyFactoryPtr       m_proxyFactory{ std::make_shared<MockProxyFactory>() };
    MockDemandTextureCachePtr m_demandTextureCache{ createMockDemandTextureCache() };
    MockMaterialResolverPtr   m_materialResolver{ createMockMaterialResolver() };
    GeometryResolverPtr       m_resolver{
        createGeometryResolver( m_options, m_programGroups, m_geometryLoader, m_proxyFactory, m_demandTextureCache, m_materialResolver ) };

    // Test data
    CUstream               m_stream{};
    OptixDeviceContext     m_fakeContext{ fakeOptixDeviceContext() };
    OptixTraversableHandle m_fakeProxyTraversable{ 0xbaddf00dU };
    SceneSyncState         m_sync{};
};

void TestGeometryResolver::SetUp()
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0 ) );
}

void TestGeometryResolver::TearDown()
{
    OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
    Test::TearDown();
}

class TestGeometryResolverInitialized : public TestGeometryResolver
{
  protected:
    void SetUp() override;

    SceneDescriptionPtr m_scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    MockSceneProxyPtr   m_sceneProxy{ createMockSceneProxy() };
    ExpectationSet      m_init;
    FrameStopwatch      m_timer{ false };
    const uint_t        m_proxyPageId{ 1234 };
};

void TestGeometryResolverInitialized::SetUp()
{
    TestGeometryResolver::SetUp();
    m_init = EXPECT_CALL( *m_geometryLoader, setSbtIndex( _ ) );
    m_init += EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( _ ) );
    m_init += EXPECT_CALL( *m_proxyFactory, scene( _, _ ) ).WillOnce( Return( m_sceneProxy ) );
    m_init += EXPECT_CALL( *m_sceneProxy, getPageId() ).WillOnce( Return( m_proxyPageId ) );
    m_init += EXPECT_CALL( *m_geometryLoader, createTraversable( _, _ ) ).WillOnce( Return( m_fakeProxyTraversable ) );
    m_resolver->initialize( m_stream, m_fakeContext, m_scene, m_sync );
}

}  // namespace

TEST_F( TestGeometryResolver, initializeCreatesProxyForWholeScene )
{
    EXPECT_CALL( *m_geometryLoader, setSbtIndex( +HitGroupIndex::PROXY_GEOMETRY ) );
    EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) );
    SceneDescriptionPtr scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    MockSceneProxyPtr   proxy{ createMockSceneProxy() };
    EXPECT_CALL( *m_proxyFactory, scene( static_cast<GeometryLoaderPtr>( m_geometryLoader ), scene ) ).WillOnce( Return( proxy ) );
    const uint_t proxyPageId{ 1234 };
    EXPECT_CALL( *proxy, getPageId() ).WillOnce( Return( proxyPageId ) );
    EXPECT_CALL( *m_geometryLoader, createTraversable( m_fakeContext, m_stream ) ).WillOnce( Return( m_fakeProxyTraversable ) );

    m_resolver->initialize( m_stream, m_fakeContext, scene, m_sync );

    ASSERT_FALSE( m_sync.topLevelInstances.empty() );
    EXPECT_EQ( m_fakeProxyTraversable, m_sync.topLevelInstances.back().traversableHandle );
}

TEST_F( TestGeometryResolverInitialized, oneShotNotTriggeredDoesNothing )
{
    m_options.oneShotGeometry = true;
    EXPECT_CALL( *m_geometryLoader, requestedProxyIds() ).Times( 0 ).After( m_init );
    EXPECT_CALL( *m_geometryLoader, clearRequestedProxyIds() ).Times( 0 ).After( m_init );

    const bool result1{ m_resolver->resolveRequestedProxyGeometries( m_stream, m_fakeContext, m_timer, m_sync ) };
    const bool result2{ m_resolver->resolveRequestedProxyGeometries( m_stream, m_fakeContext, m_timer, m_sync ) };

    EXPECT_FALSE( result1 );
    EXPECT_FALSE( result2 );
}

TEST_F( TestGeometryResolverInitialized, oneShotTriggeredRequestsProxies )
{
    m_options.oneShotGeometry = true;
    EXPECT_CALL( *m_geometryLoader, requestedProxyIds() ).After( m_init ).WillOnce( Return( std::vector<uint_t>{} ) );
    EXPECT_CALL( *m_geometryLoader, clearRequestedProxyIds() ).Times( 1 ).After( m_init );

    const bool result1{ m_resolver->resolveRequestedProxyGeometries( m_stream, m_fakeContext, m_timer, m_sync ) };
    m_resolver->resolveOneGeometry();
    const bool result2{ m_resolver->resolveRequestedProxyGeometries( m_stream, m_fakeContext, m_timer, m_sync ) };

    EXPECT_FALSE( result1 );
    EXPECT_FALSE( result2 );
}

TEST_F( TestGeometryResolverInitialized, resolveGeometry )
{
    EXPECT_CALL( *m_geometryLoader, requestedProxyIds() ).After( m_init ).WillOnce( Return( std::vector<uint_t>{ m_proxyPageId } ) );
    EXPECT_CALL( *m_geometryLoader, remove( m_proxyPageId ) ).After( m_init );
    EXPECT_CALL( *m_sceneProxy, isDecomposable() ).After( m_init ).WillOnce( Return( false ) );
    GeometryInstance geomInstance{};
    EXPECT_CALL( *m_sceneProxy, createGeometry( m_fakeContext, m_stream ) ).After( m_init ).WillOnce( Return( geomInstance ) );
    EXPECT_CALL( *m_materialResolver, resolveMaterialForGeometry( m_proxyPageId, _, _ ) ).After( m_init ).WillOnce( Return( false ) );
    EXPECT_CALL( *m_geometryLoader, clearRequestedProxyIds() ).Times( 1 ).After( m_init );
    EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).Times( 1 ).After( m_init );
    OptixTraversableHandle updatedTraversable{ 0xf00df00dU };
    EXPECT_CALL( *m_geometryLoader, createTraversable( m_fakeContext, m_stream ) ).After( m_init ).WillOnce( Return( updatedTraversable ) );
    m_sync.topLevelInstances.resize( 1 );

    const bool result{ m_resolver->resolveRequestedProxyGeometries( m_stream, m_fakeContext, m_timer, m_sync ) };

    EXPECT_TRUE( result );
    EXPECT_EQ( updatedTraversable, m_sync.topLevelInstances[0].traversableHandle );
    const GeometryResolverStatistics stats = m_resolver->getStatistics();
    EXPECT_EQ( 1U, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1U, stats.numGeometriesRealized );
}

TEST_F( TestGeometryResolverInitialized, decomposeProxy )
{
    EXPECT_CALL( *m_geometryLoader, requestedProxyIds() ).After( m_init ).WillOnce( Return( std::vector<uint_t>{ m_proxyPageId } ) );
    EXPECT_CALL( *m_geometryLoader, remove( m_proxyPageId ) ).After( m_init );
    EXPECT_CALL( *m_sceneProxy, isDecomposable() ).After( m_init ).WillOnce( Return( true ) );
    const auto createChildProxy = [&]( uint_t id ) {
        MockSceneProxyPtr child{ createMockSceneProxy() };
        EXPECT_CALL( *child, getPageId() ).After( m_init ).WillOnce( Return( id ) );
        return child;
    };
    const uint_t      childId1{ 1111 };
    const uint_t      childId2{ 2222 };
    MockSceneProxyPtr child1{ createChildProxy( childId1 ) };
    MockSceneProxyPtr child2{ createChildProxy( childId2 ) };
    EXPECT_CALL( *m_sceneProxy, decompose( static_cast<GeometryLoaderPtr>( m_geometryLoader ),
                                           static_cast<ProxyFactoryPtr>( m_proxyFactory ) ) )
        .After( m_init )
        .WillOnce( Return( std::vector<SceneProxyPtr>{ child1, child2 } ) );
    EXPECT_CALL( *m_geometryLoader, clearRequestedProxyIds() ).Times( 1 ).After( m_init );
    EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).Times( 1 ).After( m_init );
    OptixTraversableHandle updatedTraversable{ 0xf00df00dU };
    EXPECT_CALL( *m_geometryLoader, createTraversable( m_fakeContext, m_stream ) ).After( m_init ).WillOnce( Return( updatedTraversable ) );
    m_sync.topLevelInstances.resize( 1 );

    const bool result{ m_resolver->resolveRequestedProxyGeometries( m_stream, m_fakeContext, m_timer, m_sync ) };

    EXPECT_TRUE( result );
    EXPECT_EQ( updatedTraversable, m_sync.topLevelInstances[0].traversableHandle );
    const GeometryResolverStatistics stats = m_resolver->getStatistics();
    EXPECT_EQ( 1U, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 0U, stats.numGeometriesRealized );
}
