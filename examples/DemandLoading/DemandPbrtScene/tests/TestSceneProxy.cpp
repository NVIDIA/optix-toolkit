// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// gtest has to come before pbrt stuff
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "GeometryInstancePrinter.h"
#include "Matchers.h"
#include "MockGeometryLoader.h"
#include "MockMeshLoader.h"

#include <GeometryCache.h>
#include <Options.h>
#include <Params.h>
#include <SceneProxy.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/BitCast.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>

using namespace demandPbrtScene;
using namespace demandPbrtScene::testing;
using namespace otk::testing;
using namespace ::testing;

using P2 = pbrt::Point2f;
using P3 = pbrt::Point3f;
using B3 = pbrt::Bounds3f;

using Stats = ProxyFactoryStatistics;

static void PrintTo( const OptixAabb& value, std::ostream* str )
{
    *str << value;
}

template <typename Thing>
pbrt::Bounds3f transformBounds( const Thing& thing )
{
    return thing.transform( thing.bounds );
}

static otk::pbrt::ShapeDefinition translatedTriangleShape( const pbrt::Vector3f& translation )
{
    const P3 minPt{ 0.0f, 0.0f, 0.0f };
    const P3 maxPt{ 1.0f, 1.0f, 1.0f };
    const B3 bounds{ minPt, maxPt };

    otk::pbrt::PlasticMaterial material{};
    material.Ka = P3{ 0.1f, 0.2f, 0.3f };
    material.Kd = P3{ 0.4f, 0.5f, 0.6f };
    material.Ks = P3{ 0.7f, 0.8f, 0.9f };

    std::vector<P3> vertices{ P3{ 0.0f, 0.0f, 0.0f }, P3{ 1.0f, 0.0f, 0.0f }, P3{ 1.0f, 1.0f, 1.0f } };

    return { "trianglemesh", Translate( translation ), material, bounds, {}, otk::pbrt::TriangleMeshData{ { 0, 1, 2 }, std::move( vertices ) } };
}

static otk::pbrt::ShapeDefinition singleTriangleShape()
{
    return translatedTriangleShape( pbrt::Vector3f{ 1.0f, 2.0f, 3.0f } );
}

static SceneDescriptionPtr singleTriangleScene()
{
    SceneDescriptionPtr        scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    otk::pbrt::ShapeDefinition mesh{ singleTriangleShape() };
    scene->bounds = transformBounds( mesh );
    scene->freeShapes.push_back( mesh );
    return scene;
}

namespace otk {
namespace pbrt {

inline bool operator==( const ObjectDefinition& lhs, const ObjectDefinition& rhs )
{
    return lhs.transform == rhs.transform  //
           && lhs.bounds == rhs.bounds;    //
}

inline std::ostream& operator<<( std::ostream& str, const ObjectDefinition& value )
{
    return str << "ObjectDefinition{ " << value.transform << ", " << value.bounds << " }";
}

}  // namespace pbrt
}  // namespace otk


TEST( TestSceneConstruction, sceneBoundsSingleTriangleScene )
{
    SceneDescriptionPtr scene{ singleTriangleScene() };

    const otk::pbrt::ShapeList& shapes{ scene->freeShapes };
    EXPECT_EQ( 1U, shapes.size() );
    const otk::pbrt::ShapeDefinition& shape{ shapes[0] };
    EXPECT_EQ( scene->bounds, transformBounds( shape ) );
}

static otk::pbrt::ShapeDefinition singleSphereShape()
{
    const P3 minPt{ 0.0f, 0.0f, 0.0f };
    const P3 maxPt{ 1.0f, 1.0f, 1.0f };
    const B3 bounds{ minPt, maxPt };

    otk::pbrt::PlasticMaterial material{};
    material.Ka = P3{ 0.1f, 0.2f, 0.3f };
    material.Kd = P3{ 0.4f, 0.5f, 0.6f };
    material.Ks = P3{ 0.7f, 0.8f, 0.9f };

    std::vector<P3> vertices{ P3{ 0.0f, 0.0f, 0.0f }, P3{ 1.0f, 0.0f, 0.0f }, P3{ 1.0f, 1.0f, 1.0f } };

    otk::pbrt::SphereData sphere;
    sphere.radius = 1.25f;
    sphere.zMin   = -sphere.radius;
    sphere.zMax   = sphere.radius;
    sphere.phiMax = 360.0f;

    pbrt::Vector3f             translation{ 1.0f, 2.0f, 3.0f };
    return { "sphere", Translate( translation ), material, bounds, {}, {}, sphere };
}

static SceneDescriptionPtr singleSphereScene()
{
    SceneDescriptionPtr scene{ std::make_shared<otk::pbrt::SceneDescription>() };

    otk::pbrt::ShapeDefinition shape{ singleSphereShape() };
    scene->freeShapes.push_back( shape );
    scene->bounds = transformBounds( shape );

    return scene;
}

TEST( TestSceneConstruction, sceneBoundsSingleSphereScene )
{
    SceneDescriptionPtr scene{ singleSphereScene() };

    const otk::pbrt::ShapeList& shapes{ scene->freeShapes };
    EXPECT_EQ( 1U, shapes.size() );
    const otk::pbrt::ShapeDefinition& shape{ shapes[0] };
    EXPECT_EQ( scene->bounds, transformBounds( shape ) );
}

namespace otk {
namespace pbrt {

inline bool operator==( const PlyMeshData& lhs, const PlyMeshData& rhs )
{
    return lhs.fileName == rhs.fileName && lhs.loader == rhs.loader;
}

inline bool operator==( const TriangleMeshData& lhs, const TriangleMeshData& rhs )
{
    return lhs.indices == rhs.indices && lhs.points == rhs.points && lhs.normals == rhs.normals && lhs.uvs == rhs.uvs;
}

inline bool operator==( const SphereData& lhs, const SphereData& rhs )
{
    return lhs.radius == rhs.radius && lhs.zMin == rhs.zMin && lhs.zMax == rhs.zMax && lhs.phiMax == rhs.phiMax;
}

inline bool operator==( const ShapeDefinition& lhs, const ShapeDefinition& rhs )
{
    if( lhs.type != rhs.type )
        return false;

    if( lhs.type == "plymesh" )
        return lhs.plyMesh == rhs.plyMesh;

    if( lhs.type == "trianglemesh" )
        return lhs.triangleMesh == rhs.triangleMesh;

    if( lhs.type == "sphere" )
        return lhs.sphere == rhs.sphere;

    return false;
}

}  // namespace pbrt
}  // namespace otk

namespace {

class MockGeometryCache : public StrictMock<GeometryCache>
{
  public:
    ~MockGeometryCache() override = default;

    MOCK_METHOD( GeometryCacheEntry, getShape, (OptixDeviceContext, CUstream, const otk::pbrt::ShapeDefinition&), ( override ) );
    MOCK_METHOD( std::vector<GeometryCacheEntry>,
                 getObject,
                 ( OptixDeviceContext context, CUstream stream, const otk::pbrt::ObjectDefinition& object, const otk::pbrt::ShapeList& shapes ) );
    MOCK_METHOD( GeometryCacheStatistics, getStatistics, (), ( const override ) );
};

using MockGeometryCachePtr = std::shared_ptr<MockGeometryCache>;

}  // namespace

static SceneDescriptionPtr singleTrianglePlyScene( MockMeshLoaderPtr meshLoader )
{
    SceneDescriptionPtr scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    const P3            minPt{ 0.0f, 0.0f, 0.0f };
    const P3            maxPt{ 1.0f, 1.0f, 1.0f };
    const B3            bounds{ minPt, maxPt };

    otk::pbrt::PlasticMaterial material{};
    material.Ka = P3{ 0.1f, 0.2f, 0.3f };
    material.Kd = P3{ 0.4f, 0.5f, 0.6f };
    material.Ks = P3{ 0.7f, 0.8f, 0.9f };

    pbrt::Vector3f             translation{ 1.0f, 2.0f, 3.0f };
    otk::pbrt::ShapeDefinition mesh{
        "plymesh", Translate( translation ), material, bounds, otk::pbrt::PlyMeshData{ "cube-mesh.ply", meshLoader },
        {} };

    scene->bounds = transformBounds( mesh );
    scene->freeShapes.push_back( mesh );
    return scene;
}

TEST( TestSceneConstruction, sceneBoundsSingleTrianglePlyScene )
{
    MockMeshLoaderPtr   meshLoader{ createMockMeshLoader() };
    SceneDescriptionPtr scene{ singleTrianglePlyScene( meshLoader ) };

    const otk::pbrt::ShapeList& shapes{ scene->freeShapes };
    EXPECT_EQ( 1U, shapes.size() );
    const otk::pbrt::ShapeDefinition& shape{ shapes[0] };
    EXPECT_EQ( scene->bounds, transformBounds( shape ) );
}

TEST( TestSceneConstruction, meshDataSingleTrianglePlyScene )
{
    MockMeshLoaderPtr   meshLoader{ createMockMeshLoader() };
    SceneDescriptionPtr scene{ singleTrianglePlyScene( meshLoader ) };

    const otk::pbrt::ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( std::string{ "plymesh" }, shape.type );
    EXPECT_EQ( "cube-mesh.ply", shape.plyMesh.fileName );
    EXPECT_EQ( meshLoader, shape.plyMesh.loader );
}

static SceneDescriptionPtr singleTriangleWithNormalsScene()
{
    SceneDescriptionPtr scene{ singleTriangleScene() };
    std::vector<P3>     normals{ P3{ 0.1f, 0.2f, 0.3f }, P3{ 0.4f, 0.5f, 0.6f }, P3{ 0.7f, 0.8f, 0.9f } };
    scene->freeShapes[0].triangleMesh.normals = std::move( normals );
    return scene;
}

TEST( TestSceneConstruction, constructSingleTriangleWithNormalsScene )
{
    SceneDescriptionPtr scene{ singleTriangleWithNormalsScene() };

    ASSERT_FALSE( scene->freeShapes.empty() );
    const otk::pbrt::ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( "trianglemesh", shape.type );
    const otk::pbrt::TriangleMeshData& mesh{ shape.triangleMesh };
    EXPECT_FALSE( mesh.normals.empty() );
}

static SceneDescriptionPtr singleTriangleWithUVsScene()
{
    SceneDescriptionPtr scene{ singleTriangleScene() };
    std::vector<P2>     uvs{ P2{ 0.0f, 0.0f }, P2{ 1.0f, 0.0f }, P2{ 1.0f, 1.0f } };
    scene->freeShapes[0].triangleMesh.uvs = std::move( uvs );
    return scene;
}

TEST( TestSceneConstruction, constructSingleTriangleWithUVsScene )
{
    SceneDescriptionPtr scene{ singleTriangleWithUVsScene() };

    ASSERT_FALSE( scene->freeShapes.empty() );
    const otk::pbrt::ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( "trianglemesh", shape.type );
    const otk::pbrt::TriangleMeshData& mesh{ shape.triangleMesh };
    EXPECT_FALSE( mesh.uvs.empty() );
}

static SceneDescriptionPtr singleTriangleWithAlphaMapScene()
{
    SceneDescriptionPtr scene{ singleTriangleWithUVsScene() };
    scene->freeShapes[0].material.alphaMapFileName = "alphaMap.png";
    return scene;
}

TEST( TestSceneConstruction, constructSingleTriangleWithAlphaMapScene )
{
    SceneDescriptionPtr scene{ singleTriangleWithAlphaMapScene() };

    ASSERT_FALSE( scene->freeShapes.empty() );
    const otk::pbrt::ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_FALSE( shape.material.alphaMapFileName.empty() );
}

static SceneDescriptionPtr singleTriangleWithDiffuseMapScene()
{
    SceneDescriptionPtr scene{ singleTriangleWithUVsScene() };
    scene->freeShapes[0].material.diffuseMapFileName = "diffuse.png";
    return scene;
}

TEST( TestSceneConstruction, constructSingleDiffuseMapTriangleScene )
{
    SceneDescriptionPtr scene{ singleTriangleWithDiffuseMapScene() };

    ASSERT_FALSE( scene->freeShapes.empty() );
    const otk::pbrt::ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_FALSE( shape.material.diffuseMapFileName.empty() );
}

static SceneDescriptionPtr twoShapeScene()
{
    otk::pbrt::ShapeDefinition shape1{ translatedTriangleShape( pbrt::Vector3f{ 1.0f, 2.0f, 3.0f } ) };
    otk::pbrt::ShapeDefinition shape2{ translatedTriangleShape( pbrt::Vector3f{ -1.0f, -2.0f, -3.0f } ) };

    SceneDescriptionPtr scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    scene->bounds = Union( transformBounds( shape1 ), transformBounds( shape2 ) );
    scene->freeShapes.push_back( shape1 );
    scene->freeShapes.push_back( shape2 );
    return scene;
}

TEST( TestSceneConstruction, sceneBoundsTwoShapeScene )
{
    SceneDescriptionPtr scene{ twoShapeScene() };

    const otk::pbrt::ShapeList& shapes{ scene->freeShapes };
    EXPECT_EQ( 2U, shapes.size() );
    const otk::pbrt::ShapeDefinition& shape1{ shapes[0] };
    const pbrt::Bounds3f              shape1WorldBounds{ transformBounds( shape1 ) };
    EXPECT_TRUE( Overlaps( shape1WorldBounds, scene->bounds ) );
    const otk::pbrt::ShapeDefinition& shape2{ shapes[1] };
    const pbrt::Bounds3f              shape2WorldBounds{ transformBounds( shape2 ) };
    EXPECT_TRUE( Overlaps( shape2WorldBounds, scene->bounds ) );
    EXPECT_EQ( scene->bounds, Union( shape1WorldBounds, shape2WorldBounds ) );
}

static SceneDescriptionPtr singleInstanceSingleShapeScene()
{
    SceneDescriptionPtr         scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    otk::pbrt::ShapeDefinition  shape{ singleTriangleShape() };
    otk::pbrt::ObjectDefinition object;
    object.bounds                     = transformBounds( shape );
    scene->objects["triangle"]        = object;
    scene->instanceCounts["triangle"] = 1;
    otk::pbrt::ObjectInstanceDefinition instance;
    instance.name   = "triangle";
    instance.bounds = transformBounds( object );
    scene->objectInstances.push_back( instance );
    otk::pbrt::ShapeList shapeList;
    shapeList.push_back( shape );
    scene->objectShapes["triangle"] = shapeList;
    scene->bounds                   = transformBounds( instance );
    return scene;
}

TEST( TestSceneConstruction, sceneBoundsSingleInstanceSingleShapeScene )
{
    SceneDescriptionPtr scene{ singleInstanceSingleShapeScene() };

    const otk::pbrt::ShapeList& shapes{ scene->objectShapes["triangle"] };
    pbrt::Bounds3f              expectedInstanceBounds{ transformBounds( shapes[0] ) };
    EXPECT_EQ( expectedInstanceBounds, scene->objectInstances[0].bounds );
    EXPECT_EQ( scene->objectInstances[0].transform( expectedInstanceBounds ), scene->bounds );
}

static SceneDescriptionPtr singleInstanceMultipleShapesScene()
{
    SceneDescriptionPtr         scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    otk::pbrt::ShapeDefinition  shape1{ translatedTriangleShape( pbrt::Vector3f{ 1.0f, 2.0f, 3.0f } ) };
    otk::pbrt::ShapeDefinition  shape2{ translatedTriangleShape( pbrt::Vector3f{ -1.0f, -2.0f, -3.0f } ) };
    otk::pbrt::ObjectDefinition object;
    std::string                 name{ "object" };
    object.bounds               = Union( transformBounds( shape1 ), transformBounds( shape2 ) );
    scene->objects[name]        = object;
    scene->instanceCounts[name] = 1;
    otk::pbrt::ObjectInstanceDefinition instance;
    instance.name   = name;
    instance.bounds = transformBounds( object );
    scene->objectInstances.push_back( instance );
    otk::pbrt::ShapeList shapeList;
    shapeList.push_back( shape1 );
    shapeList.push_back( shape2 );
    scene->objectShapes[name] = shapeList;
    scene->bounds             = transformBounds( instance );
    return scene;
}

TEST( TestSceneConstruction, sceneBoundsSingleInstanceMultipleShapesScene )
{
    SceneDescriptionPtr scene{ singleInstanceMultipleShapesScene() };

    const otk::pbrt::ShapeList& shapes{ scene->objectShapes["object"] };
    pbrt::Bounds3f expectedInstanceBounds{ Union( transformBounds( shapes[0] ), transformBounds( shapes[1] ) ) };
    EXPECT_EQ( expectedInstanceBounds, scene->objectInstances[0].bounds );
    EXPECT_EQ( scene->objectInstances[0].transform( expectedInstanceBounds ), scene->bounds );
}

static SceneDescriptionPtr singleInstanceSingleShapeSingleFreeShapeScene()
{
    SceneDescriptionPtr         scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    otk::pbrt::ShapeDefinition  shape1{ translatedTriangleShape( pbrt::Vector3f{ 1.0f, 2.0f, 3.0f } ) };
    otk::pbrt::ObjectDefinition object;
    object.bounds = transformBounds( shape1 );
    std::string name{ "object" };
    scene->objects[name]        = object;
    scene->instanceCounts[name] = 1;
    otk::pbrt::ObjectInstanceDefinition instance;
    instance.name      = name;
    instance.bounds    = transformBounds( object );
    instance.transform = Translate( pbrt::Vector3f( -5.0f, -10.0f, -15.0f ) );
    scene->objectInstances.push_back( instance );
    otk::pbrt::ShapeList shapeList;
    shapeList.push_back( shape1 );
    scene->objectShapes[name] = shapeList;

    otk::pbrt::ShapeDefinition shape2{ translatedTriangleShape( pbrt::Vector3f{ -1.0f, -2.0f, -3.0f } ) };
    scene->freeShapes.push_back( shape2 );
    scene->bounds = Union( transformBounds( instance ), transformBounds( shape2 ) );
    return scene;
}

TEST( TestSceneConstruction, sceneBoundsSingleInstanceSingleShapeSingleFreeShapeScene )
{
    SceneDescriptionPtr scene{ singleInstanceSingleShapeSingleFreeShapeScene() };

    const otk::pbrt::ShapeList freeShapes{ scene->freeShapes };
    ASSERT_FALSE( freeShapes.empty() );
    const otk::pbrt::ShapeList& instanceShapes{ scene->objectShapes["object"] };
    ASSERT_FALSE( instanceShapes.empty() );
    pbrt::Bounds3f expectedInstanceBounds{ transformBounds( instanceShapes[0] ) };
    EXPECT_EQ( expectedInstanceBounds, scene->objectInstances[0].bounds );
    pbrt::Bounds3f expectedFreeShapeBounds{ transformBounds( freeShapes[0] ) };
    EXPECT_TRUE( Overlaps( expectedFreeShapeBounds, scene->bounds ) );
    pbrt::Bounds3f expectedObjectInstanceBounds{ scene->objectInstances[0].transform( expectedInstanceBounds ) };
    EXPECT_TRUE( Overlaps( expectedObjectInstanceBounds, scene->bounds ) )
        << expectedObjectInstanceBounds << " not in " << scene->bounds;
}

static SceneDescriptionPtr multipleInstancesSingleShape()
{
    SceneDescriptionPtr         scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    otk::pbrt::ShapeDefinition  shape{ translatedTriangleShape( pbrt::Vector3f{ 1.0f, 2.0f, 3.0f } ) };
    otk::pbrt::ObjectDefinition object;
    object.bounds = transformBounds( shape );
    std::string name{ "object" };
    scene->objects[name] = object;
    otk::pbrt::ShapeList shapeList;
    shapeList.push_back( shape );
    scene->objectShapes[name] = shapeList;
    const auto createInstance = [&]( const pbrt::Vector3f& translation ) {
        otk::pbrt::ObjectInstanceDefinition instance;
        instance.name      = name;
        instance.bounds    = transformBounds( object );
        instance.transform = Translate( translation );
        scene->objectInstances.push_back( instance );
        scene->instanceCounts[name]++;
    };
    createInstance( pbrt::Vector3f( -5.0f, -10.0f, -15.0f ) );
    createInstance( pbrt::Vector3f( 10.0f, 10.0f, 10.0f ) );

    const auto& ins1{ scene->objectInstances[0] };
    const auto& ins2{ scene->objectInstances[1] };
    scene->bounds = Union( transformBounds( ins1 ), transformBounds( ins2 ) );
    return scene;
}

TEST( TestSceneConstruction, sceneBoundsMultipleInstancesSingleShape )
{
    SceneDescriptionPtr scene{ multipleInstancesSingleShape() };

    ASSERT_TRUE( scene->freeShapes.empty() );
    const otk::pbrt::ShapeList& instanceShapes{ scene->objectShapes["object"] };
    ASSERT_FALSE( instanceShapes.empty() );
    pbrt::Bounds3f expectedShapeBounds{ transformBounds( instanceShapes[0] ) };
    EXPECT_EQ( expectedShapeBounds, scene->objectInstances[0].bounds );
    EXPECT_EQ( expectedShapeBounds, scene->objectInstances[1].bounds );
    pbrt::Bounds3f ins1Bounds{ transformBounds( scene->objectInstances[0] ) };
    pbrt::Bounds3f ins2Bounds{ transformBounds( scene->objectInstances[1] ) };
    EXPECT_NE( ins1Bounds, ins2Bounds );
    EXPECT_TRUE( Overlaps( ins1Bounds, scene->bounds ) ) << ins1Bounds << " not in " << scene->bounds;
    EXPECT_TRUE( Overlaps( ins2Bounds, scene->bounds ) ) << ins2Bounds << " not in " << scene->bounds;
}

static void identity( float ( &result )[12] )
{
    static const float matrix[12]{
        1.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 1.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 1.0f, 0.0f   //
    };
    std::copy( std::begin( matrix ), std::end( matrix ), std::begin( result ) );
}

namespace {

class TestSceneProxy : public Test
{
  protected:
    void SetUp() override
    {
        m_options.proxyGranularity     = ProxyGranularity::FINE;
        m_accelSizes.tempSizeInBytes   = 1234U;
        m_accelSizes.outputSizeInBytes = 5678U;
    }

    Expectation expectProxyBoundsAdded( const ::pbrt::Bounds3f& bounds, uint_t pageId ) const
    {
        return EXPECT_CALL( *m_geometryLoader, add( toOptixAabb( bounds ) ) ).WillOnce( Return( pageId ) );
    }
    template <typename Thing>
    Expectation expectProxyAdded( const Thing& thing, uint_t pageId )
    {
        return expectProxyBoundsAdded( transformBounds( thing ), pageId );
    }
    Expectation expectProxyBoundsAddedAfter( const ::pbrt::Bounds3f& bounds, uint_t pageId, ExpectationSet before ) const
    {
        return EXPECT_CALL( *m_geometryLoader, add( toOptixAabb( bounds ) ) ).After( before ).WillOnce( Return( pageId ) );
    }
    template <typename Thing>
    Expectation expectProxyAddedAfter( const Thing& thing, uint_t pageId, ExpectationSet before )
    {
        return expectProxyBoundsAddedAfter( transformBounds( thing ), pageId, before );
    }

    GeometryCacheEntry expectShapeFromCache( const otk::pbrt::ShapeDefinition& shape )
    {
        GeometryCacheEntry entry{};
        entry.accelBuffer = CUdeviceptr{ 0xf00dbaadf00dbaadULL };
        entry.traversable = m_fakeGeometryAS;
        if( shape.type == "trianglemesh" )
        {
            if( !shape.triangleMesh.normals.empty() )
            {
                entry.devNormals = otk::bit_cast<TriangleNormals*>( 0xbaadf00dbaaabaaaULL );
            }
            if( !shape.triangleMesh.uvs.empty() )
            {
                entry.devUVs = otk::bit_cast<TriangleUVs*>( 0xbaaabaaaf00dbaadULL );
            }
        }
        EXPECT_CALL( *m_geometryCache, getShape( m_fakeContext, m_stream, shape ) ).WillOnce( Return( entry ) );
        return entry;
    }

    CUstream               m_stream{ otk::bit_cast<CUstream>( 0xbaadfeedfeedfeedULL ) };
    uint_t                 m_pageId{ 10 };
    MockGeometryLoaderPtr  m_geometryLoader{ createMockGeometryLoader() };
    MockGeometryCachePtr   m_geometryCache{ std::make_shared<MockGeometryCache>() };
    Options                m_options{};
    ProxyFactoryPtr        m_factory{ createProxyFactory( m_options, m_geometryLoader, m_geometryCache ) };
    SceneProxyPtr          m_proxy;
    SceneDescriptionPtr    m_scene;
    OptixDeviceContext     m_fakeContext{ otk::bit_cast<OptixDeviceContext>( 0xf00df00dULL ) };
    OptixAccelBufferSizes  m_accelSizes{};
    OptixTraversableHandle m_fakeGeometryAS{ 0xfeedf00dU };
};

}  // namespace

TEST_F( TestSceneProxy, constructWholeSceneProxyForSingleTriangleMesh )
{
    m_scene = singleTriangleScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );

    m_proxy = m_factory->scene( m_scene );
    const Stats stats{ m_factory->getStatistics() };

    ASSERT_TRUE( m_proxy );
    EXPECT_EQ( m_pageId, m_proxy->getPageId() );
    const OptixAabb expectedBounds{ toOptixAabb( m_scene->bounds ) };
    EXPECT_EQ( expectedBounds, m_proxy->getBounds() ) << expectedBounds << " ! " << m_proxy->getBounds();
    EXPECT_FALSE( m_proxy->isDecomposable() );
    EXPECT_EQ( 1, stats.numGeometryProxiesCreated );
}

TEST_F( TestSceneProxy, constructTriangleASForSingleTriangleMesh )
{
    m_scene = singleTriangleScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->freeShapes[0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
}

TEST_F( TestSceneProxy, constructTriangleASForSingleTriangleMeshWithNormals )
{
    m_scene = singleTriangleWithNormalsScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->freeShapes[0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( entry.devNormals, geom.devNormals );
    EXPECT_NE( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
}

TEST_F( TestSceneProxy, constructTriangleASForSingleTriangleMeshWithUVs )
{
    m_scene = singleTriangleWithUVsScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->freeShapes[0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( entry.devUVs, geom.devUVs );
    EXPECT_NE( nullptr, geom.devUVs );
}

TEST_F( TestSceneProxy, constructTriangleASForSingleTriangleMeshWithAlphaMap )
{
    m_scene = singleTriangleWithAlphaMapScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->freeShapes[0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( entry.devUVs, geom.devUVs );
    EXPECT_NE( nullptr, geom.devUVs );
    EXPECT_FALSE( geom.alphaMapFileName.empty() );
    EXPECT_TRUE( geom.diffuseMapFileName.empty() );
    EXPECT_EQ( geom.material.flags, MaterialFlags::ALPHA_MAP );
}

TEST_F( TestSceneProxy, constructTriangleASForSingleTriangleMeshWithDiffuseMap )
{
    m_scene = singleTriangleWithDiffuseMapScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->freeShapes[0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( entry.devUVs, geom.devUVs );
    EXPECT_NE( nullptr, geom.devUVs );
    EXPECT_TRUE( geom.alphaMapFileName.empty() );
    EXPECT_FALSE( geom.diffuseMapFileName.empty() );
    EXPECT_EQ( geom.material.flags, MaterialFlags::DIFFUSE_MAP );
}

TEST_F( TestSceneProxy, constructWholeSceneProxyForMultipleShapes )
{
    m_scene = twoShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );

    m_proxy = m_factory->scene( m_scene );

    ASSERT_TRUE( m_proxy );
    EXPECT_EQ( m_pageId, m_proxy->getPageId() );
    EXPECT_EQ( toOptixAabb( m_scene->bounds ), m_proxy->getBounds() );
    EXPECT_TRUE( m_proxy->isDecomposable() );
}

TEST_F( TestSceneProxy, decomposeProxyForMultipleShapes )
{
    m_scene = twoShapeScene();
    EXPECT_EQ( m_scene->freeShapes[0].bounds, m_scene->freeShapes[1].bounds );
    EXPECT_NE( m_scene->freeShapes[0].transform, m_scene->freeShapes[1].transform );
    ExpectationSet first{ expectProxyBoundsAdded( m_scene->bounds, m_pageId ) };
    m_proxy = m_factory->scene( m_scene );
    const uint_t shape1PageId{ 1111 };
    const uint_t shape2PageId{ 2222 };
    expectProxyAdded( m_scene->freeShapes[0], shape1PageId );
    expectProxyAdded( m_scene->freeShapes[1], shape2PageId );

    std::vector<SceneProxyPtr> parts{ m_proxy->decompose( m_factory ) };

    ASSERT_FALSE( parts.empty() );
    EXPECT_TRUE( std::none_of( parts.begin(), parts.end(), []( SceneProxyPtr proxy ) { return proxy->isDecomposable(); } ) );
    EXPECT_EQ( shape1PageId, parts[0]->getPageId() );
    EXPECT_EQ( shape2PageId, parts[1]->getPageId() );
    auto transformedBounds = [&]( int index ) { return toOptixAabb( transformBounds( m_scene->freeShapes[index] ) ); };
    const OptixAabb expectedBounds1{ transformedBounds( 0 ) };
    EXPECT_EQ( expectedBounds1, parts[0]->getBounds() ) << expectedBounds1 << " != " << parts[0]->getBounds();
    const OptixAabb expectedBounds2{ transformedBounds( 1 ) };
    EXPECT_EQ( expectedBounds2, parts[1]->getBounds() ) << expectedBounds2 << " != " << parts[1]->getBounds();
}

TEST_F( TestSceneProxy, constructTriangleASForSecondMesh )
{
    m_scene = twoShapeScene();
    ExpectationSet first{ expectProxyBoundsAdded( m_scene->bounds, m_pageId ) };
    m_proxy = m_factory->scene( m_scene );
    const uint_t   shape1PageId{ 1111 };
    const uint_t   shape2PageId{ 2222 };
    ExpectationSet second;
    second += expectProxyAddedAfter( m_scene->freeShapes[0], shape1PageId, first );
    second += expectProxyAddedAfter( m_scene->freeShapes[1], shape2PageId, first );
    std::vector<SceneProxyPtr> parts{ m_proxy->decompose( m_factory ) };
    GeometryCacheEntry         entry{ expectShapeFromCache( m_scene->freeShapes[1] ) };
    float                      expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = -1.0f;
    expectedTransform[7]  = -2.0f;
    expectedTransform[11] = -3.0f;

    const GeometryInstance geom{ parts[1]->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
}

TEST_F( TestSceneProxy, constructWholeSceneProxyForSingleInstanceWithSingleShape )
{
    m_scene = singleInstanceSingleShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );

    m_proxy = m_factory->scene( m_scene );

    ASSERT_TRUE( m_proxy );
    EXPECT_EQ( m_pageId, m_proxy->getPageId() );
    EXPECT_EQ( toOptixAabb( m_scene->bounds ), m_proxy->getBounds() );
    EXPECT_FALSE( m_proxy->isDecomposable() );
}

TEST_F( TestSceneProxy, geometryForSingleInstanceWithSingleShape )
{
    m_scene = singleInstanceSingleShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->objectShapes["triangle"][0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( nullptr, entry.devNormals );
    EXPECT_EQ( nullptr, entry.devUVs );
    EXPECT_NE( CUdeviceptr{}, entry.accelBuffer );
    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
}

TEST_F( TestSceneProxy, constructWholeSceneProxyForSingleInstanceWithMultipleShape )
{
    m_scene = singleInstanceMultipleShapesScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );

    m_proxy = m_factory->scene( m_scene );

    ASSERT_TRUE( m_proxy );
    EXPECT_EQ( m_pageId, m_proxy->getPageId() );
    const OptixAabb expectedBounds{ toOptixAabb( m_scene->bounds ) };
    EXPECT_EQ( expectedBounds, m_proxy->getBounds() ) << expectedBounds << " != " << m_proxy->getBounds();
    EXPECT_TRUE( m_proxy->isDecomposable() );
}

TEST_F( TestSceneProxy, decomposeWholeSceneProxyForSingleInstanceWithMultipleShape )
{
    m_scene = singleInstanceMultipleShapesScene();
    ExpectationSet first{ expectProxyBoundsAdded( m_scene->bounds, m_pageId ) };
    m_proxy = m_factory->scene( m_scene );
    const uint_t                shape1PageId{ 1111 };
    const uint_t                shape2PageId{ 2222 };
    const otk::pbrt::ShapeList& objectShapes{ m_scene->objectShapes["object"] };
    expectProxyAddedAfter( objectShapes[0], shape1PageId, first );
    expectProxyAddedAfter( objectShapes[1], shape2PageId, first );

    std::vector<SceneProxyPtr> parts{ m_proxy->decompose( m_factory ) };

    ASSERT_FALSE( parts.empty() );
    EXPECT_TRUE( std::none_of( parts.begin(), parts.end(), []( SceneProxyPtr proxy ) { return proxy->isDecomposable(); } ) );
    EXPECT_EQ( shape1PageId, parts[0]->getPageId() );
    EXPECT_EQ( shape2PageId, parts[1]->getPageId() );
    auto transformedBounds{ [&]( int index ) { return toOptixAabb( transformBounds( objectShapes[index] ) ); } };
    EXPECT_EQ( transformedBounds( 0 ), parts[0]->getBounds() ) << transformedBounds( 0 ) << " != " << parts[0]->getBounds();
    EXPECT_EQ( transformedBounds( 1 ), parts[1]->getBounds() ) << transformedBounds( 1 ) << " != " << parts[1]->getBounds();
}

TEST_F( TestSceneProxy, constructWholeSceneProxyForSingleInstanceAndSingleFreeShape )
{
    m_scene = singleInstanceSingleShapeSingleFreeShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );

    m_proxy = m_factory->scene( m_scene );

    ASSERT_TRUE( m_proxy );
    EXPECT_EQ( m_pageId, m_proxy->getPageId() );
    const OptixAabb expectedBounds{ toOptixAabb( m_scene->bounds ) };
    EXPECT_EQ( expectedBounds, m_proxy->getBounds() ) << expectedBounds << " != " << m_proxy->getBounds();
    EXPECT_TRUE( m_proxy->isDecomposable() );
}

TEST_F( TestSceneProxy, decomposeWholeSceneProxyForSingleInstanceSingleShapeSingleFreeShapeScene )
{
    m_scene = singleInstanceSingleShapeSingleFreeShapeScene();
    ExpectationSet first{ expectProxyBoundsAdded( m_scene->bounds, m_pageId ) };
    m_proxy = m_factory->scene( m_scene );
    const uint_t shape1PageId{ 1111 };
    const uint_t shape2PageId{ 2222 };
    expectProxyAddedAfter( m_scene->objectInstances[0], shape1PageId, first );
    expectProxyAddedAfter( m_scene->freeShapes[0], shape2PageId, first );

    std::vector<SceneProxyPtr> parts{ m_proxy->decompose( m_factory ) };

    ASSERT_EQ( 2U, parts.size() );
    EXPECT_TRUE( std::none_of( parts.begin(), parts.end(), []( SceneProxyPtr proxy ) { return proxy->isDecomposable(); } ) );
    EXPECT_EQ( shape1PageId, parts[0]->getPageId() );
    EXPECT_EQ( shape2PageId, parts[1]->getPageId() );
    EXPECT_FALSE( parts[0]->isDecomposable() );
    EXPECT_FALSE( parts[1]->isDecomposable() );
    const OptixAabb instanceBounds{ toOptixAabb( transformBounds( m_scene->objectInstances[0] ) ) };
    EXPECT_EQ( instanceBounds, parts[0]->getBounds() ) << instanceBounds << " != " << parts[0]->getBounds();
    const OptixAabb freeShapeBounds{ toOptixAabb( transformBounds( m_scene->freeShapes[0] ) ) };
    EXPECT_EQ( freeShapeBounds, parts[1]->getBounds() ) << freeShapeBounds << " != " << parts[1]->getBounds();
}

TEST_F( TestSceneProxy, constructTriangleASForSinglePlyMesh )
{
    MockMeshLoaderPtr meshLoader{ createMockMeshLoader() };
    EXPECT_CALL( *meshLoader, getMeshInfo() ).WillOnce( Return( otk::pbrt::MeshInfo() ) );

    m_scene = singleTrianglePlyScene( meshLoader );
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->freeShapes[0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, geom.instance.sbtOffset );
    EXPECT_EQ( m_fakeGeometryAS, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
}

TEST_F( TestSceneProxy, multipleInstancesSingleShapeGeometry )
{
    m_scene = multipleInstancesSingleShape();
    ExpectationSet first{ expectProxyBoundsAdded( m_scene->bounds, 1111 ) };
    m_proxy = m_factory->scene( m_scene );
    expectProxyAddedAfter( m_scene->objectInstances[0], 2222, first );
    expectProxyAddedAfter( m_scene->objectInstances[1], 3333, first );

    std::vector<SceneProxyPtr> parts{ m_proxy->decompose( m_factory ) };

    const OptixAabb shape1Bounds{ toOptixAabb( transformBounds( m_scene->objectInstances[0] ) ) };
    const OptixAabb shape2Bounds{ toOptixAabb( transformBounds( m_scene->objectInstances[1] ) ) };
    EXPECT_EQ( shape1Bounds, parts[0]->getBounds() ) << shape1Bounds << " != " << parts[0]->getBounds();
    EXPECT_EQ( shape2Bounds, parts[1]->getBounds() ) << shape2Bounds << " != " << parts[1]->getBounds();
}

TEST_F( TestSceneProxy, constructProxyForSingleSphere )
{
    m_scene = singleSphereScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );

    m_proxy = m_factory->scene( m_scene );

    ASSERT_TRUE( m_proxy );
    EXPECT_EQ( m_pageId, m_proxy->getPageId() );
    const OptixAabb expectedBounds{ toOptixAabb( m_scene->bounds ) };
    EXPECT_EQ( expectedBounds, m_proxy->getBounds() ) << expectedBounds << " != " << m_proxy->getBounds();
    EXPECT_FALSE( m_proxy->isDecomposable() );
}

TEST_F( TestSceneProxy, constructSphereASForSingleSphere )
{
    m_scene = singleSphereScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->scene( m_scene );
    const GeometryCacheEntry entry{ expectShapeFromCache( m_scene->freeShapes[0] ) };
    float                    expectedTransform[12];
    identity( expectedTransform );
    expectedTransform[3]  = 1.0f;
    expectedTransform[7]  = 2.0f;
    expectedTransform[11] = 3.0f;

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );

    EXPECT_EQ( entry.accelBuffer, geom.accelBuffer );
    EXPECT_TRUE( isSameTransform( expectedTransform, geom.instance.transform ) );
    EXPECT_EQ( +HitGroupIndex::PROXY_MATERIAL_SPHERE, geom.instance.sbtOffset );
    EXPECT_EQ( entry.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( 255U, geom.instance.visibilityMask );
    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), geom.material.Ka );
    EXPECT_EQ( make_float3( 0.4f, 0.5f, 0.6f ), geom.material.Kd );
    EXPECT_EQ( make_float3( 0.7f, 0.8f, 0.9f ), geom.material.Ks );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
}

static SceneDescriptionPtr singleInstanceTwoTriangleShapeScene()
{
    SceneDescriptionPtr  scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    otk::pbrt::ShapeList shapeList;
    shapeList.push_back( singleTriangleShape() );
    shapeList.push_back( singleTriangleShape() );
    otk::pbrt::ShapeDefinition& shape1{ shapeList[0] };
    otk::pbrt::ShapeDefinition& shape2{ shapeList[1] };
    shape2.transform = Translate( ::pbrt::Vector3f( 1.0f, 1.0f, 1.0f ) );
    otk::pbrt::ObjectDefinition object;
    object.bounds = Union( transformBounds( shape1 ), transformBounds( shape2 ) );
    std::string name{ "triangle" };
    scene->objects[name]        = object;
    scene->instanceCounts[name] = 1;
    otk::pbrt::ObjectInstanceDefinition instance;
    instance.name   = name;
    instance.bounds = transformBounds( object );
    scene->objectInstances.push_back( instance );
    scene->objectShapes[name] = shapeList;
    scene->bounds             = transformBounds( instance );
    return scene;
}

TEST_F( TestSceneProxy, fineObjectInstanceDecomposable )
{
    m_options.proxyGranularity = ProxyGranularity::FINE;
    m_scene = singleInstanceTwoTriangleShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->sceneInstance( m_scene, 0 );

    EXPECT_TRUE( m_proxy->isDecomposable() );
}

TEST_F( TestSceneProxy, fineObjectInstanceCreateGeometryIsError )
{
    m_options.proxyGranularity = ProxyGranularity::FINE;
    m_scene                    = singleInstanceTwoTriangleShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->sceneInstance( m_scene, 0 );

    EXPECT_THROW( m_proxy->createGeometry( m_fakeContext, m_stream ), std::runtime_error );
}

TEST_F( TestSceneProxy, coarseObjectInstanceAllShapesSamePrimitiveNotDecomposable )
{
    m_options.proxyGranularity = ProxyGranularity::COARSE;
    m_scene = singleInstanceTwoTriangleShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->sceneInstance( m_scene, 0 );

    const bool decomposable{ m_proxy->isDecomposable() };
    const OptixAabb bounds{ m_proxy->getBounds() };

    EXPECT_FALSE( decomposable );
    EXPECT_EQ( toOptixAabb( m_scene->bounds ), bounds );
}

TEST_F( TestSceneProxy, coarseObjectInstanceAllShapesSamePrimitiveYieldsSingleGeometry )
{
    m_options.proxyGranularity = ProxyGranularity::COARSE;
    m_scene = singleInstanceTwoTriangleShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->sceneInstance( m_scene, 0 );
    GeometryCacheEntry triangles{};
    triangles.accelBuffer = 0xdeadbeefULL;
    triangles.traversable = m_fakeGeometryAS;
    triangles.primitive = GeometryPrimitive::TRIANGLE;
    std::vector<GeometryCacheEntry> entries{ triangles };
    const std::string&              name{ m_scene->objects.begin()->first };
    EXPECT_CALL( *m_geometryCache, getObject( m_fakeContext, m_stream, m_scene->objects[name], m_scene->objectShapes[name] ) )
        .WillOnce( Return( entries ) );

    const GeometryInstance geom{ m_proxy->createGeometry( m_fakeContext, m_stream ) };

    EXPECT_EQ( triangles.accelBuffer, geom.accelBuffer );
    EXPECT_EQ( triangles.primitive, geom.primitive );
    EXPECT_EQ( triangles.traversable, geom.instance.traversableHandle );
    EXPECT_EQ( PhongMaterial{}, geom.material );
    EXPECT_TRUE( geom.diffuseMapFileName.empty() );
    EXPECT_TRUE( geom.alphaMapFileName.empty() );
    EXPECT_EQ( triangles.devNormals, geom.devNormals );
    EXPECT_EQ( triangles.devUVs, geom.devUVs );
}

static SceneDescriptionPtr singleInstanceOneTriangleOneSphereShapeScene()
{
    SceneDescriptionPtr  scene{ std::make_shared<otk::pbrt::SceneDescription>() };
    otk::pbrt::ShapeList shapeList;
    shapeList.push_back( singleTriangleShape() );
    shapeList.push_back( singleSphereShape() );
    otk::pbrt::ShapeDefinition& shape1{ shapeList[0] };
    otk::pbrt::ShapeDefinition& shape2{ shapeList[1] };
    shape2.transform = Translate( ::pbrt::Vector3f( 1.0f, 1.0f, 1.0f ) );
    otk::pbrt::ObjectDefinition object;
    object.bounds                     = Union( transformBounds( shape1 ), transformBounds( shape2 ) );
    scene->objects["triangle"]        = object;
    scene->instanceCounts["triangle"] = 1;
    otk::pbrt::ObjectInstanceDefinition instance;
    instance.name   = "triangle";
    instance.bounds = transformBounds( object );
    scene->objectInstances.push_back( instance );
    scene->objectShapes["triangle"] = shapeList;
    scene->bounds                   = transformBounds( instance );
    return scene;
}

TEST_F( TestSceneProxy, coarseObjectInstanceSomeShapesDifferentPrimitiveDecomposable )
{
    m_options.proxyGranularity = ProxyGranularity::COARSE;
    m_scene = singleInstanceOneTriangleOneSphereShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->sceneInstance( m_scene, 0 );

    EXPECT_TRUE( m_proxy->isDecomposable() );
}

TEST_F( TestSceneProxy, coarseObjectInstanceSomeShapesDifferentPrimitiveDecomposedMultipleProxies )
{
    m_options.proxyGranularity = ProxyGranularity::COARSE;
    m_scene = singleInstanceOneTriangleOneSphereShapeScene();
    expectProxyBoundsAdded( m_scene->bounds, m_pageId );
    m_proxy = m_factory->sceneInstance( m_scene, 0 );

    EXPECT_TRUE( m_proxy->isDecomposable() );
}
