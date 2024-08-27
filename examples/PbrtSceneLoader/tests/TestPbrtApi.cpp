// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// gtest has to be included before any pbrt junk
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core/geometry.h>
#include <core/transform.h>

#include <OptiXToolkit/PbrtApi/PbrtApi.h>

#include <OptiXToolkit/PbrtSceneLoader/Logger.h>
#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>


using namespace otk::pbrt;
using namespace testing;

using Point2  = pbrt::Point2f;
using Point3  = pbrt::Point3f;
using Bounds3 = pbrt::Bounds3f;
using Vector3 = pbrt::Vector3f;

static const char* const g_programName{ "TestPbrtSceneLoader" };

inline pbrt::Transform translate( float x, float y, float z )
{
    return Translate( Vector3( x, y, z ) );
}

namespace {

class MockMeshInfoReader : public StrictMock<MeshInfoReader>
{
  public:
    MOCK_METHOD( MeshInfo, read, ( const std::string& fileName ), ( override ) );
    MOCK_METHOD( MeshLoaderPtr, getLoader, ( const std::string& fileName ), ( override ) );
};

class MockMeshLoader : public StrictMock<MeshLoader>
{
  public:
    MOCK_METHOD( MeshInfo, getMeshInfo, (), ( const, override ) );
    MOCK_METHOD( void, load, ( MeshData & buffers ), ( override ) );
};

class MockLogger : public Logger
{
  public:
    ~MockLogger() override = default;

    MOCK_METHOD( void, start, ( const char* programName ), ( override ) );
    MOCK_METHOD( void, stop, (), ( override ) );
    MOCK_METHOD( void, info, ( std::string text, const char* file, int line ), ( const ) );
    MOCK_METHOD( void, warning, ( std::string text, const char* file, int line ), ( const override ) );
    MOCK_METHOD( void, error, ( std::string text, const char* file, int line ), ( const override ) );
};

class MockStartStopLogger : public MockLogger
{
  public:
    MockStartStopLogger()
    {
        ExpectationSet start{ EXPECT_CALL( *this, start( _ ) ).Times( 1 ) };
        EXPECT_CALL( *this, info( _, _, _ ) ).Times( AnyNumber() ).After( start );
        EXPECT_CALL( *this, stop() ).Times( 1 ).After( start );
    }
    ~MockStartStopLogger() override = default;
};

class TestPbrtApiConstruction : public Test
{
  protected:
    std::shared_ptr<StrictMock<MockLogger>> m_logger{ std::make_shared<StrictMock<MockLogger>>() };
};

class TestPbrtApi : public Test
{
  protected:
    void configureMeshOneInfo( unsigned int numTimes );
    void expectWarnings( int count );

    std::shared_ptr<MockMeshInfoReader>  m_mockMeshInfoReader{ std::make_shared<MockMeshInfoReader>() };
    std::shared_ptr<MockStartStopLogger> m_mockLogger{ std::make_shared<StrictMock<MockStartStopLogger>>() };
    std::shared_ptr<MockMeshLoader>      m_mockLoader{ std::make_shared<MockMeshLoader>() };

    std::shared_ptr<SceneLoader> m_api{ createSceneLoader( g_programName, m_mockLogger, m_mockMeshInfoReader ) };
    Bounds3                      m_meshBounds{ Point3( -1.0f, -2.0f, -3.0f ), Point3( 4.0f, 5.0f, 6.0f ) };
    MeshInfo                     m_meshInfo{};
};

void TestPbrtApi::configureMeshOneInfo( unsigned numTimes )
{
    m_meshInfo.minCoord[0] = m_meshBounds.pMin.x;
    m_meshInfo.minCoord[1] = m_meshBounds.pMin.y;
    m_meshInfo.minCoord[2] = m_meshBounds.pMin.z;
    m_meshInfo.maxCoord[0] = m_meshBounds.pMax.x;
    m_meshInfo.maxCoord[1] = m_meshBounds.pMax.y;
    m_meshInfo.maxCoord[2] = m_meshBounds.pMax.z;
    EXPECT_CALL( *m_mockMeshInfoReader, read( EndsWith( "mesh_00001.ply" ) ) ).Times( numTimes ).WillRepeatedly( Return( m_meshInfo ) );
    EXPECT_CALL( *m_mockMeshInfoReader, getLoader( EndsWith( "mesh_00001.ply" ) ) ).Times( numTimes ).WillRepeatedly( Return( m_mockLoader ) );
}

void TestPbrtApi::expectWarnings( int count )
{
    EXPECT_CALL( *m_mockLogger, warning( _, _, _ ) ).Times( count );
}

class TestPbrtApiEmptyScene : public TestPbrtApi
{
  protected:
    void SetUp() override { m_scene = m_api->parseString( "" ); }

    SceneDescriptionPtr m_scene;
};

}  // namespace


TEST_F( TestPbrtApiConstruction, constructorStartsLogging )
{
    EXPECT_CALL( *m_logger, start( StrEq( g_programName ) ) ).Times( 1 );
    EXPECT_CALL( *m_logger, stop() ).Times( AnyNumber() );

    {
        std::shared_ptr<SceneLoader> loader( createSceneLoader( g_programName, m_logger, nullptr ) );
    }
}

TEST_F( TestPbrtApiConstruction, constructorSetsApi )
{
    EXPECT_EQ( nullptr, getApi() );
    EXPECT_CALL( *m_logger, start( _ ) ).Times( AnyNumber() );
    EXPECT_CALL( *m_logger, stop() ).Times( AnyNumber() );

    std::shared_ptr<SceneLoader> loader( createSceneLoader( g_programName, m_logger, nullptr ) );

    ASSERT_NE( nullptr, getApi() );
}

TEST_F( TestPbrtApiConstruction, destructorStopsLogging )
{
    EXPECT_CALL( *m_logger, start( _ ) ).Times( AnyNumber() );
    EXPECT_CALL( *m_logger, stop() ).Times( 1 );

    {
        std::shared_ptr<SceneLoader> loader( createSceneLoader( g_programName, m_logger, nullptr ) );
    }
}

TEST_F( TestPbrtApiConstruction, destructorResetsApi )
{
    EXPECT_CALL( *m_logger, start( _ ) ).Times( AnyNumber() );
    EXPECT_CALL( *m_logger, stop() ).Times( AnyNumber() );

    {
        std::shared_ptr<SceneLoader> loader( createSceneLoader( g_programName, m_logger, nullptr ) );
    }

    ASSERT_EQ( nullptr, getApi() );
}

TEST_F( TestPbrtApiEmptyScene, cameraZeroed )
{
    const PerspectiveCameraDefinition camera{ m_scene->camera };

    ASSERT_EQ( 0.0f, camera.fov );
    ASSERT_EQ( 0.0f, camera.focalDistance );
    ASSERT_EQ( 0.0f, camera.lensRadius );
}

TEST_F( TestPbrtApiEmptyScene, emptySceneInvalidBounds )
{
    const Bounds3 bounds{ m_scene->bounds };

    ASSERT_TRUE( bounds.pMin.x > bounds.pMax.x );
    ASSERT_TRUE( bounds.pMin.y > bounds.pMax.y );
    ASSERT_TRUE( bounds.pMin.z > bounds.pMax.z );
}

TEST_F( TestPbrtApiEmptyScene, emptyObjectMap )
{
    ASSERT_TRUE( m_scene->objects.empty() );
}

TEST_F( TestPbrtApiEmptyScene, emptyFreeShapesList )
{
    ASSERT_TRUE( m_scene->freeShapes.empty() );
}

TEST_F( TestPbrtApiEmptyScene, emptyObjectInstancesList )
{
    ASSERT_TRUE( m_scene->objectInstances.empty() );
}

TEST_F( TestPbrtApiEmptyScene, emptyObjectInstanceCountsMap )
{
    ASSERT_TRUE( m_scene->instanceCounts.empty() );
}

TEST_F( TestPbrtApiEmptyScene, emptyObjectShapesMap )
{
    ASSERT_TRUE( m_scene->objectShapes.empty() );
}

TEST_F( TestPbrtApi, perspectiveCameraToWorldTransform )
{
    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        LookAt
            1 2 3 # eye
            4 5 6 # look at
            7 8 9 # up
        Camera "perspective"
        )pbrt" ) };

    const PerspectiveCameraDefinition& camera{ scene->camera };
    ASSERT_EQ( ::pbrt::LookAt( Point3( 1.0f, 2.0f, 3.0f ), Point3( 4.0f, 5.0f, 6.0f ), Vector3( 7.0f, 8.0f, 9.0f ) ),
               camera.cameraToWorld );
}

TEST_F( TestPbrtApi, perspectiveCameraToScreenTransform )
{
    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        Camera "perspective"
            "float fov" 44.0
        )pbrt" ) };

    const PerspectiveCameraDefinition& camera{ scene->camera };
    ASSERT_EQ( ::pbrt::Perspective( 44.0f, 1e-2f, 1000.f ), camera.cameraToScreen );
}

TEST_F( TestPbrtApi, lookAtParams )
{
    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        LookAt
            1 2 3 # eye
            4 5 6 # look at
            7 8 9 # up
        )pbrt" ) };

    const LookAtDefinition& lookAt{ scene->lookAt };
    ASSERT_EQ( Point3( 1.0f, 2.0f, 3.0f ), lookAt.eye );
    ASSERT_EQ( Point3( 4.0f, 5.0f, 6.0f ), lookAt.lookAt );
    ASSERT_EQ( Vector3( 7.0f, 8.0f, 9.0f ), lookAt.up );
}

TEST_F( TestPbrtApi, perspectiveCamera )
{
    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        Camera "perspective"
            "float fov" [ 44 ]
            "float focaldistance" [ 3000 ]
            "float lensradius" [ 0.125 ]
        )pbrt" ) };

    const PerspectiveCameraDefinition& camera{ scene->camera };
    ASSERT_EQ( 44.0f, camera.fov );
    ASSERT_EQ( 3000.0f, camera.focalDistance );
    ASSERT_EQ( 0.125f, camera.lensRadius );
}

TEST_F( TestPbrtApi, perspectiveCameraHalfFOV )
{
    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        Camera "perspective"
            "float halffov" 22
        )pbrt" ) };

    const PerspectiveCameraDefinition& camera{ scene->camera };
    ASSERT_EQ( 44.0f, camera.fov );
}

TEST_F( TestPbrtApi, perspectiveCameraDefaults )
{
    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(Camera "perspective")pbrt" ) };

    const PerspectiveCameraDefinition& camera{ scene->camera };
    // Default values from pbrt v3 scene description documentation
    ASSERT_EQ( 90.0f, camera.fov );
    ASSERT_EQ( 1.0e30f, camera.focalDistance );
    ASSERT_EQ( 0.0f, camera.lensRadius );
}

TEST_F( TestPbrtApi, cameraMatrix )
{
    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        Translate 10  0  0
        Translate  0 20  0
        Translate  0  0  30
        Camera "perspective"
            "float fov" [ 44 ]
            "float focaldistance" [ 3000 ]
            "float lensradius" [ 0.125 ]
        )pbrt" ) };

    EXPECT_EQ( ::pbrt::Translate( ::pbrt::Vector3f( 10.0, 20.0, 30.0 ) ), scene->camera.cameraToWorld );
}

TEST_F( TestPbrtApi, sceneBoundsFromSingleMeshAtDefaultPosition )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Shape "plymesh" "string filename" "mesh_00001.ply"
        WorldEnd)pbrt" ) };

    const Bounds3& bounds{ scene->bounds };
    ASSERT_EQ( -1.0f, bounds.pMin.x );
    ASSERT_EQ( -2.0f, bounds.pMin.y );
    ASSERT_EQ( -3.0f, bounds.pMin.z );
    ASSERT_EQ( 4.0f, bounds.pMax.x );
    ASSERT_EQ( 5.0f, bounds.pMax.y );
    ASSERT_EQ( 6.0f, bounds.pMax.z );
}

TEST_F( TestPbrtApi, sceneBoundsFromSingleMeshAtTransformedPosition )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Translate 1 2 3
        Shape "plymesh" "string filename" "mesh_00001.ply"
        WorldEnd)pbrt" ) };

    const Bounds3& bounds{ scene->bounds };
    ASSERT_EQ( 0.0f, bounds.pMin.x );
    ASSERT_EQ( 0.0f, bounds.pMin.y );
    ASSERT_EQ( 0.0f, bounds.pMin.z );
    ASSERT_EQ( 5.0f, bounds.pMax.x );
    ASSERT_EQ( 7.0f, bounds.pMax.y );
    ASSERT_EQ( 9.0f, bounds.pMax.z );
}

TEST_F( TestPbrtApi, sceneBoundsMultipleMeshes )
{
    configureMeshOneInfo( 2 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Translate 1 2 3
        Shape "plymesh" "string filename" "mesh_00001.ply"
        Translate -1 -2 -3
        Shape "plymesh" "string filename" "mesh_00001.ply"
        WorldEnd)pbrt" ) };

    const Bounds3& bounds{ scene->bounds };
    ASSERT_EQ( -1.0f, bounds.pMin.x );
    ASSERT_EQ( -2.0f, bounds.pMin.y );
    ASSERT_EQ( -3.0f, bounds.pMin.z );
    ASSERT_EQ( 5.0f, bounds.pMax.x );
    ASSERT_EQ( 7.0f, bounds.pMax.y );
    ASSERT_EQ( 9.0f, bounds.pMax.z );
}

TEST_F( TestPbrtApi, freeMeshShape )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Translate 1 2 3
        Shape "plymesh" "string filename" "mesh_00001.ply"
        WorldEnd)pbrt" ) };

    EXPECT_EQ( 1U, scene->freeShapes.size() );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
    EXPECT_TRUE( scene->instanceCounts.empty() );
    const ShapeDefinition shape{ scene->freeShapes[0] };
    EXPECT_EQ( translate( 1.0f, 2.0f, 3.0f ), shape.transform );
}

TEST_F( TestPbrtApi, freeMeshGetsMaterial )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Texture "color1" "spectrum" "imagemap" "string filename" [ "color1.png" ]
        Texture "color2" "spectrum" "imagemap" "string filename" [ "color2.png" ]
        Material "basic"
            "rgb Ka" [ 1 2 3 ]
            "rgb Kd" [ 4 5 6 ]
            "rgb Ks" [ 7 8 9 ]
            "texture Kd" [ "color1" ]
            "texture Ks" [ "color2" ]
        Shape "plymesh" "string filename" "mesh_00001.ply"
        WorldEnd)pbrt" ) };

    const ShapeList& shapes{ scene->freeShapes };
    EXPECT_EQ( 1U, shapes.size() );
    const ShapeDefinition shape{ shapes[0] };
    EXPECT_EQ( Point3( 1.0f, 2.0f, 3.0f ), shape.material.Ka );
    EXPECT_EQ( Point3( 4.0f, 5.0f, 6.0f ), shape.material.Kd );
    EXPECT_EQ( Point3( 7.0f, 8.0f, 9.0f ), shape.material.Ks );
    EXPECT_THAT( shape.material.diffuseMapFileName.c_str(), EndsWith( "color1.png" ) );
    EXPECT_THAT( shape.material.specularMapFileName.c_str(), EndsWith( "color2.png" ) );
}

TEST_F( TestPbrtApi, freeMeshGetsAlphaTexture )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Texture "texture1"
            "float" "imagemap"
            "string filename" [ "alpha1.png" ]
        Shape "plymesh"
            "string filename" "mesh_00001.ply"
            "texture alpha" "texture1"
        WorldEnd)pbrt" ) };

    const ShapeList& shapes{ scene->freeShapes };
    EXPECT_EQ( 1U, shapes.size() );
    const ShapeDefinition shape{ shapes[0] };
    EXPECT_THAT( shape.material.alphaMapFileName, EndsWith( "alpha1.png" ) );
}

TEST_F( TestPbrtApi, freeMeshGetsMaterialCombinedThroughAttributes )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Texture "color1" "spectrum" "imagemap" "string filename" [ "color1.png" ]
        Texture "color2" "spectrum" "imagemap" "string filename" [ "color2.png" ]
        Material "basic"
            "rgb Ka" [ 1 2 3 ]
            "rgb Kd" [ 4 5 6 ]
            "rgb Ks" [ 7 8 9 ]
        AttributeBegin
            Material "textured"
                "texture Kd" [ "color1" ]
                "texture Ks" [ "color2" ]
            Shape "plymesh" "string filename" "mesh_00001.ply"
        AttributeEnd
        WorldEnd)pbrt" ) };

    const ShapeList& shapes{ scene->freeShapes };
    EXPECT_EQ( 1U, shapes.size() );
    const ShapeDefinition shape{ shapes[0] };
    EXPECT_EQ( Point3( 1.0f, 2.0f, 3.0f ), shape.material.Ka );
    EXPECT_EQ( Point3( 4.0f, 5.0f, 6.0f ), shape.material.Kd );
    EXPECT_EQ( Point3( 7.0f, 8.0f, 9.0f ), shape.material.Ks );
    EXPECT_THAT( shape.material.diffuseMapFileName.c_str(), EndsWith( "color1.png" ) );
    EXPECT_THAT( shape.material.specularMapFileName.c_str(), EndsWith( "color2.png" ) );
}

TEST_F( TestPbrtApi, objectGetsShape )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Texture "color1" "spectrum" "imagemap" "string filename" [ "color1.png" ]
        Texture "color2" "spectrum" "imagemap" "string filename" [ "color2.png" ]
        Material "basic"
            "rgb Ka" [ 1 2 3 ]
            "rgb Kd" [ 4 5 6 ]
            "rgb Ks" [ 7 8 9 ]
        AttributeBegin
            Translate 1 2 3
            ObjectBegin "object1"
                Material "textured"
                    "texture Kd" [ "color1" ]
                    "texture Ks" [ "color2" ]
                Shape "plymesh" "string filename" "mesh_00001.ply"
            ObjectEnd
        AttributeEnd
        WorldEnd)pbrt" ) };

    const ObjectMap& objects{ scene->objects };
    EXPECT_EQ( 1U, objects.size() );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_FALSE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
    EXPECT_TRUE( scene->instanceCounts.empty() );
    const std::string objectName{ "object1" };
    const auto        it{ objects.find( objectName ) };
    EXPECT_NE( it, objects.cend() );
    const ObjectDefinition object1{ it->second };
    EXPECT_EQ( objectName, object1.name );
    EXPECT_EQ( translate( 1.0f, 2.0f, 3.0f ), object1.transform );
    const Bounds3 shapeBounds{ translate( 1.0f, 2.0f, 3.0f )( m_meshBounds ) };
    EXPECT_TRUE( shapeBounds == object1.bounds ) << shapeBounds << " != " << object1.bounds;
}

TEST_F( TestPbrtApi, objectInstance )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        Texture "color1" "spectrum" "imagemap" "string filename" [ "color1.png" ]
        Texture "color2" "spectrum" "imagemap" "string filename" [ "color2.png" ]
        Material "basic"
            "rgb Ka" [ 1 2 3 ]
            "rgb Kd" [ 4 5 6 ]
            "rgb Ks" [ 7 8 9 ]
        AttributeBegin
            Translate 1 2 3
            ObjectBegin "object1"
                Material "textured"
                    "texture Kd" [ "color1" ]
                    "texture Ks" [ "color2" ]
                Shape "plymesh" "string filename" "mesh_00001.ply"
            ObjectEnd
        AttributeEnd
        Translate 10 10 10
        ObjectInstance "object1"
        WorldEnd)pbrt" ) };

    const ObjectInstanceList& instances{ scene->objectInstances };
    EXPECT_EQ( 1U, instances.size() );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_FALSE( scene->objectShapes.empty() );
    const ObjectInstanceCountMap& counts{ scene->instanceCounts };
    EXPECT_FALSE( counts.empty() );
    EXPECT_EQ( 1U, counts.find( "object1" )->second );
    const ObjectInstanceDefinition instance{ instances[0] };
    EXPECT_EQ( "object1", instance.name );
    EXPECT_EQ( translate( 10.0f, 10.0f, 10.0f ), instance.transform );
}

TEST_F( TestPbrtApi, objectInstanceTransformedShape )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        ObjectBegin "object1"
            AttributeBegin
                Translate 1 2 3
                Shape "plymesh" "string filename" "mesh_00001.ply"
            AttributeEnd
        ObjectEnd
        ObjectInstance "object1"
        WorldEnd)pbrt" ) };

    const ObjectInstanceList& instances{ scene->objectInstances };
    EXPECT_EQ( 1U, instances.size() );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_FALSE( scene->objectShapes.empty() );
    EXPECT_FALSE( scene->objectShapes["object1"].empty() );
    const auto& shape{ scene->objectShapes["object1"][0] };
    EXPECT_EQ( translate( 1.0f, 2.0f, 3.0f ), shape.transform );
    EXPECT_EQ( m_meshBounds, shape.bounds );
    const ObjectInstanceCountMap& counts{ scene->instanceCounts };
    EXPECT_FALSE( counts.empty() );
    EXPECT_EQ( 1U, counts.find( "object1" )->second );
    const ObjectDefinition& object{ scene->objects["object1"] };
    EXPECT_EQ( pbrt::Transform(), object.transform );
    EXPECT_EQ( shape.transform( m_meshBounds ), object.bounds );
    const ObjectInstanceDefinition instance{ instances[0] };
    EXPECT_EQ( "object1", instance.name );
    EXPECT_EQ( pbrt::Transform(), instance.transform );
    EXPECT_EQ( object.transform( object.bounds ), instance.bounds );
    EXPECT_EQ( instance.transform( instance.bounds ), scene->bounds );
    const Bounds3 expectedInstanceBounds{ shape.transform( m_meshBounds ) };
    EXPECT_EQ( expectedInstanceBounds, instance.bounds ) << expectedInstanceBounds << " != " << instance.bounds;
    EXPECT_EQ( expectedInstanceBounds, scene->bounds ) << expectedInstanceBounds << " != " << instance.bounds;
}

TEST_F( TestPbrtApi, objectInstanceTransformedObject )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene{ m_api->parseString( R"pbrt(
        WorldBegin
        AttributeBegin
            ObjectBegin "object1"
                Shape "plymesh" "string filename" "mesh_00001.ply"
                Translate 1 2 3
            ObjectEnd
        AttributeEnd
        ObjectInstance "object1"
        WorldEnd)pbrt" ) };

    const ObjectInstanceList& instances{ scene->objectInstances };
    EXPECT_EQ( 1U, instances.size() );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_FALSE( scene->objectShapes.empty() );
    EXPECT_FALSE( scene->objectShapes["object1"].empty() );
    const auto& shape{ scene->objectShapes["object1"][0] };
    EXPECT_EQ( pbrt::Transform(), shape.transform );
    EXPECT_EQ( m_meshBounds, shape.bounds );
    const ObjectInstanceCountMap& counts{ scene->instanceCounts };
    EXPECT_FALSE( counts.empty() );
    EXPECT_EQ( 1U, counts.find( "object1" )->second );
    const ObjectDefinition& object{ scene->objects["object1"] };
    EXPECT_EQ( translate( 1.0f, 2.0f, 3.0f ), object.transform );
    EXPECT_EQ( shape.transform( m_meshBounds ), object.bounds );
    const ObjectInstanceDefinition instance{ instances[0] };
    EXPECT_EQ( "object1", instance.name );
    EXPECT_EQ( pbrt::Transform(), instance.transform );
    const Bounds3 expectedInstanceBounds{ object.transform( object.bounds ) };
    EXPECT_EQ( expectedInstanceBounds, instance.bounds );
    EXPECT_EQ( instance.transform( instance.bounds ), scene->bounds );
}

TEST_F( TestPbrtApi, objectInstanceTransformedInstance )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
        ObjectBegin "object1"
            Shape "plymesh" "string filename" "mesh_00001.ply"
        ObjectEnd
        AttributeBegin
            Translate 1 2 3
            ObjectInstance "object1"
        AttributeEnd
        WorldEnd)pbrt" );

    const ObjectInstanceList& instances{ scene->objectInstances };
    EXPECT_EQ( 1U, instances.size() );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_FALSE( scene->objectShapes.empty() );
    EXPECT_FALSE( scene->objectShapes["object1"].empty() );
    const auto& shape{ scene->objectShapes["object1"][0] };
    EXPECT_EQ( pbrt::Transform(), shape.transform );
    EXPECT_EQ( m_meshBounds, shape.bounds );
    const ObjectInstanceCountMap& counts{ scene->instanceCounts };
    EXPECT_FALSE( counts.empty() );
    EXPECT_EQ( 1U, counts.find( "object1" )->second );
    const ObjectDefinition& object{ scene->objects["object1"] };
    EXPECT_EQ( pbrt::Transform(), object.transform );
    EXPECT_EQ( shape.transform( m_meshBounds ), object.bounds );
    const ObjectInstanceDefinition instance{ instances[0] };
    EXPECT_EQ( "object1", instance.name );
    EXPECT_EQ( translate( 1.0f, 2.0f, 3.0f ), instance.transform );
    const Bounds3 expectedInstanceBounds{ object.transform( object.bounds ) };
    EXPECT_EQ( expectedInstanceBounds, instance.bounds );
    EXPECT_EQ( instance.transform( instance.bounds ), scene->bounds );
}

TEST_F( TestPbrtApi, plyMeshShape )
{
    configureMeshOneInfo( 1 );

    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
        Shape "plymesh" "string filename" "mesh_00001.ply"
        WorldEnd)pbrt" );

    ASSERT_FALSE( scene->freeShapes.empty() );
    const ShapeDefinition& shape{ scene->freeShapes[0] };
    ASSERT_EQ( SHAPE_TYPE_PLY_MESH, shape.type );
    ASSERT_THAT( shape.plyMesh.fileName, EndsWith( "mesh_00001.ply" ) );
}

TEST_F( TestPbrtApi, triangleMeshShape )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Shape "trianglemesh"
                "integer indices" [0 2 1 0 3 2]
                "point P" [
                    550 0   0
                    0   0   0
                    0   0   560
                    550 0   560
                ]
                "point N" [
                    0.1     0.2     0.3
                    0.4     0.5     0.6
                    0.7     0.8     0.9
                    1.0     1.1     1.2
                ]
                "float uv" [
                    0.1     0.2
                    0.3     0.4
                    0.5     0.6
                    0.7     0.8
                ]
        WorldEnd)pbrt" );

    EXPECT_FALSE( scene->freeShapes.empty() );
    const ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( SHAPE_TYPE_TRIANGLE_MESH, shape.type );
    const TriangleMeshData& triMesh{ shape.triangleMesh };
    EXPECT_EQ( 6U, triMesh.indices.size() );
    EXPECT_EQ( std::vector<int>( { 0, 2, 1, 0, 3, 2 } ), triMesh.indices );
    const auto p3{ []( float x, float y, float z ) { return Point3( x, y, z ); } };
    EXPECT_EQ( std::vector<Point3>( { p3( 550, 0, 0 ), p3( 0, 0, 0 ), p3( 0, 0, 560 ), p3( 550, 0, 560 ) } ), triMesh.points );
    EXPECT_EQ( std::vector<Point3>( { p3( 0.1f, 0.2f, 0.3f ), p3( 0.4f, 0.5f, 0.6f ), p3( 0.7f, 0.8f, 0.9f ), p3( 1.0f, 1.1f, 1.2f ) } ),
               triMesh.normals );
    const auto p2{ []( float x, float y ) { return Point2( x, y ); } };
    EXPECT_EQ( std::vector<Point2>( { p2( 0.1f, 0.2f ), p2( 0.3f, 0.4f ), p2( 0.5f, 0.6f ), p2( 0.7f, 0.8f ) } ),
               triMesh.uvs );
    const Bounds3 expectedBounds{ Point3( 0, 0, 0 ), Point3( 550, 0, 560 ) };
    EXPECT_EQ( expectedBounds, shape.bounds );
}

TEST_F( TestPbrtApi, sphereShape )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Shape "sphere"
                "float radius" 2.2
                "float zmin" 3.3
                "float zmax" 4.4
                "float phimax" 5.5
        WorldEnd)pbrt" );

    EXPECT_FALSE( scene->freeShapes.empty() );
    const ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( SHAPE_TYPE_SPHERE, shape.type );
    const SphereData& sphere{ shape.sphere };
    EXPECT_EQ( 2.2f, sphere.radius );
    EXPECT_EQ( 3.3f, sphere.zMin );
    EXPECT_EQ( 4.4f, sphere.zMax );
    EXPECT_EQ( 5.5f, sphere.phiMax );
    const Bounds3 expectedBounds{ Point3( -2.2f, -2.2f, -2.2f ), Point3( 2.2f, 2.2f, 2.2f ) };
    EXPECT_EQ( expectedBounds, shape.bounds );
}

TEST_F( TestPbrtApi, sphereShapeDefaults )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Shape "sphere"
        WorldEnd)pbrt" );

    EXPECT_FALSE( scene->freeShapes.empty() );
    const ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( SHAPE_TYPE_SPHERE, shape.type );
    const SphereData& sphere{ shape.sphere };
    EXPECT_EQ( 1.0f, sphere.radius );
    EXPECT_EQ( -1.0f, sphere.zMin );
    EXPECT_EQ( 1.0f, sphere.zMax );
    EXPECT_EQ( 360.0f, sphere.phiMax );
    const Bounds3 expectedBounds{ Point3( -1.0f, -1.0f, -1.0f ), Point3( 1.0f, 1.0f, 1.0f ) };
    EXPECT_EQ( expectedBounds, shape.bounds );
}

TEST_F( TestPbrtApi, triangleMeshMaterial )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Shape "trianglemesh"
                "integer indices" [0 2 1 0 3 2]
                "point P" [
                    550 0   0
                    0   0   0
                    0   0   560
                    550 0   560
                ]
                "rgb Ka" [ 1 2 3 ]
        WorldEnd)pbrt" );

    EXPECT_FALSE( scene->freeShapes.empty() );
    const ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( Point3( 1.0f, 2.0f, 3.0f ), shape.material.Ka );
}

TEST_F( TestPbrtApi, stateResetForEachParse )
{
    const SceneDescriptionPtr scene1 = m_api->parseString( R"pbrt(
        WorldBegin
            Shape "trianglemesh"
                "integer indices" [0 2 1]
                "point P" [
                    550 0   0
                    0   0   0
                ]
                "rgb Ka" [ 1 2 3 ]
        WorldEnd)pbrt" );
    const SceneDescriptionPtr scene2 = m_api->parseString( R"pbrt(
        WorldBegin
            Translate 100 100 100
            Shape "trianglemesh"
                "integer indices" [0 2 1]
                "point P" [
                    550 0   0
                    0   0   0
                ]
                "rgb Ka" [ 1 2 3 ]
        WorldEnd)pbrt" );

    EXPECT_EQ( 1U, scene1->freeShapes.size() );
    EXPECT_EQ( 1U, scene2->freeShapes.size() );
    EXPECT_EQ( Bounds3( Point3( 0, 0, 0 ), Point3( 550, 0, 0 ) ), scene1->bounds );
    EXPECT_EQ( Bounds3( Point3( 100, 100, 100 ), Point3( 650, 100, 100 ) ), scene2->bounds );
}

TEST_F( TestPbrtApi, emptyObjectInstancesAreDropped )
{
    EXPECT_CALL( *m_mockLogger, warning( ContainsRegex( "Skipping instances of empty object" ), _, _ ) ).Times( 1 );
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            # add a non-empty shape
            Shape "trianglemesh"
                "integer indices" [0 2 1]
                "point P" [
                    550 0   0
                    0   0   0
                ]
                "rgb Ka" [ 1 2 3 ]

            # define an empty object
            AttributeBegin
                ObjectBegin "Brennnessel Instance.4"
                ObjectEnd
            AttributeEnd

            # instantiate an empty object
            AttributeBegin
                ConcatTransform [ -1.3296067715 0.0453767702 0.6984055042 0.0000000000 0.0190331340 1.5011873245 -0.0613002740 0.0000000000 -0.6996192336 -0.0453974940 -1.3289678097 0.0000000000 2801.2675781250 204.4016723633 4959.5805664062 1.0000000000  ]
                ObjectInstance "Brennnessel Instance.4"
            AttributeEnd

        WorldEnd)pbrt" );

    EXPECT_EQ( 1U, scene->freeShapes.size() );
    EXPECT_EQ( scene->freeShapes[0].bounds, scene->bounds );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
}

TEST_F( TestPbrtApi, emptyShapesAreDropped )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            # add a non-empty shape
            Shape "trianglemesh"
                "integer indices" [0 2 1]
                "point P" [
                    550 0   0
                    0   0   0
                ]
                "rgb Ka" [ 1 2 3 ]

            # define an empty shape
            AttributeBegin
                Translate 200 0 0
                Shape "trianglemesh"
                    "integer indices" []
                    "point P" []
                    "rgb Ka" [ 1 2 3 ]
            AttributeEnd

        WorldEnd)pbrt" );

    ASSERT_EQ( 1U, scene->freeShapes.size() );
    EXPECT_EQ( scene->freeShapes[0].bounds, scene->bounds );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
}

TEST_F( TestPbrtApi, distantLightDefaultValues )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            LightSource "distant"
        WorldEnd)pbrt" );

    ASSERT_EQ( 1U, scene->distantLights.size() );
    const DistantLightDefinition& light{ scene->distantLights[0] };
    EXPECT_EQ( ::pbrt::Point3f( 1.0f, 1.0f, 1.0f ), light.scale );
    EXPECT_EQ( ::pbrt::Point3f( 1.0f, 1.0f, 1.0f ), light.color );
    EXPECT_EQ( ::pbrt::Vector3f( 0.0f, 0.0f, -1.0f ), light.direction );
    EXPECT_EQ( ::pbrt::Transform(), light.lightToWorld );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
}

TEST_F( TestPbrtApi, distantLightNonDefaultValues )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Translate 100 200 300
            LightSource "distant"
                "rgb scale"     [   10      20      30  ]
                "rgb L"         [   0.5     0.6     0.7 ]
                "point from"    [   1       1       1   ]
                "point to"      [   0       0       0   ]
        WorldEnd)pbrt" );

    ASSERT_EQ( 1U, scene->distantLights.size() );
    const DistantLightDefinition& light{ scene->distantLights[0] };
    EXPECT_EQ( ::pbrt::Point3f( 10.0f, 20.0f, 30.0f ), light.scale );
    EXPECT_EQ( ::pbrt::Point3f( 0.5f, 0.6f, 0.7f ), light.color );
    EXPECT_EQ( ::pbrt::Vector3f( 1.0f, 1.0f, 1.0f ), light.direction );
    EXPECT_EQ( ::pbrt::Translate( ::pbrt::Vector3f( 100.0f, 200.0f, 300.0f ) ), light.lightToWorld );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
}

TEST_F( TestPbrtApi, infiniteLightDefaultValues )
{
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            LightSource "infinite"
        WorldEnd)pbrt" );

    ASSERT_EQ( 1U, scene->infiniteLights.size() );
    const InfiniteLightDefinition& light{ scene->infiniteLights[0] };
    EXPECT_EQ( ::pbrt::Point3f( 1.0f, 1.0f, 1.0f ), light.color );
    EXPECT_EQ( ::pbrt::Point3f( 1.0f, 1.0f, 1.0f ), light.scale );
    EXPECT_EQ( 1, light.shadowSamples );
    EXPECT_EQ( "", light.environmentMapName );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
}

TEST_F( TestPbrtApi, infiniteLightNonDefaultValues )
{
    expectWarnings( 1 );
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Translate 100 200 300
            LightSource "infinite"
                "rgb scale"     [   10      20      30  ]
                "rgb L"         [   0.5     0.6     0.7 ]
                "integer samples" 15
                "string mapname" "skybox.png"
        WorldEnd)pbrt" );

    ASSERT_EQ( 1U, scene->infiniteLights.size() );
    const InfiniteLightDefinition& light{ scene->infiniteLights[0] };
    EXPECT_EQ( ::pbrt::Point3f( 10.0f, 20.0f, 30.0f ), light.scale );
    EXPECT_EQ( ::pbrt::Point3f( 0.5f, 0.6f, 0.7f ), light.color );
    EXPECT_EQ( 15, light.shadowSamples );
    EXPECT_THAT( light.environmentMapName, EndsWith( "skybox.png" ) );
    EXPECT_EQ( ::pbrt::Translate( ::pbrt::Vector3f( 100.0f, 200.0f, 300.0f ) ), light.lightToWorld );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
}

TEST_F( TestPbrtApi, infiniteLightNSamples )
{
    expectWarnings( 1 );
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Translate 100 200 300
            LightSource "infinite"
                "integer nsamples" 15
                "string mapname" "skybox.png"
        WorldEnd)pbrt" );

    ASSERT_EQ( 1U, scene->infiniteLights.size() );
    const InfiniteLightDefinition& light{ scene->infiniteLights[0] };
    EXPECT_EQ( 15, light.shadowSamples );
    EXPECT_TRUE( scene->freeShapes.empty() );
    EXPECT_TRUE( scene->objects.empty() );
    EXPECT_TRUE( scene->objectShapes.empty() );
    EXPECT_TRUE( scene->objectInstances.empty() );
}

TEST_F( TestPbrtApi, mixMaterialUberTranslucentChoosesUber )
{
    configureMeshOneInfo( 1 );
    SceneDescriptionPtr scene = m_api->parseString( R"pbrt(
        WorldBegin
            Texture "Aesculus_hippocastanum_lf_01_su_co_fr_color" "spectrum" "imagemap"
                "string filename" [ "textures/Aesculus_hippocastanum_lf_01_su_co_fr-diffuse.png" ]
            Texture "Aesculus_hippocastanum_lf_01_su_co_fr_alpha" "float" "imagemap"
                "string filename" [ "textures/Aesculus_hippocastanum_lf_01_su_co_fr-alpha.png" ]
            Texture "Aesculus_hippocastanum_lf_01_su_co_fr_bump" "float" "imagemap"
                "string filename" [ "textures/Aesculus_hippocastanum_lf_01_su_co_fr-bump.png" ]
            MakeNamedMaterial "xref_aesculus_02medium.c4d/aesculus_hippocastanum_lf_01_su_fr_front"
                "float index" [ 1.3329999447 ]
                "float roughness" [ 0.5000000000 ]
                "string type" [ "uber" ]
                "texture Kd" [ "Aesculus_hippocastanum_lf_01_su_co_fr_color" ]
                "texture bumpmap" [ "Aesculus_hippocastanum_lf_01_su_co_fr_bump" ]
                "rgb Ks" [ 0.0196066480 0.0196066480 0.0196066480 ]
                "rgb Kr" [ 0.5225215554 0.5225215554 0.5225215554 ]
            MakeNamedMaterial "xref_aesculus_02medium.c4d/aesculus_hippocastanum_lf_01_su_fr_back"
                "string type" [ "translucent" ]
                "texture Kd" [ "Aesculus_hippocastanum_lf_01_su_co_fr_color" ]
                "rgb reflect" [ 0 0 0 ]
                "rgb transmit" [ 1 1 1 ]
            Material "mix"
                "string namedmaterial1" [ "xref_aesculus_02medium.c4d/aesculus_hippocastanum_lf_01_su_fr_front" ]
                "string namedmaterial2" [ "xref_aesculus_02medium.c4d/aesculus_hippocastanum_lf_01_su_fr_back" ]
                "rgb amount" [ 0.4000000060 0.4000000060 0.4000000060 ]
            Shape "plymesh" "string filename" "geometry/mesh_00001.ply"
                "texture alpha" "Aesculus_hippocastanum_lf_01_su_co_fr_alpha"
                "texture shadowalpha" "Aesculus_hippocastanum_lf_01_su_co_fr_alpha"
        WorldEnd)pbrt" );

    ASSERT_EQ( 1U, scene->freeShapes.size() );
    const ShapeDefinition& shape{ scene->freeShapes[0] };
    EXPECT_EQ( ::pbrt::Point3f( 0.0f, 0.0f, 0.0f ), shape.material.Ka );
    EXPECT_EQ( ::pbrt::Point3f( 1.0f, 1.0f, 1.0f ), shape.material.Kd );
    EXPECT_NE( ::pbrt::Point3f( 0.0f, 0.0f, 0.0f ), shape.material.Ks );
    EXPECT_THAT( shape.material.alphaMapFileName, EndsWith( "Aesculus_hippocastanum_lf_01_su_co_fr-alpha.png" ) );
    EXPECT_THAT( shape.material.diffuseMapFileName, EndsWith( "Aesculus_hippocastanum_lf_01_su_co_fr-diffuse.png" ) );
}
