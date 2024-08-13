// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// gtest has to come before pbrt stuff
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Matchers.h"
#include "MockGeometryLoader.h"
#include "MockMeshLoader.h"
#include "ParamsPrinters.h"

#include <GeometryCache.h>
#include <SceneProxy.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockOptix.h>
#include <OptiXToolkit/DemandGeometry/Mocks/OptixCompare.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/BitCast.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <optix.h>

#include <type_traits>

using namespace ::testing;
using namespace demandPbrtScene;
using namespace demandPbrtScene::testing;
using namespace otk::testing;

using P2 = pbrt::Point2f;
using P3 = pbrt::Point3f;
using B3 = pbrt::Bounds3f;

using Stats = GeometryCacheStatistics;

inline void PrintTo( const OptixAabb& value, std::ostream* str )
{
    *str << value;
}

namespace demandPbrtScene {

inline void PrintTo( const TriangleNormals& value, std::ostream* str )
{
    *str << value;
}

inline void PrintTo( const TriangleUVs& value, std::ostream* str )
{
    *str << value;
}

inline bool operator==( const GeometryCacheEntry& lhs, const GeometryCacheEntry& rhs )
{
    return lhs.accelBuffer == rhs.accelBuffer && lhs.traversable == rhs.traversable && lhs.devNormals == rhs.devNormals;
}

}  // namespace demandPbrtScene

MATCHER_P( hasDeviceTriangleNormals, triangleMesh, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "pointer to triangle normals is nullptr";
        return false;
    }
    std::vector<TriangleNormals> actual;
    actual.resize( triangleMesh->indices.size() / 3 );
    OTK_ERROR_CHECK( cudaMemcpy( actual.data(), arg, sizeof( TriangleNormals ) * actual.size(), cudaMemcpyDeviceToHost ) );
    bool       result{ true };
    const auto toFloat3{ []( const pbrt::Point3f& val ) { return make_float3( val.x, val.y, val.z ); } };
    for( size_t tri = 0; tri < actual.size(); ++tri )
    {
        for( int vert = 0; vert < 3; ++vert )
        {
            const float3 expected{ toFloat3( triangleMesh->normals[triangleMesh->indices[tri * 3 + vert]] ) };
            if( actual[tri].N[vert] != expected )
            {
                if( !result )
                {
                    *result_listener << "; ";
                }
                *result_listener << "index " << tri << " has normal " << actual[tri].N[vert] << ", expected " << expected;
                result = false;
            }
        }
    }
    if( result )
    {
        *result_listener << "has expected device triangle normals";
    }
    return result;
}

MATCHER_P( hasDevicePlyNormals, buffers, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "pointer to ply normals is nullptr";
        return false;
    }
    std::vector<TriangleNormals> actual;
    actual.resize( buffers->indices.size() / 3 );
    OTK_ERROR_CHECK( cudaMemcpy( actual.data(), arg, sizeof( TriangleNormals ) * actual.size(), cudaMemcpyDeviceToHost ) );
    bool       result{ true };
    const auto toFloat3{ [&]( size_t tri, int vert ) {
        return make_float3( buffers->normalCoords[buffers->indices[tri * 3 + vert] * 3 + 0],
                            buffers->normalCoords[buffers->indices[tri * 3 + vert] * 3 + 1],
                            buffers->normalCoords[buffers->indices[tri * 3 + vert] * 3 + 2] );
    } };
    for( size_t tri = 0; tri < actual.size(); ++tri )
    {
        for( int vert = 0; vert < 3; ++vert )
        {
            const float3 expected{ toFloat3( tri, vert ) };
            if( actual[tri].N[vert] != expected )
            {
                if( !result )
                {
                    *result_listener << "; ";
                }
                *result_listener << "triangle " << tri << ", vertex " << vert << " has normal " << actual[tri].N[vert]
                                 << ", expected " << expected;
                result = false;
            }
        }
    }
    if( result )
    {
        *result_listener << "has expected device ply normals";
    }
    return result;
}

MATCHER_P( hasDevicePlyUVs, buffers, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "pointer to ply uvs is nullptr";
        return false;
    }
    std::vector<TriangleUVs> actual;
    actual.resize( buffers->indices.size() / 3 );
    OTK_ERROR_CHECK( cudaMemcpy( actual.data(), arg, sizeof( TriangleUVs ) * actual.size(), cudaMemcpyDeviceToHost ) );
    bool       result{ true };
    const auto toFloat2{ [&]( size_t tri, int vert ) {
        return make_float2( buffers->uvCoords[buffers->indices[tri * 3 + vert] * 2 + 0],
                            buffers->uvCoords[buffers->indices[tri * 3 + vert] * 2 + 1] );
    } };
    for( size_t tri = 0; tri < actual.size(); ++tri )
    {
        for( int vert = 0; vert < 3; ++vert )
        {
            const float2 expected{ toFloat2( tri, vert ) };
            if( actual[tri].UV[vert] != expected )
            {
                if( !result )
                {
                    *result_listener << "; ";
                }
                *result_listener << "triangle " << tri << ", vertex " << vert << " has uv " << actual[tri].UV[vert]
                                 << ", expected " << expected;
                result = false;
            }
        }
    }
    if( result )
    {
        *result_listener << "has expected device ply UVs";
    }
    return result;
}

ListenerPredicate<OptixBuildInputSphereArray> hasDeviceSphereVertices( const std::vector<float3>& expectedVertices )
{
    return [&]( MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        std::vector<float3> actualVertices;
        actualVertices.resize( expectedVertices.size() );
        OTK_ERROR_CHECK( cudaMemcpy( actualVertices.data(), otk::bit_cast<void*>( spheres.vertexBuffers[0] ),
                                     sizeof( float3 ) * expectedVertices.size(), cudaMemcpyDeviceToHost ) );
        bool       result{ true };
        const auto separator{ [&] {
            if( !result )
                *listener << "; ";
            return listener;
        } };
        for( int i = 0; i < static_cast<int>( expectedVertices.size() ); ++i )
        {
            if( actualVertices[i] != expectedVertices[i] )
            {
                *separator() << "has vertex[" << i << "] " << actualVertices[i] << ", expected " << expectedVertices[i];
                result = false;
            }
        }

        if( result )
        {
            *separator() << "has expected sphere vertices";
        }
        return result;
    };
}

MATCHER_P( hasDeviceTriangleUVs, triangleMesh, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "pointer to triangle UVs is nullptr";
        return false;
    }
    std::vector<TriangleUVs> actual;
    actual.resize( triangleMesh->indices.size() / 3 );
    OTK_ERROR_CHECK( cudaMemcpy( actual.data(), arg, sizeof( TriangleUVs ) * actual.size(), cudaMemcpyDeviceToHost ) );
    bool       result{ true };
    const auto toFloat2{ []( const pbrt::Point2f& val ) { return make_float2( val.x, val.y ); } };
    for( size_t tri = 0; tri < actual.size(); ++tri )
    {
        for( int vert = 0; vert < 3; ++vert )
        {
            const float2 expected{ toFloat2( triangleMesh->uvs[triangleMesh->indices[tri * 3 + vert]] ) };
            if( actual[tri].UV[vert] != expected )
            {
                if( !result )
                {
                    *result_listener << "; ";
                }
                *result_listener << "index " << tri << " has UV " << actual[tri].UV[vert] << ", expected " << expected;
                result = false;
            }
        }
    }
    if( result )
    {
        *result_listener << "has expected device triangle UVs";
    }
    return result;
}

namespace {

constexpr const char*        ARBITRARY_PLY_FILENAME{ "cube-mesh.ply" };
constexpr unsigned long long ARBITRARY_PLY_FILE_SIZE{ 12031964U };

class MockFileSystemInfo : public StrictMock<FileSystemInfo>
{
  public:
    ~MockFileSystemInfo() override = default;

    MOCK_METHOD( unsigned long long, getSize, (const std::string&), ( const, override ) );
};

using MockFileSystemInfoPtr = std::shared_ptr<MockFileSystemInfo>;

class TestGeometryCache : public Test
{
  protected:
    void SetUp() override
    {
        initMockOptix( m_optix );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
        OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0 ) );
        m_accelSizes.tempSizeInBytes   = 1234U;
        m_accelSizes.outputSizeInBytes = 5678U;
        m_expectedFlags.push_back( OPTIX_GEOMETRY_FLAG_NONE );
    }

    void TearDown() override
    {
        if( m_geom.devNormals )
            OTK_ERROR_CHECK( cudaFree( m_geom.devNormals ) );
        if( m_geom.devUVs )
            OTK_ERROR_CHECK( cudaFree( m_geom.devUVs ) );
        if( m_geom.accelBuffer )
            OTK_ERROR_CHECK( cuMemFree( m_geom.accelBuffer ) );
        OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
    }

    template <typename OptionMatcher, typename BuildInputMatcher>
    Expectation configureAccelComputeMemoryUsage( OptionMatcher& expectedOptions, BuildInputMatcher& expectedInput )
    {
        const uint_t numBuildInputs{ 1 };
        return EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeContext, expectedOptions, expectedInput,
                                                              numBuildInputs, NotNull() ) )
            .WillOnce( DoAll( SetArgPointee<4>( m_accelSizes ), Return( OPTIX_SUCCESS ) ) );
    }

    template <typename OptionMatcher, typename BuildInputMatcher>
    Expectation configureAccelBuild( OptionMatcher& expectedOptions, BuildInputMatcher& expectedInput, OptixTraversableHandle result )
    {
        const uint_t numBuildInputs{ 1 };
        return EXPECT_CALL( m_optix, accelBuild( m_fakeContext, m_stream, expectedOptions, expectedInput, numBuildInputs,
                                                 Ne( CUdeviceptr{} ), m_accelSizes.tempSizeInBytes, Ne( CUdeviceptr{} ),
                                                 m_accelSizes.outputSizeInBytes, NotNull(), nullptr, 0 ) )
            .WillOnce( DoAll( SetArgPointee<9>( result ), Return( OPTIX_SUCCESS ) ) );
    }
    template <typename OptionMatcher, typename BuildInputMatcher>
    Expectation configureAccelBuild( OptionMatcher& expectedOptions, BuildInputMatcher& expectedInput )
    {
        return configureAccelBuild( expectedOptions, expectedInput, m_fakeGeomAS );
    }

    void expectPlyFileSizeReturned()
    {
        EXPECT_CALL( *m_fileSystemInfo, getSize( StrEq( ARBITRARY_PLY_FILENAME ) ) ).WillRepeatedly( Return( ARBITRARY_PLY_FILE_SIZE ) );
    }

    CUstream               m_stream{};
    MockFileSystemInfoPtr  m_fileSystemInfo{ std::make_shared<MockFileSystemInfo>() };
    GeometryCachePtr       m_geometryCache{ createGeometryCache( m_fileSystemInfo ) };
    StrictMock<MockOptix>  m_optix;
    OptixDeviceContext     m_fakeContext{ otk::bit_cast<OptixDeviceContext>( 0xf00df00dULL ) };
    GeometryCacheEntry     m_geom{};
    OptixAccelBufferSizes  m_accelSizes{};
    OptixTraversableHandle m_fakeGeomAS{ 0xfeedf00dU };
    std::vector<uint_t>    m_expectedFlags;
};

}  // namespace

static void singleTrianglePlyData( otk::pbrt::MeshData& buffers, otk::pbrt::MeshInfo& info )
{
    static std::vector<float> coords{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
    static std::vector<int>   indices{ 0, 1, 2 };
    buffers.vertexCoords = coords;
    buffers.indices      = indices;
    info.numVertices     = coords.size() / 3;
    info.numTriangles    = indices.size() / 3;
}

static otk::pbrt::ShapeDefinition makePlyShape( MockMeshLoaderPtr loader, otk::pbrt::MeshData& buffers, otk::pbrt::MeshInfo& info )
{
    otk::pbrt::ShapeDefinition shape{};
    shape.type    = "plymesh";
    shape.plyMesh = otk::pbrt::PlyMeshData{ ARBITRARY_PLY_FILENAME, loader };
    EXPECT_CALL( *loader, getMeshInfo() ).WillOnce( Return( info ) );
    EXPECT_CALL( *loader, load( _ ) ).WillOnce( SetArgReferee<0>( buffers ) );
    return shape;
}

static otk::pbrt::ShapeDefinition singleTrianglePlyMesh( MockMeshLoaderPtr loader, otk::pbrt::MeshData& buffers, otk::pbrt::MeshInfo& info )
{
    singleTrianglePlyData( buffers, info );
    return makePlyShape( loader, buffers, info );
}

static otk::pbrt::ShapeDefinition singleTrianglePlyMeshWithNormals( MockMeshLoaderPtr    loader,
                                                                    otk::pbrt::MeshData& buffers,
                                                                    otk::pbrt::MeshInfo& info )
{
    singleTrianglePlyData( buffers, info );
    info.numNormals = info.numVertices;
    static std::vector<float> normalCoords{ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    buffers.normalCoords = normalCoords;
    return makePlyShape( loader, buffers, info );
}

static otk::pbrt::ShapeDefinition singleTrianglePlyMeshWithUVs( MockMeshLoaderPtr    loader,
                                                                otk::pbrt::MeshData& buffers,
                                                                otk::pbrt::MeshInfo& info )
{
    singleTrianglePlyData( buffers, info );
    info.numTextureCoordinates = info.numVertices;
    static std::vector<float> uvCoords{ 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
    buffers.uvCoords = uvCoords;
    return makePlyShape( loader, buffers, info );
}

TEST_F( TestGeometryCache, constructTriangleASForPlyMesh )
{
    expectPlyFileSizeReturned();
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::MeshData        buffers;
    otk::pbrt::MeshInfo        info{};
    otk::pbrt::ShapeDefinition shape{ singleTrianglePlyMesh( meshLoader, buffers, info ) };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto                 expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( buffers.vertexCoords ),
                                                                            hasDeviceIndices( buffers.indices ), hasSbtFlags( m_expectedFlags ),
                                                                            hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                            hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    EXPECT_NE( CUdeviceptr{}, m_geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, m_geom.traversable );
    EXPECT_EQ( nullptr, m_geom.devNormals );
    EXPECT_EQ( nullptr, m_geom.devUVs );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
    EXPECT_EQ( ARBITRARY_PLY_FILE_SIZE, stats.totalBytesRead );
}

TEST_F( TestGeometryCache, constructTriangleASForPlyMeshWithNormals )
{
    expectPlyFileSizeReturned();
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::MeshData        buffers;
    otk::pbrt::MeshInfo        info{};
    otk::pbrt::ShapeDefinition shape{ singleTrianglePlyMeshWithNormals( meshLoader, buffers, info ) };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto                 expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( buffers.vertexCoords ),
                                                                            hasDeviceIndices( buffers.indices ), hasSbtFlags( m_expectedFlags ),
                                                                            hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                            hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    EXPECT_NE( CUdeviceptr{}, m_geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, m_geom.traversable );
    EXPECT_NE( nullptr, m_geom.devNormals );
    EXPECT_THAT( m_geom.devNormals, hasDevicePlyNormals( &buffers ) );
    EXPECT_EQ( nullptr, m_geom.devUVs );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 3, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
}

TEST_F( TestGeometryCache, constructTriangleASForPlyMeshWithUVs )
{
    expectPlyFileSizeReturned();
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::MeshData        buffers;
    otk::pbrt::MeshInfo        info{};
    otk::pbrt::ShapeDefinition shape{ singleTrianglePlyMeshWithUVs( meshLoader, buffers, info ) };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto                 expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( buffers.vertexCoords ),
                                                                            hasDeviceIndices( buffers.indices ), hasSbtFlags( m_expectedFlags ),
                                                                            hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                            hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    EXPECT_NE( CUdeviceptr{}, m_geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, m_geom.traversable );
    EXPECT_EQ( nullptr, m_geom.devNormals );
    EXPECT_NE( nullptr, m_geom.devUVs );
    EXPECT_THAT( m_geom.devUVs, hasDevicePlyUVs( &buffers ) );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 3, stats.numUVs );
}

static otk::pbrt::ShapeDefinition singleTriangleTriangleMesh( otk::pbrt::MeshData& buffers )
{
    static std::vector<float>         coords{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
    static std::vector<int>           indices{ 0, 1, 2 };
    static std::vector<pbrt::Point3f> points{ P3{ coords[0 + 0], coords[0 + 1], coords[0 + 2] },
                                              P3{ coords[3 + 0], coords[3 + 1], coords[3 + 2] },
                                              P3{ coords[6 + 0], coords[6 + 1], coords[6 + 2] } };
    buffers.vertexCoords = coords;
    buffers.indices      = indices;
    otk::pbrt::ShapeDefinition shape{};
    shape.type         = "trianglemesh";
    shape.triangleMesh = otk::pbrt::TriangleMeshData{ indices, points, {}, {} };
    return shape;
}

static otk::pbrt::ShapeDefinition singleTriangleTriangleMeshWithNormals( otk::pbrt::MeshData& buffers )
{
    otk::pbrt::ShapeDefinition        shape{ singleTriangleTriangleMesh( buffers ) };
    static const std::vector<float>   coords{ -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f };
    static std::vector<pbrt::Point3f> normals{ P3{ coords[0 + 0], coords[0 + 1], coords[0 + 2] },
                                               P3{ coords[3 + 0], coords[3 + 1], coords[3 + 2] },
                                               P3{ coords[6 + 0], coords[6 + 1], coords[6 + 2] } };
    buffers.normalCoords       = coords;
    shape.triangleMesh.normals = normals;
    return shape;
}

static otk::pbrt::ShapeDefinition singleTriangleTriangleMeshWithUVs( otk::pbrt::MeshData& buffers )
{
    otk::pbrt::ShapeDefinition        shape{ singleTriangleTriangleMesh( buffers ) };
    static const std::vector<float>   coords{ -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f };
    static std::vector<pbrt::Point2f> uvs{ P2{ coords[0 + 0], coords[0 + 1] }, P2{ coords[2 + 0], coords[2 + 1] },
                                           P2{ coords[4 + 0], coords[4 + 1] } };
    buffers.uvCoords       = coords;
    shape.triangleMesh.uvs = uvs;
    return shape;
}

TEST_F( TestGeometryCache, constructTriangleASForTriangleMesh )
{
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::MeshData        buffers;
    otk::pbrt::ShapeDefinition shape{ singleTriangleTriangleMesh( buffers ) };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto                 expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( buffers.vertexCoords ),
                                                                            hasDeviceIndices( buffers.indices ), hasSbtFlags( m_expectedFlags ),
                                                                            hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                            hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    EXPECT_NE( CUdeviceptr{}, m_geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, m_geom.traversable );
    EXPECT_EQ( nullptr, m_geom.devNormals );
    EXPECT_EQ( nullptr, m_geom.devUVs );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
}

TEST( TestHasDeviceTriangleNormals, normalsPointerIsNull )
{
    otk::pbrt::TriangleMeshData triangleMesh{};

    EXPECT_THAT( nullptr, Not( hasDeviceTriangleNormals( &triangleMesh ) ) );
}

static TriangleNormals expectedNormal()
{
    TriangleNormals actual;
    actual.N[0] = make_float3( 1.0f, 0.0f, 0.0f );
    actual.N[1] = make_float3( 0.0f, 1.0f, 0.0f );
    actual.N[2] = make_float3( 0.0f, 0.0f, 1.0f );
    return actual;
}

TEST( TestHasDeviceTriangleNormals, normalsDontMatchFirstVertex )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleNormals> actual;
    actual.resize( 1 );
    actual[0] = expectedNormal();
    actual.copyToDevice();
    const otk::pbrt::TriangleMeshData triangleMesh{
        { 0, 1, 2 }, {}, { P3{ 0.0f, 1.0f, 0.0f }, P3{ 0.0f, 1.0f, 0.0f }, P3{ 0.0f, 0.0f, 1.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), Not( hasDeviceTriangleNormals( &triangleMesh ) ) );
}

TEST( TestHasDeviceTriangleNormals, normalsDontMatchSecondVertex )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleNormals> actual;
    actual.resize( 1 );
    actual[0] = expectedNormal();
    actual.copyToDevice();
    otk::pbrt::TriangleMeshData triangleMesh{
        { 0, 1, 2 }, {}, { P3{ 1.0f, 0.0f, 0.0f }, P3{ 0.0f, 0.0f, 1.0f }, P3{ 0.0f, 0.0f, 1.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), Not( hasDeviceTriangleNormals( &triangleMesh ) ) );
}

TEST( TestHasDeviceTriangleNormals, normalsDontMatchThirdVertex )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleNormals> actual;
    actual.resize( 1 );
    actual[0] = expectedNormal();
    actual.copyToDevice();
    otk::pbrt::TriangleMeshData triangleMesh{
        { 0, 1, 2 }, {}, { P3{ 1.0f, 0.0f, 0.0f }, P3{ 0.0f, 1.0f, 0.0f }, P3{ 1.0f, 0.0f, 0.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), Not( hasDeviceTriangleNormals( &triangleMesh ) ) );
}

TEST( TestHasDeviceTriangleNormals, allNormalsMatch )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleNormals> actual;
    actual.resize( 1 );
    actual[0] = expectedNormal();
    actual.copyToDevice();
    otk::pbrt::TriangleMeshData triangleMesh{
        { 0, 1, 2 }, {}, { P3{ 1.0f, 0.0f, 0.0f }, P3{ 0.0f, 1.0f, 0.0f }, P3{ 0.0f, 0.0f, 1.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), hasDeviceTriangleNormals( &triangleMesh ) );
}

TEST_F( TestGeometryCache, constructTriangleASForTriangleMeshWithNormals )
{
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::MeshData        buffers;
    otk::pbrt::ShapeDefinition shape{ singleTriangleTriangleMeshWithNormals( buffers ) };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto                 expectedInput{ AllOf(
        NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( buffers.vertexCoords ),
                                                                     hasDeviceIndices( buffers.indices ), hasSbtFlags( m_expectedFlags ),
                                                                     hasNoPreTransform(), hasNoSbtIndexOffsets(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    EXPECT_NE( CUdeviceptr{}, m_geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, m_geom.traversable );
    EXPECT_THAT( m_geom.devNormals, hasDeviceTriangleNormals( &shape.triangleMesh ) );
    EXPECT_EQ( nullptr, m_geom.devUVs );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 3, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
}

TEST( TestHasDeviceTriangleUVs, uvsPointerIsNull )
{
    otk::pbrt::TriangleMeshData triangleMesh{};

    EXPECT_THAT( nullptr, Not( hasDeviceTriangleUVs( &triangleMesh ) ) );
}

static TriangleUVs expectedUV()
{
    TriangleUVs actual;
    actual.UV[0] = make_float2( 1.0f, 0.0f );
    actual.UV[1] = make_float2( 0.0f, 1.0f );
    actual.UV[2] = make_float2( 0.0f, 0.0f );
    return actual;
}

TEST( TestHasDeviceTriangleUVs, uvsDontMatchFirstVertex )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleUVs> actual;
    actual.resize( 1 );
    actual[0] = expectedUV();
    actual.copyToDevice();
    const otk::pbrt::TriangleMeshData triangleMesh{ { 0, 1, 2 }, {}, {}, { P2{ 1.0f, 1.0f }, P2{ 0.0f, 1.0f }, P2{ 0.0f, 0.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), Not( hasDeviceTriangleUVs( &triangleMesh ) ) );
}

TEST( TestHasDeviceTriangleUVs, uvsDontMatchSecondVertex )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleUVs> actual;
    actual.resize( 1 );
    actual[0] = expectedUV();
    actual.copyToDevice();
    const otk::pbrt::TriangleMeshData triangleMesh{ { 0, 1, 2 }, {}, {}, { P2{ 1.0f, 0.0f }, P2{ 1.0f, 1.0f }, P2{ 0.0f, 0.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), Not( hasDeviceTriangleUVs( &triangleMesh ) ) );
}

TEST( TestHasDeviceTriangleUVs, uvsDontMatchThirdVertex )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleUVs> actual;
    actual.resize( 1 );
    actual[0] = expectedUV();
    actual.copyToDevice();
    const otk::pbrt::TriangleMeshData triangleMesh{ { 0, 1, 2 }, {}, {}, { P2{ 1.0f, 0.0f }, P2{ 0.0f, 1.0f }, P2{ 1.0f, 1.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), Not( hasDeviceTriangleUVs( &triangleMesh ) ) );
}

TEST( TestHasDeviceTriangleUVs, allUVsMatch )
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    otk::SyncVector<TriangleUVs> actual;
    actual.resize( 1 );
    actual[0] = expectedUV();
    actual.copyToDevice();
    const otk::pbrt::TriangleMeshData triangleMesh{ { 0, 1, 2 }, {}, {}, { P2{ 1.0f, 0.0f }, P2{ 0.0f, 1.0f }, P2{ 0.0f, 0.0f } } };

    EXPECT_THAT( actual.typedDevicePtr(), hasDeviceTriangleUVs( &triangleMesh ) );
}

TEST_F( TestGeometryCache, constructTriangleASForTriangleMeshWithUVs )
{
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::MeshData        buffers;
    otk::pbrt::ShapeDefinition shape{ singleTriangleTriangleMeshWithUVs( buffers ) };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto                 expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( buffers.vertexCoords ),
                                                                            hasDeviceIndices( buffers.indices ), hasSbtFlags( m_expectedFlags ),
                                                                            hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                            hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    EXPECT_NE( CUdeviceptr{}, m_geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, m_geom.traversable );
    EXPECT_EQ( nullptr, m_geom.devNormals );
    EXPECT_THAT( m_geom.devUVs, hasDeviceTriangleUVs( &shape.triangleMesh ) );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 3, stats.numUVs );
}

TEST_F( TestGeometryCache, twoPlyInstancesShareSameGAS )
{
    expectPlyFileSizeReturned();
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::MeshData        buffers;
    otk::pbrt::MeshInfo        info{};
    otk::pbrt::ShapeDefinition shape{ singleTrianglePlyMesh( meshLoader, buffers, info ) };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto                 expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( buffers.vertexCoords ),
                                                                            hasDeviceIndices( buffers.indices ), hasSbtFlags( m_expectedFlags ),
                                                                            hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                            hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );
    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    const GeometryCacheEntry geom2{ m_geometryCache->getShape( m_fakeContext, m_stream, shape ) };

    EXPECT_EQ( geom2, m_geom );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
}

static otk::pbrt::ShapeDefinition singleSphere()
{
    otk::pbrt::SphereData sphere;
    sphere.radius = 10.0f;
    sphere.zMin   = -10.0f;
    sphere.zMax   = 10.0f;
    sphere.phiMax = 360.0f;
    otk::pbrt::ShapeDefinition shape{};
    shape.type   = "sphere";
    shape.sphere = sphere;
    return shape;
}

TEST_F( TestGeometryCache, constructSphereASForSphere )
{
    MockMeshLoaderPtr          meshLoader{ createMockMeshLoader() };
    otk::pbrt::ShapeDefinition shape{ singleSphere() };
    const auto                 expectedOptions{ buildAllowsRandomVertexAccess() };
    std::vector<float3>        centers{ make_float3( 0.0f, 0.0f, 0.0f ) };
    std::vector<float>         radii{ shape.sphere.radius };
    const auto                 expectedInput{
        AllOf( NotNull(), hasSphereBuildInput( 0, hasAll( hasDeviceSphereVertices( centers ),
                                                          hasDeviceSphereSingleRadius( shape.sphere.radius ) ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    m_geom = m_geometryCache->getShape( m_fakeContext, m_stream, shape );
    OTK_ERROR_CHECK( cudaDeviceSynchronize() );
    const Stats stats{ m_geometryCache->getStatistics() };

    EXPECT_NE( CUdeviceptr{}, m_geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, m_geom.traversable );
    EXPECT_EQ( nullptr, m_geom.devNormals );
    EXPECT_EQ( nullptr, m_geom.devUVs );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 0, stats.numTriangles );
    EXPECT_EQ( 1, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
}

struct TestObject
{
    otk::pbrt::ObjectDefinition      object;
    std::vector<otk::pbrt::MeshData> buffers;
    otk::pbrt::ShapeList             shapes;
    std::vector<float>               expectedVertices;
    std::vector<int>                 expectedIndices;
    std::vector<float3>              expectedSphereCenters;
    float                            expectedSphereSingleRadius;
};

static otk::pbrt::ShapeDefinition translateTriangleMesh( TestObject& object, int index, float tx, float ty, float tz )
{
    otk::pbrt::ShapeDefinition mesh{ singleTriangleTriangleMesh( object.buffers[index] ) };
    mesh.transform = ::pbrt::Translate( ::pbrt::Vector3f( tx, ty, tz ) );
    return mesh;
}

static TestObject twoTriangleMeshes()
{
    TestObject object;
    object.buffers.push_back( otk::pbrt::MeshData{} );
    object.buffers.push_back( otk::pbrt::MeshData{} );
    object.shapes.push_back( translateTriangleMesh( object, 0, 10.0f, 10.0f, 10.0f ) );
    object.shapes.push_back( singleTriangleTriangleMesh( object.buffers[1] ) );
    for( size_t i = 0; i < object.shapes.size(); ++i )
    {
        const std::vector<float>& vertices{ object.buffers[i].vertexCoords };
        for( size_t c = 0; c < vertices.size() / 3; ++c )
        {
            ::pbrt::Point3f pt{ vertices[c * 3 + 0], vertices[c * 3 + 1], vertices[c * 3 + 2] };
            pt = object.shapes[i].transform( pt );
            object.expectedVertices.push_back( pt.x );
            object.expectedVertices.push_back( pt.y );
            object.expectedVertices.push_back( pt.z );
        }
        const std::vector<int>& indices{ object.buffers[i].indices };
        std::copy( indices.cbegin(), indices.cend(), std::back_inserter( object.expectedIndices ) );
    }

    return object;
}

static TestObject oneTriangleMeshOnePlyMesh( MockMeshLoaderPtr loader, otk::pbrt::MeshInfo& info )
{
    // TODO: give each triangle mesh different transforms
    TestObject object;
    object.buffers.push_back( otk::pbrt::MeshData{} );
    object.buffers.push_back( otk::pbrt::MeshData{} );
    object.shapes.push_back( translateTriangleMesh( object, 0, 10.0f, 10.0f, 10.0f ) );
    object.shapes.push_back( singleTrianglePlyMesh( loader, object.buffers[1], info ) );

    for( size_t i = 0; i < object.shapes.size(); ++i )
    {
        const std::vector<float>& vertices{ object.buffers[i].vertexCoords };
        for( size_t c = 0; c < vertices.size() / 3; ++c )
        {
            ::pbrt::Point3f pt{ vertices[c * 3 + 0], vertices[c * 3 + 1], vertices[c * 3 + 2] };
            pt = object.shapes[i].transform( pt );
            object.expectedVertices.push_back( pt.x );
            object.expectedVertices.push_back( pt.y );
            object.expectedVertices.push_back( pt.z );
        }
        const std::vector<int>& indices{ object.buffers[i].indices };
        std::copy( indices.cbegin(), indices.cend(), std::back_inserter( object.expectedIndices ) );
    }

    return object;
}

static TestObject oneTriangleMeshOneSphere()
{
    TestObject object;
    object.buffers.push_back( otk::pbrt::MeshData{} );
    object.buffers.push_back( otk::pbrt::MeshData{} );
    object.shapes.push_back( translateTriangleMesh( object, 0, 10.0f, 10.0f, 10.0f ) );
    object.shapes.push_back( singleSphere() );
    {
        const std::vector<float>& vertices{ object.buffers[0].vertexCoords };
        for( size_t c = 0; c < vertices.size() / 3; ++c )
        {
            ::pbrt::Point3f pt{ vertices[c * 3 + 0], vertices[c * 3 + 1], vertices[c * 3 + 2] };
            pt = object.shapes[0].transform( pt );
            object.expectedVertices.push_back( pt.x );
            object.expectedVertices.push_back( pt.y );
            object.expectedVertices.push_back( pt.z );
        }
        const std::vector<int>& indices{ object.buffers[0].indices };
        std::copy( indices.cbegin(), indices.cend(), std::back_inserter( object.expectedIndices ) );
    }
    {
        object.expectedSphereCenters.push_back( make_float3( 0.0f, 0.0f, 0.0f ) );
        object.expectedSphereSingleRadius = object.shapes[1].sphere.radius;
    }

    return object;
}

TEST_F( TestGeometryCache, constructTriangleASForObjectTwoMeshes )
{
    const TestObject object{ twoTriangleMeshes() };
    const auto       expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto       expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( object.expectedVertices ),
                                                                  hasDeviceIndices( object.expectedIndices ), hasSbtFlags( m_expectedFlags ),
                                                                  hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                  hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    const std::vector<GeometryCacheEntry> result{
        m_geometryCache->getObject( m_fakeContext, m_stream, object.object, object.shapes ) };
    const Stats stats{ m_geometryCache->getStatistics() };

    ASSERT_FALSE( result.empty() );
    const GeometryCacheEntry& geom{ result[0] };
    EXPECT_NE( CUdeviceptr{}, geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, geom.traversable );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 2, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
    EXPECT_EQ( 0, stats.totalBytesRead );
}

TEST_F( TestGeometryCache, constructTriangleASForObjectOneTriMeshOnePlyMesh )
{
    expectPlyFileSizeReturned();
    MockMeshLoaderPtr   loader{ createMockMeshLoader() };
    otk::pbrt::MeshInfo info{};
    const TestObject    object{ oneTriangleMeshOnePlyMesh( loader, info ) };
    const auto          expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto          expectedInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( object.expectedVertices ),
                                                                     hasDeviceIndices( object.expectedIndices ), hasSbtFlags( m_expectedFlags ),
                                                                     hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                     hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedInput );
    configureAccelBuild( expectedOptions, expectedInput );

    const std::vector<GeometryCacheEntry> result{
        m_geometryCache->getObject( m_fakeContext, m_stream, object.object, object.shapes ) };
    const Stats stats = m_geometryCache->getStatistics();

    ASSERT_FALSE( result.empty() );
    const GeometryCacheEntry& geom{ result[0] };
    EXPECT_NE( CUdeviceptr{}, geom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, geom.traversable );
    EXPECT_EQ( nullptr, geom.devNormals );
    EXPECT_EQ( nullptr, geom.devUVs );
    EXPECT_EQ( 1, stats.numTraversables );
    EXPECT_EQ( 2, stats.numTriangles );
    EXPECT_EQ( 0, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
    EXPECT_EQ( ARBITRARY_PLY_FILE_SIZE, stats.totalBytesRead );
}

TEST_F( TestGeometryCache, constructTriangleASForObjectOneTriMeshOneSphere )
{
    const TestObject object{ oneTriangleMeshOneSphere() };
    const auto       expectedOptions{ buildAllowsRandomVertexAccess() };
    const auto       expectedTriangleInput{
        AllOf( NotNull(), hasTriangleBuildInput( 0, hasAll( hasDeviceVertexCoords( object.expectedVertices ),
                                                                  hasDeviceIndices( object.expectedIndices ), hasSbtFlags( m_expectedFlags ),
                                                                  hasNoPreTransform(), hasNoSbtIndexOffsets(),
                                                                  hasNoPrimitiveIndexOffset(), hasNoOpacityMap() ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedTriangleInput );
    configureAccelBuild( expectedOptions, expectedTriangleInput );
    const auto expectedSphereInput{
        AllOf( NotNull(), hasSphereBuildInput( 0, hasAll( hasDeviceSphereVertices( object.expectedSphereCenters ),
                                                          hasDeviceSphereSingleRadius( object.expectedSphereSingleRadius ) ) ) ) };
    configureAccelComputeMemoryUsage( expectedOptions, expectedSphereInput );
    configureAccelBuild( expectedOptions, expectedSphereInput );

    const std::vector<GeometryCacheEntry> result{
        m_geometryCache->getObject( m_fakeContext, m_stream, object.object, object.shapes ) };
    const Stats stats{ m_geometryCache->getStatistics() };

    ASSERT_EQ( 2U, result.size() );
    const GeometryCacheEntry& meshGeom{ result[0] };
    EXPECT_NE( CUdeviceptr{}, meshGeom.accelBuffer );
    EXPECT_EQ( m_fakeGeomAS, meshGeom.traversable );
    EXPECT_EQ( nullptr, meshGeom.devNormals );
    EXPECT_EQ( nullptr, meshGeom.devUVs );
    const GeometryCacheEntry& sphereGeom{ result[1] };
    EXPECT_NE( CUdeviceptr{}, sphereGeom.accelBuffer);
    EXPECT_EQ( 2, stats.numTraversables );
    EXPECT_EQ( 1, stats.numTriangles );
    EXPECT_EQ( 1, stats.numSpheres );
    EXPECT_EQ( 0, stats.numNormals );
    EXPECT_EQ( 0, stats.numUVs );
    EXPECT_EQ( 0, stats.totalBytesRead );
}
