// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/GeometryCache.h"

#include "DemandPbrtScene/MaterialAdapters.h"
#include "DemandPbrtScene/Stopwatch.h"

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iterator>

using namespace otk::pbrt;

namespace demandPbrtScene {

namespace {

class GeometryCacheImpl : public GeometryCache
{
  public:
    GeometryCacheImpl( FileSystemInfoPtr fileSystemInfo )
        : m_fileSystemInfo( std::move( fileSystemInfo ) )
    {
    }
    ~GeometryCacheImpl() override = default;

    GeometryCacheEntry getShape( OptixDeviceContext context, CUstream stream, const ShapeDefinition& shape ) override;

    GeometryCacheEntry getObject( OptixDeviceContext      context,
                                  CUstream                stream,
                                  const ObjectDefinition& object,
                                  const ShapeList&        shapes,
                                  GeometryPrimitive       primitive,
                                  MaterialFlags           flags ) override;

    GeometryCacheStatistics getStatistics() const override { return m_stats; }

  private:
    GeometryCacheEntry getPlyMesh( OptixDeviceContext context, CUstream stream, const PlyMeshData& plyMesh );
    GeometryCacheEntry getTriangleMesh( OptixDeviceContext context, CUstream stream, const TriangleMeshData& mesh );
    GeometryCacheEntry getSphere( OptixDeviceContext context, CUstream stream, const SphereData& sphere );
    GeometryCacheEntry buildTriangleGAS( OptixDeviceContext context, CUstream stream );
    GeometryCacheEntry buildSphereGAS( OptixDeviceContext context, CUstream stream );
    GeometryCacheEntry buildGAS( OptixDeviceContext     context,
                                 CUstream               stream,
                                 GeometryPrimitive      primitive,
                                 TriangleNormals*       normals,
                                 TriangleUVs*           uvs,
                                 const OptixBuildInput& build );
    void               appendPlyMesh( const pbrt::Transform& transform, const PlyMeshData& plyMesh );
    void               appendTriangleMesh( const pbrt::Transform& transform, const TriangleMeshData& mesh );
    void               appendSphere( const pbrt::Transform& transform, const SphereData& sphereData );

    FileSystemInfoPtr                         m_fileSystemInfo;
    std::map<std::string, GeometryCacheEntry> m_plyCache;
    otk::SyncVector<float3>                   m_vertices;
    otk::SyncVector<std::uint32_t>            m_indices;
    otk::SyncVector<float>                    m_radii;
    otk::SyncVector<TriangleNormals>          m_normals;
    otk::SyncVector<TriangleUVs>              m_uvs;
    std::vector<uint_t>                       m_primitiveGroupBeginIndices;
    GeometryCacheStatistics                   m_stats{};
};

GeometryCacheEntry GeometryCacheImpl::getShape( OptixDeviceContext context, CUstream stream, const ShapeDefinition& shape )
{
    if( shape.type == SHAPE_TYPE_PLY_MESH )
        return getPlyMesh( context, stream, shape.plyMesh );

    if( shape.type == SHAPE_TYPE_TRIANGLE_MESH )
        return getTriangleMesh( context, stream, shape.triangleMesh );

    if( shape.type == SHAPE_TYPE_SPHERE )
        return getSphere( context, stream, shape.sphere );

    return {};
}

std::string toString( GeometryPrimitive primitive )
{
    switch( primitive )
    {
        case GeometryPrimitive::NONE:
            return "NONE";
        case GeometryPrimitive::TRIANGLE:
            return "TRIANGLE";
        case GeometryPrimitive::SPHERE:
            return "SPHERE";
    }
    return "?Unknown (" + std::to_string( static_cast<int>( primitive ) ) + ")";
}

GeometryCacheEntry GeometryCacheImpl::getObject( OptixDeviceContext      context,
                                                 CUstream                stream,
                                                 const ObjectDefinition& object,
                                                 const ShapeList&        shapes,
                                                 GeometryPrimitive       primitive,
                                                 MaterialFlags           flags )
{
    m_vertices.clear();
    m_indices.clear();
    m_normals.clear();
    m_uvs.clear();
    m_primitiveGroupBeginIndices.clear();

    switch( primitive )
    {
        case GeometryPrimitive::TRIANGLE:
            for( const ShapeDefinition& shape : shapes )
            {
                if( shapeMaterialFlags( shape ) == flags )
                {
                    if( shape.type == SHAPE_TYPE_TRIANGLE_MESH )
                    {
                        appendTriangleMesh( shape.transform, shape.triangleMesh );
                    }
                    else if( shape.type == SHAPE_TYPE_PLY_MESH )
                    {
                        appendPlyMesh( shape.transform, shape.plyMesh );
                    }
                }
            }
            return buildTriangleGAS( context, stream );

        case GeometryPrimitive::SPHERE:
            for( const ShapeDefinition& shape : shapes )
            {
                if( shapeMaterialFlags( shape ) == flags && shape.type == SHAPE_TYPE_SPHERE )
                {
                    appendSphere( shape.transform, shape.sphere );
                }
            }
            return buildSphereGAS( context, stream );

        case GeometryPrimitive::NONE:
            break;
    }

    throw std::runtime_error( "Unknown primitive type " + toString( primitive ) );
}

GeometryCacheEntry GeometryCacheImpl::getPlyMesh( OptixDeviceContext context, CUstream stream, const PlyMeshData& plyMesh )
{
    const auto it{ m_plyCache.find( plyMesh.fileName ) };
    if( it != m_plyCache.end() )
    {
        return it->second;
    }

    m_primitiveGroupBeginIndices.clear();
    m_primitiveGroupBeginIndices.push_back( 0 );

    const MeshLoaderPtr loader{ plyMesh.loader };
    const MeshInfo      meshInfo{ loader->getMeshInfo() };
    MeshData            buffers{};
    {
        Stopwatch stopwatch;
        loader->load( buffers );
        m_stats.totalReadTime += stopwatch.elapsed();
    }
    m_stats.totalBytesRead += m_fileSystemInfo->getSize( plyMesh.fileName );
    m_vertices.resize( meshInfo.numVertices );
    for( int i = 0; i < meshInfo.numVertices; ++i )
    {
        m_vertices[i] =
            make_float3( buffers.vertexCoords[i * 3 + 0], buffers.vertexCoords[i * 3 + 1], buffers.vertexCoords[i * 3 + 2] );
    }
    m_indices.resize( meshInfo.numTriangles * 3U );
    OTK_ASSERT( meshInfo.numTriangles * 3U == buffers.indices.size() );
    std::transform( buffers.indices.begin(), buffers.indices.end(), m_indices.begin(),
                    []( int index ) { return static_cast<std::uint32_t>( index ); } );

    if( meshInfo.numNormals > 0 )
    {
        if( meshInfo.numNormals != meshInfo.numVertices )
        {
            throw std::runtime_error( "Expected " + std::to_string( meshInfo.numVertices ) + " vertex normals, got "
                                      + std::to_string( meshInfo.numNormals ) );
        }

        // When building the GAS, we have the luxury of supplying the vertex array and the
        // index array, but in the closest hit program, we only have the primitive index,
        // not the vertex/normal index.  So we size the array of TriangleNormals structures
        // to the number of primitives and use the index array to select the appropriate normal
        // for each vertex.
        //
        m_normals.resize( m_indices.size() / 3 );
        for( size_t face = 0; face < m_normals.size(); ++face )
        {
            for( size_t vert = 0; vert < 3; ++vert )
            {
                // 3 coords per vertex
                // 3 vertices per face, 3 indices per face
                const int idx           = buffers.indices[face * 3 + vert] * 3;
                m_normals[face].N[vert] = make_float3( buffers.normalCoords[idx + 0], buffers.normalCoords[idx + 1],
                                                       buffers.normalCoords[idx + 2] );
            }
        }
    }

    if( meshInfo.numTextureCoordinates > 0 )
    {
        if( meshInfo.numTextureCoordinates != meshInfo.numVertices )
        {
            throw std::runtime_error( "Expected " + std::to_string( meshInfo.numVertices )
                                      + " vertex texture coordinates, got " + std::to_string( meshInfo.numTextureCoordinates ) );
        }

        // When building the GAS, we have the luxury of supplying the vertex array and the
        // index array, but in the closest hit program, we only have the primitive index,
        // not the vertex/normal/uv index.  So we size the array of TriangleUVs structures
        // to the number of primitives and use the index array to select the appropriate normal
        // for each vertex.
        m_uvs.resize( m_indices.size() / 3 );
        for( size_t face = 0; face < m_uvs.size(); ++face )
        {
            for( size_t vert = 0; vert < 3; ++vert )
            {
                // 2 coords per vertex
                // 3 vertices per face, 3 indices per face
                const int idx        = buffers.indices[face * 3 + vert] * 2;
                m_uvs[face].UV[vert] = make_float2( buffers.uvCoords[idx + 0], buffers.uvCoords[idx + 1] );
            }
        }
    }

    const GeometryCacheEntry result{ buildTriangleGAS( context, stream ) };
    m_plyCache[plyMesh.fileName] = result;
    return result;
}

GeometryCacheEntry GeometryCacheImpl::getTriangleMesh( OptixDeviceContext context, CUstream stream, const TriangleMeshData& triangleMesh )
{
    // TODO: implement a cache for trianglemesh shapes?
    m_vertices.clear();
    m_indices.clear();
    m_normals.clear();
    m_uvs.clear();
    m_primitiveGroupBeginIndices.clear();
    appendTriangleMesh( pbrt::Transform(), triangleMesh );
    return buildTriangleGAS( context, stream );
}

GeometryCacheEntry GeometryCacheImpl::buildGAS( OptixDeviceContext     context,
                                                CUstream               stream,
                                                GeometryPrimitive      primitive,
                                                TriangleNormals*       normals,
                                                TriangleUVs*           uvs,
                                                const OptixBuildInput& build )
{
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( context, &options, &build, 1, &sizes ) );

    otk::DeviceBuffer temp;
    temp.resize( sizes.tempSizeInBytes );
    otk::DeviceBuffer output;
    output.resize( sizes.outputSizeInBytes );
    OptixTraversableHandle traversable{};
    OTK_ERROR_CHECK( optixAccelBuild( context, stream, &options, &build, 1, temp, temp.size(), output, output.size(),
                                      &traversable, nullptr, 0 ) );
#ifndef NDEBUG
    OTK_CUDA_SYNC_CHECK();
#endif

    ++m_stats.numTraversables;
    switch( primitive )
    {
        case GeometryPrimitive::NONE:
            break;
        case GeometryPrimitive::TRIANGLE:
            m_stats.numTriangles += build.triangleArray.numIndexTriplets;
            m_stats.numNormals += static_cast<unsigned int>( m_normals.size() * 3 );
            m_stats.numUVs += static_cast<unsigned int>( m_uvs.size() * 3 );
            break;
        case GeometryPrimitive::SPHERE:
            m_stats.numSpheres += build.sphereArray.numVertices;
            break;
    }

    return { output.detach(), traversable, primitive, normals, uvs, m_primitiveGroupBeginIndices };
}

template <typename Container>
void growContainer( Container& coll, size_t increase )
{
    coll.reserve( coll.size() + increase );
}

void GeometryCacheImpl::appendPlyMesh( const pbrt::Transform& transform, const PlyMeshData& plyMesh )
{
    m_primitiveGroupBeginIndices.push_back( static_cast<uint_t>( m_vertices.size() / 3U ) );
    const MeshLoaderPtr loader{ plyMesh.loader };
    const MeshInfo      meshInfo{ loader->getMeshInfo() };
    MeshData            buffers{};
    {
        Stopwatch stopwatch;
        loader->load( buffers );
        m_stats.totalReadTime += stopwatch.elapsed();
    }
    m_stats.totalBytesRead += m_fileSystemInfo->getSize( plyMesh.fileName );

    growContainer( m_vertices, meshInfo.numVertices );
    for( int i = 0; i < meshInfo.numVertices; ++i )
    {
        pbrt::Point3f pt{ buffers.vertexCoords[i * 3 + 0], buffers.vertexCoords[i * 3 + 1], buffers.vertexCoords[i * 3 + 2] };
        pt = transform( pt );
        m_vertices.push_back( make_float3( pt.x, pt.y, pt.z ) );
    }
    growContainer( m_indices, meshInfo.numTriangles * 3U );
    std::transform( buffers.indices.begin(), buffers.indices.end(), std::back_inserter( m_indices ),
                    []( int index ) { return static_cast<std::uint32_t>( index ); } );

    const size_t numTriangles{ m_indices.size() / 3 };
    if( meshInfo.numNormals > 0 )
    {
        if( meshInfo.numNormals != meshInfo.numVertices )
        {
            throw std::runtime_error( "Expected " + std::to_string( meshInfo.numVertices ) + " vertex normals, got "
                                      + std::to_string( meshInfo.numNormals ) );
        }

        // When building the GAS, we have the luxury of supplying the vertex array and the
        // index array, but in the closest hit program, we only have the primitive index,
        // not the vertex/normal index.  So we size the array of TriangleNormals structures
        // to the number of primitives and use the index array to select the appropriate normal
        // for each vertex.
        //
        growContainer( m_normals, numTriangles );
        for( size_t face = 0; face < numTriangles; ++face )
        {
            TriangleNormals normals;
            for( size_t vert = 0; vert < 3; ++vert )
            {
                // 3 coords per vertex
                // 3 vertices per face, 3 indices per face
                const int idx{ buffers.indices[face * 3 + vert] * 3 };
                normals.N[vert] = make_float3( buffers.normalCoords[idx + 0], buffers.normalCoords[idx + 1],
                                               buffers.normalCoords[idx + 2] );
            }
            m_normals.push_back( normals );
        }
    }

    if( meshInfo.numTextureCoordinates > 0 )
    {
        if( meshInfo.numTextureCoordinates != meshInfo.numVertices )
        {
            throw std::runtime_error( "Expected " + std::to_string( meshInfo.numVertices )
                                      + " vertex texture coordinates, got " + std::to_string( meshInfo.numTextureCoordinates ) );
        }

        // When building the GAS, we have the luxury of supplying the vertex array and the
        // index array, but in the closest hit program, we only have the primitive index,
        // not the vertex/normal/uv index.  So we size the array of TriangleUVs structures
        // to the number of primitives and use the index array to select the appropriate normal
        // for each vertex.
        growContainer( m_uvs, numTriangles );
        for( size_t face = 0; face < numTriangles; ++face )
        {
            TriangleUVs uvs;
            for( size_t vert = 0; vert < 3; ++vert )
            {
                // 2 coords per vertex
                // 3 vertices per face, 3 indices per face
                const int idx{ buffers.indices[face * 3 + vert] * 2 };
                uvs.UV[vert] = make_float2( buffers.uvCoords[idx + 0], buffers.uvCoords[idx + 1] );
            }
            m_uvs.push_back( uvs );
        }
    }
}

void GeometryCacheImpl::appendTriangleMesh( const pbrt::Transform& transform, const TriangleMeshData& triangleMesh )
{
    m_primitiveGroupBeginIndices.push_back( static_cast<uint_t>( m_vertices.size() / 3U ) );
    auto toFloat3 = [&]( const pbrt::Point3f& point ) {
        const pbrt::Point3f pt{ transform( point ) };
        return make_float3( pt.x, pt.y, pt.z );
    };
    std::transform( triangleMesh.points.begin(), triangleMesh.points.end(), std::back_inserter( m_vertices ), toFloat3 );
    growContainer( m_indices, triangleMesh.indices.size() );
    std::transform( triangleMesh.indices.begin(), triangleMesh.indices.end(), std::back_inserter( m_indices ),
                    []( const int index ) { return static_cast<std::uint32_t>( index ); } );

    const size_t numTriangles{ triangleMesh.indices.size() / 3 };
    if( !triangleMesh.normals.empty() )
    {
        // When building the GAS, we have the luxury of supplying the vertex array and the
        // index array, but in the closest hit program, we only have the primitive index,
        // not the vertex/normal index.  So we size the array of TriangleNormals structures
        // to the number of primitives and use the index array to select the appropriate normal
        // for each vertex.
        growContainer( m_normals, numTriangles );
        for( size_t i = 0; i < numTriangles; ++i )
        {
            TriangleNormals normals;
            for( int j = 0; j < 3; ++j )
            {
                normals.N[j] = toFloat3( transform( triangleMesh.normals[triangleMesh.indices[i * 3 + j]] ) );
            }
            m_normals.push_back( normals );
        }
    }
    if( !triangleMesh.uvs.empty() )
    {
        auto toFloat2 = []( const pbrt::Point2f& value ) { return make_float2( value.x, value.y ); };
        growContainer( m_uvs, numTriangles );
        for( size_t i = 0; i < numTriangles; ++i )
        {
            TriangleUVs uvs;
            for( int vert = 0; vert < 3; ++vert )
            {
                uvs.UV[vert] = toFloat2( triangleMesh.uvs[triangleMesh.indices[i * 3 + vert]] );
            }
            m_uvs.push_back( uvs );
        }
    }
}

void GeometryCacheImpl::appendSphere( const pbrt::Transform& transform, const SphereData& sphere )
{
    m_primitiveGroupBeginIndices.push_back( static_cast<unsigned int>( m_vertices.size() ) );
    const pbrt::Point3f center{ transform( pbrt::Point3f( 0.0f, 0.0f, 0.0f ) ) };
    m_vertices.push_back( make_float3( center.x, center.y, center.z ) );
    const pbrt::Point3f scaledUnitVector{ transform( pbrt::Point3f( 1.0f, 0.0f, 0.0f ) ) };
    m_radii.push_back( scaledUnitVector.x * sphere.radius );
}

GeometryCacheEntry GeometryCacheImpl::buildSphereGAS( OptixDeviceContext context, CUstream stream )
{
    m_vertices.copyToDeviceAsync( stream );
    m_radii.copyToDeviceAsync( stream );

    OptixBuildInput build{};
    build.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

    OptixBuildInputSphereArray& spheres = build.sphereArray;

    CUdeviceptr vertexBuffers[]{ m_vertices.detach() };
    spheres.vertexBuffers = vertexBuffers;
    spheres.numVertices   = 1;
    CUdeviceptr radiusBuffers[]{ m_radii.detach() };
    spheres.radiusBuffers = radiusBuffers;
    spheres.singleRadius  = 1;
    const uint_t flags    = OPTIX_GEOMETRY_FLAG_NONE;
    spheres.flags         = &flags;
    spheres.numSbtRecords = 1;

    return buildGAS( context, stream, GeometryPrimitive::SPHERE, nullptr, nullptr, build );
}

GeometryCacheEntry GeometryCacheImpl::getSphere( OptixDeviceContext context, CUstream stream, const SphereData& sphere )
{
    m_vertices.clear();
    m_radii.clear();
    m_primitiveGroupBeginIndices.clear();
    appendSphere( pbrt::Transform(), sphere );
    return buildSphereGAS( context, stream );
}

GeometryCacheEntry GeometryCacheImpl::buildTriangleGAS( OptixDeviceContext context, CUstream stream )
{
    m_vertices.copyToDeviceAsync( stream );
    m_indices.copyToDeviceAsync( stream );
    m_normals.copyToDevice();
    m_uvs.copyToDevice();

    OptixBuildInput build{};
    build.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    OptixBuildInputTriangleArray& triangles = build.triangleArray;

    CUdeviceptr m_vertexBuffers[]{ m_vertices.detach() };
    triangles.vertexBuffers    = m_vertexBuffers;
    triangles.numVertices      = m_vertices.size();
    triangles.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangles.indexBuffer      = m_indices.detach();
    triangles.numIndexTriplets = m_indices.size() / 3;
    triangles.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    const uint_t flags         = OPTIX_GEOMETRY_FLAG_NONE;
    triangles.flags            = &flags;
    triangles.numSbtRecords    = 1;

    TriangleNormals* triangleNormals = m_normals.typedDevicePtr();
    TriangleUVs*     triangleUVs     = m_uvs.typedDevicePtr();
    static_cast<void>( m_normals.detach() );
    static_cast<void>( m_uvs.detach() );
    return buildGAS( context, stream, GeometryPrimitive::TRIANGLE, triangleNormals, triangleUVs, build );
}

class FileSystemInfoImpl : public FileSystemInfo
{
  public:
    ~FileSystemInfoImpl() override = default;
    unsigned long long getSize( const std::string& path ) const override
    {
        return static_cast<unsigned long long>( std::filesystem::file_size( path ) );
    }
};

}  // namespace

FileSystemInfoPtr createFileSystemInfo()
{
    return std::make_shared<FileSystemInfoImpl>();
}

GeometryCachePtr createGeometryCache( FileSystemInfoPtr fileSystemInfo )
{
    return std::make_shared<GeometryCacheImpl>( std::move( fileSystemInfo ) );
}

}  // namespace demandPbrtScene
