//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "GeometryCache.h"

#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>

#include <chrono>
#include <cstdint>
#include <filesystem>

namespace demandPbrtScene {

namespace {

class Stopwatch
{
public:
    Stopwatch()
        : startTime( std::chrono::high_resolution_clock::now() )
    {
    }

    /// Returns the time in seconds since the Stopwatch was constructed.
    double elapsed() const
    {
        using namespace std::chrono;
        return duration_cast<duration<double>>( high_resolution_clock::now() - startTime ).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

class GeometryCacheImpl : public GeometryCache
{
  public:
    GeometryCacheImpl( FileSystemInfoPtr fileSystemInfo )
        : m_fileSystemInfo( fileSystemInfo )
    {
    }
    ~GeometryCacheImpl() override = default;

    GeometryCacheEntry getShape( OptixDeviceContext context, CUstream stream, const otk::pbrt::ShapeDefinition& shape ) override;

    GeometryCacheStatistics getStatistics() const override { return m_stats; }

  private:
    GeometryCacheEntry getPlyMesh( OptixDeviceContext context, CUstream stream, const otk::pbrt::PlyMeshData& plyMesh );
    GeometryCacheEntry getTriangleMesh( OptixDeviceContext context, CUstream stream, const otk::pbrt::TriangleMeshData& mesh );
    GeometryCacheEntry getSphere( OptixDeviceContext context, CUstream stream, const otk::pbrt::SphereData& sphere );
    GeometryCacheEntry buildTriangleGAS( OptixDeviceContext context, CUstream stream );
    GeometryCacheEntry buildGAS( OptixDeviceContext     context,
                                 CUstream               stream,
                                 GeometryPrimitive      primitive,
                                 TriangleNormals*       normals,
                                 TriangleUVs*           uvs,
                                 const OptixBuildInput& build );

    std::shared_ptr<FileSystemInfo>           m_fileSystemInfo;
    std::map<std::string, GeometryCacheEntry> m_plyCache;
    otk::SyncVector<float3>                   m_vertices;
    otk::SyncVector<std::uint32_t>            m_indices;
    otk::SyncVector<float>                    m_radii;
    otk::SyncVector<TriangleNormals>          m_normals;
    otk::SyncVector<TriangleUVs>              m_uvs;
    GeometryCacheStatistics                   m_stats{};
};

GeometryCacheEntry GeometryCacheImpl::getShape( OptixDeviceContext context, CUstream stream, const otk::pbrt::ShapeDefinition& shape )
{
    if( shape.type == "plymesh" )
        return getPlyMesh( context, stream, shape.plyMesh );

    if( shape.type == "trianglemesh" )
        return getTriangleMesh( context, stream, shape.triangleMesh );

    if( shape.type == "sphere" )
        return getSphere( context, stream, shape.sphere );

    return {};
}

GeometryCacheEntry GeometryCacheImpl::getPlyMesh( OptixDeviceContext context, CUstream stream, const otk::pbrt::PlyMeshData& plyMesh )
{
    auto it = m_plyCache.find( plyMesh.fileName );
    if( it != m_plyCache.end() )
    {
        return it->second;
    }

    const otk::pbrt::MeshLoaderPtr loader   = plyMesh.loader;
    const otk::pbrt::MeshInfo      meshInfo = loader->getMeshInfo();
    otk::pbrt::MeshData            buffers{};
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
    m_vertices.copyToDeviceAsync( stream );
    m_indices.resize( meshInfo.numTriangles * 3U );
    OTK_ASSERT( meshInfo.numTriangles * 3U == buffers.indices.size() );
    std::transform( buffers.indices.begin(), buffers.indices.end(), m_indices.begin(),
                    []( int index ) { return static_cast<std::uint32_t>( index ); } );
    m_indices.copyToDeviceAsync( stream );

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
        m_normals.copyToDevice();
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
        m_uvs.copyToDevice();
    }

    const GeometryCacheEntry result = buildTriangleGAS( context, stream );
    m_plyCache[plyMesh.fileName]    = result;
    return result;
}

GeometryCacheEntry GeometryCacheImpl::getTriangleMesh( OptixDeviceContext context, CUstream stream, const otk::pbrt::TriangleMeshData& triangleMesh )
{
    // TODO: implement a cache for trianglemesh shapes?

    m_vertices.resize( triangleMesh.points.size() );
    auto toFloat3 = []( const ::pbrt::Point3f& point ) { return make_float3( point.x, point.y, point.z ); };
    std::transform( triangleMesh.points.begin(), triangleMesh.points.end(), m_vertices.begin(), toFloat3 );
    m_vertices.copyToDeviceAsync( stream );
    m_indices.resize( triangleMesh.indices.size() );
    std::transform( triangleMesh.indices.begin(), triangleMesh.indices.end(), m_indices.begin(),
                    []( const int index ) { return static_cast<std::uint16_t>( index ); } );
    m_indices.copyToDeviceAsync( stream );

    if( !triangleMesh.normals.empty() )
    {
        // When building the GAS, we have the luxury of supplying the vertex array and the
        // index array, but in the closest hit program, we only have the primitive index,
        // not the vertex/normal index.  So we size the array of TriangleNormals structures
        // to the number of primitives and use the index array to select the appropriate normal
        // for each vertex.
        m_normals.resize( m_indices.size() / 3 );
        for( size_t i = 0; i < m_normals.size(); ++i )
            for( int j = 0; j < 3; ++j )
                m_normals[i].N[j] = toFloat3( triangleMesh.normals[triangleMesh.indices[i * 3 + j]] );
        m_normals.copyToDevice();
    }
    if( !triangleMesh.uvs.empty() )
    {
        auto toFloat2 = []( const pbrt::Point2f& value ) { return make_float2( value.x, value.y ); };
        m_uvs.resize( m_indices.size() / 3 );
        for( size_t i = 0; i < m_uvs.size(); ++i )
            for( int vert = 0; vert < 3; ++vert )
                m_uvs[i].UV[vert] = toFloat2( triangleMesh.uvs[triangleMesh.indices[i * 3 + vert]] );
        m_uvs.copyToDevice();
    }

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

    return { output.detach(), traversable, primitive, normals, uvs };
}

GeometryCacheEntry GeometryCacheImpl::getSphere( OptixDeviceContext context, CUstream stream, const otk::pbrt::SphereData& sphere )
{
    m_vertices.resize( 1 );
    m_vertices[0] = make_float3( 0.0f, 0.0f, 0.0f );
    m_vertices.copyToDeviceAsync( stream );
    m_radii.resize( 1 );
    m_radii[0] = sphere.radius;
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

GeometryCacheEntry GeometryCacheImpl::buildTriangleGAS( OptixDeviceContext context, CUstream stream )
{
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
    return std::make_shared<GeometryCacheImpl>(fileSystemInfo);
}

}  // namespace demandPbrtScene
