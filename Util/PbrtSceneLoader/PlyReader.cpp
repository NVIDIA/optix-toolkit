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

#include <OptiXToolkit/PbrtSceneLoader/PlyReader.h>

#include <OptiXToolkit/Memory/BitCast.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace ply {

namespace {

class PlyHandle
{
  public:
    explicit PlyHandle( p_ply handle )
        : m_handle( handle )
    {
    }
    ~PlyHandle()
    {
        if( m_handle != nullptr )
        {
            ply_close( m_handle );
            m_handle = nullptr;
        }
    }
    operator p_ply() const { return m_handle; }

  private:
    p_ply m_handle;
};

enum
{
    PLY_ERROR   = 0,
    PLY_SUCCESS = 1,
};

class MeshLoader : public ::otk::pbrt::MeshLoader
{
  public:
    MeshLoader( const std::string& filename, otk::pbrt::MeshInfo info )
        : m_filename( filename )
        , m_meshInfo( info )
    {
    }
    ~MeshLoader() override = default;

    otk::pbrt::MeshInfo getMeshInfo() const override { return m_meshInfo; }

    void load( otk::pbrt::MeshData& buffers ) override;

  private:
    std::string         m_filename;
    otk::pbrt::MeshInfo m_meshInfo;

    static void s_errorCallback( p_ply ply, const char* message );
};

int readFloat( p_ply_argument arg )
{
    std::vector<float>* buffer{};
    long                offset{};
    if( ply_get_argument_user_data( arg, otk::bit_cast<void**>( &buffer ), &offset ) == PLY_ERROR )
        return PLY_ERROR;

    long element{};
    if( ply_get_argument_element( arg, nullptr, &element ) == PLY_ERROR )
        return PLY_ERROR;

    const long stride = offset >> 4U;
    offset &= 0xF;
    ( *buffer )[element * stride + offset] = static_cast<float>( ply_get_argument_value( arg ) );
    return PLY_SUCCESS;
}

int readIntList( p_ply_argument arg )
{
    long length{};
    long valueIndex{};
    if( ply_get_argument_property( arg, nullptr, &length, &valueIndex ) == PLY_ERROR )
    {
        return PLY_ERROR;
    }
    if( valueIndex == -1 )  // we just read the list length
    {
        // Quads are ignored.
        return length == 3 || length == 4 ? PLY_SUCCESS : PLY_ERROR;
    }

    std::vector<int>* buffer{};
    long              stride{};
    if( ply_get_argument_user_data( arg, otk::bit_cast<void**>( &buffer ), &stride ) == PLY_ERROR )
        return PLY_ERROR;

    long element{};
    if( ply_get_argument_element( arg, nullptr, &element ) == PLY_ERROR )
        return PLY_ERROR;

    ( *buffer )[element * stride + valueIndex] = static_cast<int>( ply_get_argument_value( arg ) );
    return PLY_SUCCESS;
}

template <int N, typename T>
int numOf( T ( & /*array*/ )[N] )
{
    return N;
}

void MeshLoader::s_errorCallback( p_ply ply, const char* message )
{
    throw std::runtime_error( message );
}    

void MeshLoader::load( otk::pbrt::MeshData& buffers )
{
    PlyHandle ply( ply_open( m_filename.c_str(), s_errorCallback, 0, nullptr ) );
    if( ply == nullptr )
    {
        throw std::runtime_error( m_filename + " is not a valid PLY file." );
    }

    if( ply_read_header( ply ) == PLY_ERROR )
    {
        throw std::runtime_error( "Could not read the PLY header of " + m_filename );
    }
    buffers.vertexCoords.resize( m_meshInfo.numVertices * 3 );
    buffers.indices.resize( m_meshInfo.numTriangles * 3 );
    buffers.normalCoords.resize( m_meshInfo.numNormals * 3 );
    buffers.uvCoords.resize( m_meshInfo.numTextureCoordinates * 2 );

    auto       makeStride   = []( int value ) { return value << 4U; };
    const long float3Stride = makeStride( 3 );
    const long numVertices  = ply_set_read_cb( ply, "vertex", "x", readFloat, &buffers.vertexCoords, float3Stride | 0 );
    ply_set_read_cb( ply, "vertex", "y", readFloat, &buffers.vertexCoords, float3Stride | 1 );
    ply_set_read_cb( ply, "vertex", "z", readFloat, &buffers.vertexCoords, float3Stride | 2 );

    const long int3ListStride = 3;
    const long numFaces = ply_set_read_cb( ply, "face", "vertex_indices", readIntList, &buffers.indices, int3ListStride );

    const long numNormals = ply_set_read_cb( ply, "vertex", "nx", readFloat, &buffers.normalCoords, float3Stride | 0 );
    ply_set_read_cb( ply, "vertex", "ny", readFloat, &buffers.normalCoords, float3Stride | 1 );
    ply_set_read_cb( ply, "vertex", "nz", readFloat, &buffers.normalCoords, float3Stride | 2 );

    const long        float2Stride = makeStride( 2 );
    const char* const names[]{ "u", "s", "texture_u", "texture_s" };
    const char* const names2[]{ "v", "t", "texture_v", "texture_t" };
    long              numUVs{};
    for( int i = 0; i < numOf( names ); ++i )
    {
        numUVs = ply_set_read_cb( ply, "vertex", names[i], readFloat, &buffers.uvCoords, float2Stride | 0 );
        if( numUVs > 0 )
        {
            ply_set_read_cb( ply, "vertex", names2[i], readFloat, &buffers.uvCoords, float2Stride | 1 );
            break;
        }
    }

    if( numVertices != m_meshInfo.numVertices || numFaces != m_meshInfo.numTriangles
        || numNormals != m_meshInfo.numNormals || numUVs != m_meshInfo.numTextureCoordinates )
    {
        throw std::runtime_error(
            m_filename + ": Data count mismatch: expected " + std::to_string( m_meshInfo.numVertices )
            + " vertices, got " + std::to_string( numVertices ) + "; expected "
            + std::to_string( m_meshInfo.numTriangles ) + " triangles, got " + std::to_string( numFaces ) + "; expected "
            + std::to_string( m_meshInfo.numNormals ) + " normals, got " + std::to_string( numNormals ) + "; expected "
            + std::to_string( m_meshInfo.numTextureCoordinates ) + " texture coordinates, got " + std::to_string( numUVs ) );
    }

    if( ply_read( ply ) == PLY_ERROR )
    {
        throw std::runtime_error( "Error reading PLY file " + m_filename );
    }

    std::cout << "Loaded " << m_filename << '\n';
}

}  // namespace

void InfoReader::s_errorCallback( p_ply ply, const char* message )
{
    throw std::runtime_error( message );
}

otk::pbrt::MeshInfo InfoReader::read( const std::string& filename )
{
    PlyHandle ply( ply_open( filename.c_str(), InfoReader::s_errorCallback, 0, nullptr ) );
    if( ply == nullptr )
    {
        throw std::runtime_error( filename + " is not a valid PLY file." );
    }

    if( ply_read_header( ply ) == PLY_ERROR )
    {
        throw std::runtime_error( "Could not read the PLY header of " + filename );
    }
    m_meshInfo.numVertices = ply_set_read_cb( ply, "vertex", "x", s_readVertex, this, 0 );
    ply_set_read_cb( ply, "vertex", "y", s_readVertex, this, 1 );
    ply_set_read_cb( ply, "vertex", "z", s_readVertex, this, 2 );
    m_meshInfo.numNormals = ply_set_read_cb( ply, "vertex", "nx", nullptr, nullptr, 3 );
    // variant texture coordinate names: (u,v), (s,t), (texture_u,texture_v), (texture_s,texture_t)
    const char* const names[]{ "u", "s", "texture_u", "texture_s" };
    for( const char* name : names )
    {
        m_meshInfo.numTextureCoordinates = ply_set_read_cb( ply, "vertex", name, nullptr, nullptr, 2 );
        if( m_meshInfo.numTextureCoordinates > 0 )
        {
            break;
        }
    }
    m_meshInfo.numTriangles = ply_set_read_cb( ply, "face", "vertex_indices", nullptr, nullptr, 0 );

    std::fill( std::begin( m_firstVertex ), std::end( m_firstVertex ), true );
    if( ply_read( ply ) == PLY_ERROR )
    {
        throw std::runtime_error( "Couldn't read PLY data from " + filename );
    }

    return m_meshInfo;
}

otk::pbrt::MeshLoaderPtr InfoReader::getLoader( const std::string& filename )
{
    return std::make_shared<MeshLoader>( filename, m_meshInfo );
}

int InfoReader::s_readVertex( p_ply_argument argument )
{
    void* self{};
    long  index{};
    if( ply_get_argument_user_data( argument, &self, &index ) == PLY_ERROR )
    {
        throw std::runtime_error( "ply_get_argument_user_data" );
    }
    return static_cast<InfoReader*>( self )->readVertex( argument, index );
}

int InfoReader::readVertex( p_ply_argument argument, long index )
{
    const float value = static_cast<float>( ply_get_argument_value( argument ) );
    if( m_firstVertex[index] )
    {
        m_meshInfo.minCoord[index] = value;
        m_meshInfo.maxCoord[index] = value;
        m_firstVertex[index]       = false;
    }
    else
    {
        m_meshInfo.minCoord[index] = std::min( m_meshInfo.minCoord[index], value );
        m_meshInfo.maxCoord[index] = std::max( m_meshInfo.maxCoord[index], value );
    }

    return 1;
}

}  // namespace ply
