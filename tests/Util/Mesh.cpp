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

#include "Mesh.h"
#include <cstring>

cudaError_t Mesh::create( float3 vertMin, float3 vertMax, float2 uvMin, float2 uvMax, uint2 gridRes, unsigned numTextures, Format format )
{
    std::vector<float2>   texCoords;
    std::vector<float3>   vertices;
    std::vector<uint3>    indices;
    std::vector<unsigned> textures;

    float3 vertExtent = vertMax - vertMin;
    float2 uvExtent   = uvMax - uvMin;

    for( uint32_t y = 0; y <= gridRes.y; y++ )
        for( uint32_t x = 0; x <= gridRes.x; x++ )
        {
            vertices.push_back( { vertMin.x + vertExtent.x * x / (float)( gridRes.x ), vertMin.y + vertExtent.y * y / (float)( gridRes.y ), vertMin.z } );
            texCoords.push_back( { uvMin.x + uvExtent.x * x / (float)( gridRes.x ), uvMin.y + uvExtent.y * y / (float)( gridRes.y ) } );
        }

    for( uint32_t y = 0; y < gridRes.y; y++ )
        for( uint32_t x = 0; x < gridRes.x; x++ )
        {
            indices.push_back( { y * ( gridRes.x + 1 ) + x, ( y + 1 ) * ( gridRes.x + 1 ) + x, ( y + 1 ) * ( gridRes.x + 1 ) + ( x + 1 ) } );
            indices.push_back( { y * ( gridRes.x + 1 ) + x, ( y + 1 ) * ( gridRes.x + 1 ) + ( x + 1 ), y * ( gridRes.x + 1 ) + ( x + 1 ) } );

            if( numTextures > 1 )
            {
                textures.push_back( ( x + y ) % numTextures );
                textures.push_back( ( x + y ) % numTextures );
            }
        }

    if( format.indexFormat == cuOmmBaking::IndexFormat::NONE )
    {
        std::vector<float3> flatVertices;
        std::vector<float2> flatTexCoords;

        for( const auto& idx3 : indices )
        {
            flatVertices.push_back( vertices[idx3.x] );
            flatVertices.push_back( vertices[idx3.y] );
            flatVertices.push_back( vertices[idx3.z] );

            flatTexCoords.push_back( texCoords[idx3.x] );
            flatTexCoords.push_back( texCoords[idx3.y] );
            flatTexCoords.push_back( texCoords[idx3.z] );
        }

        std::swap( flatVertices, vertices );
        std::swap( flatTexCoords, texCoords );
        indices.clear();
    }

    OMM_CUDA_CHECK( create( texCoords, vertices, indices, textures, format ) );

    return cudaSuccess;
}

cudaError_t Mesh::create( const std::vector<float2>& texCoords, const std::vector<float3>& vertices, const std::vector<uint3>& indices, const std::vector<unsigned> textures, Format format )
{
    destroy();

    if( vertices.empty() )
        format.verticesStrideInBytes = 0;
    if( texCoords.empty() )
        format.texCoordStrideInBytes = 0;
    if( indices.empty() )
    {
        format.indicesStrideInBytes = 0;
        format.indexFormat = cuOmmBaking::IndexFormat::NONE;
    }
    if( textures.empty() )
    {
        format.textureIndicesStrideInBytes = 0;
        format.textureIndexFormat = cuOmmBaking::IndexFormat::NONE;
    }

    uint32_t texCoordStrideInBytes = format.texCoordStrideInBytes;
    uint32_t verticesStrideInBytes = format.verticesStrideInBytes;
    uint32_t indicesStrideInBytes  = format.indicesStrideInBytes;
    uint32_t textureIndicesStrideInBytes = format.textureIndicesStrideInBytes;

    if( texCoordStrideInBytes == 0 )
        texCoordStrideInBytes = sizeof( float2 );
    if( verticesStrideInBytes == 0 )
        verticesStrideInBytes = sizeof( float3 );
    if( indicesStrideInBytes == 0 )
    {
        switch( format.indexFormat )
        {
        case cuOmmBaking::IndexFormat::I32_UINT: 
            indicesStrideInBytes = sizeof( uint3 );
            break;
        case cuOmmBaking::IndexFormat::I16_UINT:
            indicesStrideInBytes = sizeof( ushort3 );
            break;
        case cuOmmBaking::IndexFormat::NONE:
            indicesStrideInBytes = 0;
            break;
        default:
            return cudaErrorInvalidValue;
        };
    }
    if( textureIndicesStrideInBytes == 0 )
    {
        switch( format.textureIndexFormat )
        {
        case cuOmmBaking::IndexFormat::I32_UINT:
            textureIndicesStrideInBytes = sizeof( uint32_t );
            break;
        case cuOmmBaking::IndexFormat::I16_UINT:
            textureIndicesStrideInBytes = sizeof( uint16_t );
            break;
        case cuOmmBaking::IndexFormat::I8_UINT:
            textureIndicesStrideInBytes = sizeof( uint8_t );
            break;
        case cuOmmBaking::IndexFormat::NONE:
            textureIndicesStrideInBytes = 0;
            break;
        default:
            return cudaErrorInvalidValue;
        };
    }

    std::vector<char> raw_vertices( verticesStrideInBytes * vertices.size() );
    std::vector<char> raw_texCoords( texCoordStrideInBytes * texCoords.size() );
    std::vector<char> raw_indices( indicesStrideInBytes * indices.size() );
    std::vector<char> raw_textures( textureIndicesStrideInBytes * textures.size() );
    for( size_t i = 0; i < vertices.size(); ++i )
        memcpy( raw_vertices.data() + verticesStrideInBytes * i, &vertices[i], sizeof( float3 ) );
    for( size_t i = 0; i < texCoords.size(); ++i )
        memcpy( raw_texCoords.data() + texCoordStrideInBytes * i, &texCoords[i], sizeof( float2 ) );
    for( size_t i = 0; i < indices.size(); ++i )
    {
        uint3 idx32 = indices[i];
        switch( format.indexFormat )
        {
            case cuOmmBaking::IndexFormat::I16_UINT: {
                ushort3 idx16 = { (uint16_t)idx32.x, (uint16_t)idx32.y, (uint16_t)idx32.z };
                memcpy( raw_indices.data() + indicesStrideInBytes * i, &idx16, sizeof( ushort3 ) );
            }
            break;
            case cuOmmBaking::IndexFormat::I32_UINT : {
                memcpy( raw_indices.data() + indicesStrideInBytes * i, &idx32, sizeof( uint3 ) );
            }
            break;
            default:
                return cudaErrorInvalidValue;
        }
    }

    for( size_t i = 0; i < textures.size(); ++i )
    {
        uint32_t idx32 = textures[i];
        switch( format.textureIndexFormat )
        {
        case cuOmmBaking::IndexFormat::I8_UINT:
        {
            uint8_t idx8 = ( uint8_t )idx32;
            memcpy( raw_textures.data() + textureIndicesStrideInBytes * i, &idx8, sizeof( uint8_t ) );
        }
        break;
        case cuOmmBaking::IndexFormat::I16_UINT:
        {
            uint16_t idx16 = ( uint16_t )idx32;
            memcpy( raw_textures.data() + textureIndicesStrideInBytes * i, &idx16, sizeof( uint16_t ) );
        }
        break;
        case cuOmmBaking::IndexFormat::I32_UINT:
        {
            memcpy( raw_textures.data() + textureIndicesStrideInBytes * i, &idx32, sizeof( uint32_t ) );
        }
        break;
        default:
            return cudaErrorInvalidValue;
        }
    }

    OMM_CUDA_CHECK( m_vertices.allocAndUpload( raw_vertices ) );
    OMM_CUDA_CHECK( m_texCoords.allocAndUpload( raw_texCoords ) );
    OMM_CUDA_CHECK( m_indices.allocAndUpload( raw_indices ) );
    OMM_CUDA_CHECK( m_textures.allocAndUpload( raw_textures ) );

    m_numVertices  = (uint32_t)vertices.size();
    m_numTexCoords = (uint32_t)texCoords.size();
    m_numIndices   = (uint32_t)indices.size();

    m_format = format;

    return cudaSuccess;
}

cuOmmBaking::BakeInputDesc Mesh::getBakingInputDesc() const
{
    cuOmmBaking::BakeInputDesc desc = {};
    
    desc.texCoordBuffer = m_texCoords.get();
    desc.texCoordFormat = cuOmmBaking::TexCoordFormat::UV32_FLOAT2;
    desc.texCoordStrideInBytes = m_format.texCoordStrideInBytes;
    desc.numTexCoords = m_numTexCoords;

    desc.indexBuffer        = m_indices.get();
    desc.indexTripletStrideInBytes = m_format.indicesStrideInBytes;
    desc.numIndexTriplets   = m_numIndices;
    desc.indexFormat        = m_format.indexFormat;

    desc.textureIndexBuffer        = m_textures.get();
    desc.textureIndexFormat        = m_format.textureIndexFormat;
    desc.textureIndexStrideInBytes = m_format.textureIndicesStrideInBytes;

    return desc;
}

