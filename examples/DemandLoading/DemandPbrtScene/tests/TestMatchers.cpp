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

// gtest has to come before pbrt stuff
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Matchers.h"

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/SyncVector.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace testing;
using namespace otk::testing;

TEST( TestPointCompare, equal )
{
    ::pbrt::Point3f pbrtPoint{ 1.0f, 2.0f, 3.0f };
    float3          cudaPoint{ make_float3( 1.0f, 2.0f, 3.0f ) };

    ASSERT_EQ( pbrtPoint, cudaPoint );
    ASSERT_EQ( cudaPoint, pbrtPoint );
}

TEST( TestPointCompare, notEqual )
{
    ::pbrt::Point3f pbrtPoint{ 1.0f, 2.0f, 3.0f };
    float3          cudaPoint{ make_float3( 4.0f, 5.0f, 6.0f ) };

    ASSERT_NE( pbrtPoint, cudaPoint );
    ASSERT_NE( cudaPoint, pbrtPoint );
}

class TestHasDeviceVerticesMatcher : public Test
{
  public:
    ~TestHasDeviceVerticesMatcher() override = default;

  protected:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
        std::copy( std::begin( m_expectedCoords ), std::end( m_expectedCoords ), std::back_inserter( m_vertices ) );
        m_vertices.copyToDevice();
        m_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_input.triangleArray.numVertices   = m_expectedCoords.size() / 3;
        m_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        m_vertexBuffers[0]                  = m_vertices;
        m_input.triangleArray.vertexBuffers = m_vertexBuffers;
    }

    otk::SyncVector<float> m_vertices;
    CUdeviceptr            m_vertexBuffers[1]{};
    std::vector<float>     m_expectedCoords{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f };
    OptixBuildInput        m_input{};
};

TEST_F( TestHasDeviceVerticesMatcher, notTriangleBuildInput )
{
    m_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceVertexCoords( m_expectedCoords ) ) ) );
}

TEST_F( TestHasDeviceVerticesMatcher, differentNumVertices )
{
    m_input.triangleArray.numVertices = 1;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceVertexCoords( m_expectedCoords ) ) ) );
}

TEST_F( TestHasDeviceVerticesMatcher, differentVertexFormat )
{
    m_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_HALF3;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceVertexCoords( m_expectedCoords ) ) ) );
}

TEST_F( TestHasDeviceVerticesMatcher, nullVertexBuffer )
{
    CUdeviceptr vertexBuffers[1]{ CUdeviceptr{} };
    m_input.triangleArray.vertexBuffers = vertexBuffers;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceVertexCoords( m_expectedCoords ) ) ) );
}

TEST_F( TestHasDeviceVerticesMatcher, nonZeroStride )
{
    m_input.triangleArray.vertexStrideInBytes = sizeof( float3 );

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceVertexCoords( m_expectedCoords ) ) ) );
}

TEST_F( TestHasDeviceVerticesMatcher, differentVertexValues )
{
    std::vector<float> differentVertices{ m_expectedCoords };
    differentVertices[0] = 99.0f;
    differentVertices[1] = 999.0f;
    differentVertices[2] = 9999.0f;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceVertexCoords( differentVertices ) ) ) );
}

TEST_F( TestHasDeviceVerticesMatcher, matchesExpectedVertices )
{
    EXPECT_THAT( &m_input, hasTriangleBuildInput( 0, hasDeviceVertexCoords( m_expectedCoords ) ) );
}

class TestHasDeviceIndicesMatcher : public Test
{
  public:
    ~TestHasDeviceIndicesMatcher() override = default;

  protected:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
        std::copy( std::begin( m_expectedIndices ), std::end( m_expectedIndices ), std::back_inserter( m_indices ) );
        m_indices.copyToDevice();
        m_input.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_input.triangleArray.numIndexTriplets = m_expectedIndices.size() / 3;
        m_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        m_input.triangleArray.indexBuffer      = m_indices;
    }

    otk::SyncVector<unsigned int> m_indices;
    std::vector<unsigned int>     m_expectedIndices{ 0, 1, 2 };
    OptixBuildInput               m_input{};
};

TEST_F( TestHasDeviceIndicesMatcher, notTriangleBuildInput )
{
    m_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceIndices( m_expectedIndices ) ) ) );
}

TEST_F( TestHasDeviceIndicesMatcher, differentNumTriplets )
{
    m_input.triangleArray.numIndexTriplets = 2;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceIndices( m_expectedIndices ) ) ) );
}

TEST_F( TestHasDeviceIndicesMatcher, differentIndexFormat )
{
    m_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceIndices( m_expectedIndices ) ) ) );
}

TEST_F( TestHasDeviceIndicesMatcher, nullIndexBuffer )
{
    m_input.triangleArray.indexBuffer = CUdeviceptr{};

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceIndices( m_expectedIndices ) ) ) );
}

TEST_F( TestHasDeviceIndicesMatcher, nonZeroStride )
{
    m_input.triangleArray.indexStrideInBytes = sizeof( ushort3 );

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceIndices( m_expectedIndices ) ) ) );
}

TEST_F( TestHasDeviceIndicesMatcher, differentIndexValues )
{
    std::vector<unsigned int> differentIndices{ m_expectedIndices };
    differentIndices[0] = 999;

    EXPECT_THAT( &m_input, Not( hasTriangleBuildInput( 0, hasDeviceIndices( differentIndices ) ) ) );
}

TEST_F( TestHasDeviceIndicesMatcher, matchesExpectedIndices )
{
    EXPECT_THAT( &m_input, hasTriangleBuildInput( 0, hasDeviceIndices( m_expectedIndices ) ) );
}
