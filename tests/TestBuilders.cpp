//
//  Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <OptiXToolkit/OptiXMemory/Builders.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

namespace {

const unsigned int NUM_BUILD_INPUTS = 1;

class TestBuildInputBuilder : public testing::Test
{
  protected:
    OptixBuildInput        buildInputs[NUM_BUILD_INPUTS]{};
    otk::BuildInputBuilder builder{ buildInputs };
};

}  // namespace

TEST_F( TestBuildInputBuilder, ZeroInstancesSetsInstanceArrayToZero )
{
    const CUdeviceptr FAKE_INSTANCES = 0xdeadbeefU;

    builder.instanceArray( FAKE_INSTANCES, 0 );

    EXPECT_EQ( OPTIX_BUILD_INPUT_TYPE_INSTANCES, buildInputs[0].type );
    EXPECT_EQ( 0U, buildInputs[0].instanceArray.numInstances );
    EXPECT_EQ( 0U, buildInputs[0].instanceArray.instances );
}

#if OPTIX_VERSION >= 70600
inline bool operator==( const OptixBuildInputOpacityMicromap& lhs, const OptixBuildInputOpacityMicromap& rhs )
{
    return lhs.indexingMode == rhs.indexingMode && lhs.opacityMicromapArray == rhs.opacityMicromapArray
           && lhs.indexBuffer == rhs.indexBuffer && lhs.indexSizeInBytes == rhs.indexSizeInBytes
           && lhs.indexStrideInBytes == rhs.indexStrideInBytes && lhs.indexOffset == rhs.indexOffset
           && lhs.numMicromapUsageCounts == rhs.numMicromapUsageCounts && lhs.micromapUsageCounts == rhs.micromapUsageCounts;
}
#endif

TEST_F( TestBuildInputBuilder, Triangles )
{
    const unsigned int NUM_VERTICES{ 6 };
    const unsigned int NUM_MOTION_STEPS{ 1 };
    const CUdeviceptr  FAKE_VERTICES{ 0xdeadbeefU };
    const CUdeviceptr  FAKE_INDICES{ 0xbadf00dU };
    const CUdeviceptr  devVertexBuffers[NUM_MOTION_STEPS]{ FAKE_VERTICES };
    const CUdeviceptr  devIndexBuffer{ FAKE_INDICES };
    const unsigned int NUM_TRIANGLES{ 0U };
    const unsigned int NUM_TRIANGLE_SBT_RECORDS{ 1U };
    // SBT flags
    std::array<uint32_t, NUM_TRIANGLE_SBT_RECORDS> flags;
    std::fill( flags.begin(), flags.end(), OPTIX_GEOMETRY_FLAG_NONE );

    builder.triangles( NUM_VERTICES, devVertexBuffers, OPTIX_VERTEX_FORMAT_FLOAT3, NUM_TRIANGLES, devIndexBuffer,
                       OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3, flags.data(), NUM_TRIANGLE_SBT_RECORDS );

    EXPECT_EQ( OPTIX_BUILD_INPUT_TYPE_TRIANGLES, buildInputs[0].type );
    OptixBuildInputTriangleArray& triangles = buildInputs[0].triangleArray;
    EXPECT_EQ( devVertexBuffers, triangles.vertexBuffers );
    EXPECT_EQ( NUM_VERTICES, triangles.numVertices );
    EXPECT_EQ( OPTIX_VERTEX_FORMAT_FLOAT3, triangles.vertexFormat );
    EXPECT_EQ( 0U, triangles.vertexStrideInBytes );
    EXPECT_EQ( devIndexBuffer, triangles.indexBuffer );
    EXPECT_EQ( NUM_TRIANGLES, triangles.numIndexTriplets );
    EXPECT_EQ( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3, triangles.indexFormat );
    EXPECT_EQ( 0U, triangles.indexStrideInBytes );
    EXPECT_EQ( CUdeviceptr{}, triangles.preTransform );
    EXPECT_EQ( flags.data(), triangles.flags );
    EXPECT_EQ( NUM_TRIANGLE_SBT_RECORDS, triangles.numSbtRecords );
    EXPECT_EQ( CUdeviceptr{}, triangles.sbtIndexOffsetBuffer );
    EXPECT_EQ( 0U, triangles.sbtIndexOffsetSizeInBytes );
    EXPECT_EQ( 0U, triangles.sbtIndexOffsetStrideInBytes );
    EXPECT_EQ( 0U, triangles.primitiveIndexOffset );
#if OPTIX_VERSION >= 70600
    EXPECT_EQ( OptixBuildInputOpacityMicromap{}, triangles.opacityMicromap );
#endif
}

#if OPTIX_VERSION >= 70500
TEST_F( TestBuildInputBuilder, ZeroSpheresSetsNullBufferPointers )
{
    const unsigned int                           NUM_MOTION_STEPS{ 1 };
    const CUdeviceptr                            FAKE_SPHERE_CENTERS{ 0xdeadbeefU };
    const CUdeviceptr                            FAKE_SPHERE_RADII{ 0xbadf00dU };
    const CUdeviceptr                            devVertexBuffers[NUM_MOTION_STEPS]{ FAKE_SPHERE_CENTERS };
    const CUdeviceptr                            devRadiiBuffers[NUM_MOTION_STEPS]{ FAKE_SPHERE_RADII };
    const unsigned int                           NUM_SPHERES{ 0U };
    const unsigned int                           NUM_SPHERE_SBT_RECORDS{ 1U };
    std::array<uint32_t, NUM_SPHERE_SBT_RECORDS> flags;
    std::fill( flags.begin(), flags.end(), OPTIX_GEOMETRY_FLAG_NONE );

    builder.spheres( devVertexBuffers, NUM_SPHERES, devRadiiBuffers, flags.data(), NUM_SPHERE_SBT_RECORDS );

    EXPECT_EQ( OPTIX_BUILD_INPUT_TYPE_SPHERES, buildInputs[0].type );
    EXPECT_EQ( 0U, buildInputs[0].sphereArray.numVertices );
    EXPECT_EQ( nullptr, buildInputs[0].sphereArray.vertexBuffers );
    EXPECT_EQ( nullptr, buildInputs[0].sphereArray.radiusBuffers );
}

TEST_F( TestBuildInputBuilder, ZeroSpheresWithSbtIndicesSetsNullBufferPointers )
{
    const unsigned int                           NUM_MOTION_STEPS{ 1 };
    const CUdeviceptr                            FAKE_SPHERE_CENTERS{ 0xdeadbeefU };
    const CUdeviceptr                            FAKE_SPHERE_RADII{ 0xbadf00dU };
    const CUdeviceptr                            devVertexBuffers[NUM_MOTION_STEPS]{ FAKE_SPHERE_CENTERS };
    const CUdeviceptr                            devRadiiBuffers[NUM_MOTION_STEPS]{ FAKE_SPHERE_RADII };
    const unsigned int                           NUM_SPHERES{ 0U };
    const unsigned int                           NUM_SPHERE_SBT_RECORDS{ 1U };
    std::array<uint32_t, NUM_SPHERE_SBT_RECORDS> flags;
    std::fill( flags.begin(), flags.end(), OPTIX_GEOMETRY_FLAG_NONE );
    const CUdeviceptr FAKE_SBT_INDICES{ 0xdeadc000U };

    builder.spheres( devVertexBuffers, NUM_SPHERES, devRadiiBuffers, flags.data(), NUM_SPHERE_SBT_RECORDS,
                     reinterpret_cast<void*>( FAKE_SBT_INDICES ), sizeof( uint32_t ) );

    EXPECT_EQ( OPTIX_BUILD_INPUT_TYPE_SPHERES, buildInputs[0].type );
    EXPECT_EQ( 0U, buildInputs[0].sphereArray.numVertices );
    EXPECT_EQ( nullptr, buildInputs[0].sphereArray.vertexBuffers );
    EXPECT_EQ( nullptr, buildInputs[0].sphereArray.radiusBuffers );
}
#endif // OPTIX_VERSION >= 70500
