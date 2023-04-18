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

#include <OptiXToolkit/Util/Builders.h>

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
