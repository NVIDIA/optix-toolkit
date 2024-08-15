// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/OptiXMemory/Builders.h>

#include <OptiXToolkit/Memory/BitCast.h>

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
#endif  // OPTIX_VERSION >= 70500

class TestProgramGroupDescBuilder : public ::testing::Test
{
  protected:
    OptixProgramGroupDesc m_desc[1]{};
    OptixModule           m_module{ otk::bit_cast<OptixModule>( 9999ULL ) };
    OptixModule           m_raygenModule{ otk::bit_cast<OptixModule>( 1111ULL ) };
    OptixModule           m_missModule{ otk::bit_cast<OptixModule>( 2222ULL ) };
    OptixModule           m_isModule{ otk::bit_cast<OptixModule>( 3333ULL ) };
    OptixModule           m_ahModule{ otk::bit_cast<OptixModule>( 4444ULL ) };
    OptixModule           m_chModule{ otk::bit_cast<OptixModule>( 5555ULL ) };
    const char*           m_raygenName{ "__raygen__test" };
    const char*           m_missName{ "__miss__test" };
    const char*           m_isName{ "__intersect__test" };
    const char*           m_ahName{ "__anyhit__test" };
    const char*           m_chName{ "__closesthit__test" };
};

TEST_F( TestProgramGroupDescBuilder, raygenAssumedModule )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).raygen( m_raygenName );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_RAYGEN, m_desc[0].kind );
    const OptixProgramGroupSingleModule& raygen = m_desc[0].raygen;
    EXPECT_EQ( m_module, raygen.module );
    EXPECT_STREQ( m_raygenName, raygen.entryFunctionName );
}

TEST_F( TestProgramGroupDescBuilder, raygenExplicitModule )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).raygen( { m_raygenModule, m_raygenName } );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_RAYGEN, m_desc[0].kind );
    const OptixProgramGroupSingleModule& raygen = m_desc[0].raygen;
    EXPECT_EQ( m_raygenModule, raygen.module );
    EXPECT_STREQ( m_raygenName, raygen.entryFunctionName );
}

TEST_F( TestProgramGroupDescBuilder, missAssumedModule )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).miss( m_missName );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_MISS, m_desc[0].kind );
    const OptixProgramGroupSingleModule& miss = m_desc[0].miss;
    EXPECT_EQ( m_module, miss.module );
    EXPECT_STREQ( m_missName, miss.entryFunctionName );
}

TEST_F( TestProgramGroupDescBuilder, missExplicitModule )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).miss( { m_missModule, m_missName } );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_MISS, m_desc[0].kind );
    const OptixProgramGroupSingleModule& miss = m_desc[0].miss;
    EXPECT_EQ( m_missModule, miss.module );
    EXPECT_STREQ( m_missName, miss.entryFunctionName );
}

TEST_F( TestProgramGroupDescBuilder, hitGroupCH )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).hitGroupCH( m_chModule, m_chName );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_desc[0].kind );
    const OptixProgramGroupHitgroup& hitGroup = m_desc[0].hitgroup;
    EXPECT_EQ( OptixModule{}, hitGroup.moduleIS );
    EXPECT_EQ( nullptr, hitGroup.entryFunctionNameIS );
    EXPECT_EQ( OptixModule{}, hitGroup.moduleAH );
    EXPECT_EQ( nullptr, hitGroup.entryFunctionNameAH );
    EXPECT_EQ( m_chModule, hitGroup.moduleCH );
    EXPECT_STREQ( m_chName, hitGroup.entryFunctionNameCH );
}

TEST_F( TestProgramGroupDescBuilder, hitGroupCHBundled )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).hitGroupCH( { m_chModule, m_chName } );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_desc[0].kind );
    const OptixProgramGroupHitgroup& hitGroup = m_desc[0].hitgroup;
    EXPECT_EQ( OptixModule{}, hitGroup.moduleIS );
    EXPECT_EQ( nullptr, hitGroup.entryFunctionNameIS );
    EXPECT_EQ( OptixModule{}, hitGroup.moduleAH );
    EXPECT_EQ( nullptr, hitGroup.entryFunctionNameAH );
    EXPECT_EQ( m_chModule, hitGroup.moduleCH );
    EXPECT_STREQ( m_chName, hitGroup.entryFunctionNameCH );
}

TEST_F( TestProgramGroupDescBuilder, hitGroupISCH )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).hitGroupISCH( m_isModule, m_isName, m_chModule, m_chName );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_desc[0].kind );
    const OptixProgramGroupHitgroup& hitGroup = m_desc[0].hitgroup;
    EXPECT_EQ( m_isModule, hitGroup.moduleIS );
    EXPECT_STREQ( m_isName, hitGroup.entryFunctionNameIS );
    EXPECT_EQ( OptixModule{}, hitGroup.moduleAH );
    EXPECT_EQ( nullptr, hitGroup.entryFunctionNameAH );
    EXPECT_EQ( m_chModule, hitGroup.moduleCH );
    EXPECT_STREQ( m_chName, hitGroup.entryFunctionNameCH );
}

TEST_F( TestProgramGroupDescBuilder, hitGroupISCHAssumedModule )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).hitGroupISCH( m_isName, m_chName );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_desc[0].kind );
    const OptixProgramGroupHitgroup& hitGroup = m_desc[0].hitgroup;
    EXPECT_EQ( m_module, hitGroup.moduleIS );
    EXPECT_STREQ( m_isName, hitGroup.entryFunctionNameIS );
    EXPECT_EQ( OptixModule{}, hitGroup.moduleAH );
    EXPECT_EQ( nullptr, hitGroup.entryFunctionNameAH );
    EXPECT_EQ( m_module, hitGroup.moduleCH );
    EXPECT_STREQ( m_chName, hitGroup.entryFunctionNameCH );
}

TEST_F( TestProgramGroupDescBuilder, hitGroupISCHBundled )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).hitGroupISCH( { m_isModule, m_isName }, { m_chModule, m_chName } );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_desc[0].kind );
    const OptixProgramGroupHitgroup& hitGroup = m_desc[0].hitgroup;
    EXPECT_EQ( m_isModule, hitGroup.moduleIS );
    EXPECT_STREQ( m_isName, hitGroup.entryFunctionNameIS );
    EXPECT_EQ( OptixModule{}, hitGroup.moduleAH );
    EXPECT_EQ( nullptr, hitGroup.entryFunctionNameAH );
    EXPECT_EQ( m_chModule, hitGroup.moduleCH );
    EXPECT_STREQ( m_chName, hitGroup.entryFunctionNameCH );
}

TEST_F( TestProgramGroupDescBuilder, hitGroupISAHCH )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).hitGroupISAHCH( m_isModule, m_isName, m_ahModule, m_ahName, m_chModule, m_chName );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_desc[0].kind );
    const OptixProgramGroupHitgroup& hitGroup = m_desc[0].hitgroup;
    EXPECT_EQ( m_isModule, hitGroup.moduleIS );
    EXPECT_STREQ( m_isName, hitGroup.entryFunctionNameIS );
    EXPECT_EQ( m_ahModule, hitGroup.moduleAH );
    EXPECT_STREQ( m_ahName, hitGroup.entryFunctionNameAH );
    EXPECT_EQ( m_chModule, hitGroup.moduleCH );
    EXPECT_STREQ( m_chName, hitGroup.entryFunctionNameCH );
}

TEST_F( TestProgramGroupDescBuilder, hitGroupISAHCHBundled )
{
    otk::ProgramGroupDescBuilder( m_desc, m_module ).hitGroupISAHCH( { m_isModule, m_isName }, { m_ahModule, m_ahName }, { m_chModule, m_chName } );

    EXPECT_EQ( OPTIX_PROGRAM_GROUP_KIND_HITGROUP, m_desc[0].kind );
    const OptixProgramGroupHitgroup& hitGroup = m_desc[0].hitgroup;
    EXPECT_EQ( m_isModule, hitGroup.moduleIS );
    EXPECT_STREQ( m_isName, hitGroup.entryFunctionNameIS );
    EXPECT_EQ( m_ahModule, hitGroup.moduleAH );
    EXPECT_STREQ( m_ahName, hitGroup.entryFunctionNameAH );
    EXPECT_EQ( m_chModule, hitGroup.moduleCH );
    EXPECT_STREQ( m_chName, hitGroup.entryFunctionNameCH );
}
