// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>

#include <optix.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

using namespace testing;
using namespace otk::testing;

namespace {

template <typename T>
class TestOptixStructMatcher : public Test
{
  protected:
    T m_data{};
};

using TestOptixBuildInput = TestOptixStructMatcher<OptixBuildInput>;

class TestInstanceBuildInput : public TestOptixBuildInput
{
  protected:
    void SetUp() override
    {
        TestOptixBuildInput::SetUp();
        m_data.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    }
};

}  // namespace

TEST_F( TestInstanceBuildInput, differentBuiltType )
{
    m_data.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_data, Not( hasInstanceBuildInput( 0, any<OptixBuildInputInstanceArray>() ) ) );
}

TEST_F( TestInstanceBuildInput, matchesBuildInput )
{
    EXPECT_THAT( &m_data, hasInstanceBuildInput( 0, any<OptixBuildInputInstanceArray>() ) );
}

TEST_F( TestInstanceBuildInput, hasNumInstances )
{
    m_data.instanceArray.numInstances = 3;
    EXPECT_THAT( &m_data, hasInstanceBuildInput( 0, hasNumInstances( 3 ) ) );
}

namespace {

class TestInstanceBuildInputDeviceInstances : public TestInstanceBuildInput
{
  protected:
    void SetUp() override
    {
        TestInstanceBuildInput::SetUp();
        cudaFree( nullptr );
        cudaMalloc( &m_devInstances, m_sizeInBytes );
        copyInstancesToDevice();
        m_data.instanceArray.numInstances = 1;
        m_data.instanceArray.instances    = reinterpret_cast<CUdeviceptr>( m_devInstances );
    }

    void TearDown() override { cudaFree( m_devInstances ); }

    void copyInstancesToDevice()
    {
        cudaMemcpy( m_devInstances, m_instances.data(), m_sizeInBytes, cudaMemcpyHostToDevice );
    }

    std::vector<OptixInstance> m_instances{ 1 };
    size_t                     m_sizeInBytes{ m_instances.size() * sizeof( OptixInstance ) };
    OptixInstance*             m_devInstances{};
};

}  // namespace

TEST_F( TestInstanceBuildInputDeviceInstances, hasDeviceInstanceId )
{
    m_instances[0].instanceId = 234;
    copyInstancesToDevice();

    EXPECT_THAT( &m_data, hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1 ),
                                                            hasDeviceInstances( hasInstance( 0, hasInstanceId( 234 ) ) ) ) ) );
}

class TestHasSbtFlags : public TestOptixBuildInput
{
  public:
    ~TestHasSbtFlags() override = default;

  protected:
    void SetUp() override
    {
        TestOptixBuildInput::SetUp();
        m_data.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_data.triangleArray.numSbtRecords = static_cast<unsigned int>( m_expectedFlags.size() );
        m_flags                            = m_expectedFlags;
        m_data.triangleArray.flags         = m_flags.data();
    }

    std::vector<unsigned int> m_flags;
    std::vector<unsigned int> m_expectedFlags{ OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
                                               OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
};

TEST_F( TestHasSbtFlags, notTriangleBuildInput )
{
    m_data.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasSbtFlags( m_expectedFlags ) ) ) );
}

TEST_F( TestHasSbtFlags, differentNumSbtRecords )
{
    m_data.triangleArray.numSbtRecords = 1;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasSbtFlags( m_expectedFlags ) ) ) );
}

TEST_F( TestHasSbtFlags, differentFlags )
{
    m_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasSbtFlags( m_expectedFlags ) ) ) );
}

TEST_F( TestHasSbtFlags, differentFlagsOtherIndex )
{
    m_flags[1] = OPTIX_GEOMETRY_FLAG_NONE;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasSbtFlags( m_expectedFlags ) ) ) );
}

TEST_F( TestHasSbtFlags, matchesExpectedFlags )
{
    EXPECT_THAT( &m_data, hasTriangleBuildInput( 0, hasSbtFlags( m_expectedFlags ) ) );
}

class TestHasNoPreTransform : public TestOptixBuildInput
{
  public:
    ~TestHasNoPreTransform() override = default;

  protected:
    void SetUp() override
    {
        TestOptixBuildInput::SetUp();
        m_data.type                          = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_data.triangleArray.preTransform    = CUdeviceptr{};
        m_data.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
    }
};

TEST_F( TestHasNoPreTransform, notTriangleBuildInput )
{
    m_data.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoPreTransform() ) ) );
}

TEST_F( TestHasNoPreTransform, differentTransformFormat )
{
    m_data.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoPreTransform() ) ) );
}

TEST_F( TestHasNoPreTransform, differentPreTransform )
{
    m_data.triangleArray.preTransform = CUdeviceptr{ 0xbeefbeefU };

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoPreTransform() ) ) );
}

TEST_F( TestHasNoPreTransform, matchesNoTransform )
{
    EXPECT_THAT( &m_data, hasTriangleBuildInput( 0, hasNoPreTransform() ) );
}

class TestHasNoSbtIndexOffsets : public TestOptixBuildInput
{
  public:
    ~TestHasNoSbtIndexOffsets() override = default;

  protected:
    void SetUp() override
    {
        TestOptixBuildInput::SetUp();
        m_data.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    }
};

TEST_F( TestHasNoSbtIndexOffsets, notTriangleBuildInput )
{
    m_data.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoSbtIndexOffsets() ) ) );
}

TEST_F( TestHasNoSbtIndexOffsets, nonNullSbtIndexOffsetBuffer )
{
    m_data.triangleArray.sbtIndexOffsetBuffer = CUdeviceptr{ 0xf00dbeef };

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoSbtIndexOffsets() ) ) );
}

TEST_F( TestHasNoSbtIndexOffsets, nonZeroSbtIndexOffsetSize )
{
    m_data.triangleArray.sbtIndexOffsetSizeInBytes = sizeof( int );

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoSbtIndexOffsets() ) ) );
}

TEST_F( TestHasNoSbtIndexOffsets, nonZeroSbtIndexOffsetStride )
{
    m_data.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( int );

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoSbtIndexOffsets() ) ) );
}

TEST_F( TestHasNoSbtIndexOffsets, matchesExpectedOffsets )
{
    EXPECT_THAT( &m_data, hasTriangleBuildInput( 0, hasNoSbtIndexOffsets() ) );
}

class TestHasNoPrimitiveIndexOffset : public TestOptixBuildInput
{
  public:
    ~TestHasNoPrimitiveIndexOffset() override = default;

  protected:
    void SetUp() override
    {
        TestOptixBuildInput::SetUp();
        m_data.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    }
};

TEST_F( TestHasNoPrimitiveIndexOffset, notTriangleBuildInput )
{
    m_data.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoPrimitiveIndexOffset() ) ) );
}

TEST_F( TestHasNoPrimitiveIndexOffset, nonZeroOffset )
{
    m_data.triangleArray.primitiveIndexOffset = 1U;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoPrimitiveIndexOffset() ) ) );
}

TEST_F( TestHasNoPrimitiveIndexOffset, noPrimitiveIndexOffsets )
{
    EXPECT_THAT( &m_data, hasTriangleBuildInput( 0, hasNoPrimitiveIndexOffset() ) );
}

class TestHasNoOpacityMap : public TestOptixBuildInput
{
  public:
    ~TestHasNoOpacityMap() override = default;

  protected:
    void SetUp() override
    {
        TestOptixBuildInput::SetUp();
        m_data.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    }
};

TEST_F( TestHasNoOpacityMap, notTriangleBuildInput )
{
    m_data.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

#if OPTIX_VERSION >= 70600
TEST_F( TestHasNoOpacityMap, differentIndexingMode )
{
    m_data.triangleArray.opacityMicromap.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

TEST_F( TestHasNoOpacityMap, nonZeroMicromapArray )
{
    m_data.triangleArray.opacityMicromap.opacityMicromapArray = CUdeviceptr{ 0xbeefbeef };

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

TEST_F( TestHasNoOpacityMap, nonZeroIndexBuffer )
{
    m_data.triangleArray.opacityMicromap.indexBuffer = CUdeviceptr{ 0xbeefbeef };

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

TEST_F( TestHasNoOpacityMap, nonZeroIndexSize )
{
    m_data.triangleArray.opacityMicromap.indexSizeInBytes = sizeof( int );

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

TEST_F( TestHasNoOpacityMap, nonZeroIndexStride )
{
    m_data.triangleArray.opacityMicromap.indexStrideInBytes = sizeof( int );

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

TEST_F( TestHasNoOpacityMap, nonZeroIndexOffset )
{
    m_data.triangleArray.opacityMicromap.indexOffset = 1U;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

TEST_F( TestHasNoOpacityMap, nonZeroUsageCount )
{
    m_data.triangleArray.opacityMicromap.numMicromapUsageCounts = 1U;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}

TEST_F( TestHasNoOpacityMap, nonNullUsageCountList )
{
    OptixOpacityMicromapUsageCount count{};
    m_data.triangleArray.opacityMicromap.micromapUsageCounts = &count;

    EXPECT_THAT( &m_data, Not( hasTriangleBuildInput( 0, hasNoOpacityMap() ) ) );
}
#endif

class TestHasNumCustomPrimitives : public TestOptixBuildInput
{
  public:
    ~TestHasNumCustomPrimitives() override = default;

  protected:
    void SetUp() override
    {
        TestOptixBuildInput::SetUp();
        m_data.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    }
};

TEST_F( TestHasNumCustomPrimitives, notTriangleBuildInput )
{
    m_data.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    EXPECT_THAT( &m_data, Not( hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) ) );
}

TEST_F( TestHasNumCustomPrimitives, differentNumPrimitives )
{
    m_data.customPrimitiveArray.numPrimitives = 2U;

    EXPECT_THAT( &m_data, Not( hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) ) );
}

TEST_F( TestHasNumCustomPrimitives, hasNumPrimitives )
{
    m_data.customPrimitiveArray.numPrimitives = 1U;

    EXPECT_THAT( &m_data, hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) );
}

class TestOptixAccelBuildOptionsMatchers : public TestOptixStructMatcher<OptixAccelBuildOptions>
{
};

TEST_F( TestOptixAccelBuildOptionsMatchers, notBuildOperation )
{
    m_data.operation = OPTIX_BUILD_OPERATION_UPDATE;

    EXPECT_THAT( &m_data, Not( isBuildOperation() ) );
}

TEST_F( TestOptixAccelBuildOptionsMatchers, isBuildOperation )
{
    m_data.operation = OPTIX_BUILD_OPERATION_BUILD;

    EXPECT_THAT( &m_data, isBuildOperation() );
}

TEST_F( TestOptixAccelBuildOptionsMatchers, allowsUpdate )
{
    m_data.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    EXPECT_THAT( &m_data, buildAllowsUpdate() );
}

TEST_F( TestOptixAccelBuildOptionsMatchers, doesNotAllowUpdate )
{
    EXPECT_THAT( &m_data, Not( buildAllowsUpdate() ) );
}

TEST_F( TestOptixAccelBuildOptionsMatchers, allowsRandomVertexAccess )
{
    m_data.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

    EXPECT_THAT( &m_data, buildAllowsRandomVertexAccess() );
}

TEST_F( TestOptixAccelBuildOptionsMatchers, doesNotAllowRandomVertexAccess )
{
    EXPECT_THAT( &m_data, Not( buildAllowsRandomVertexAccess() ) );
}

class TestOptixPipelineCompileOptionMatchers : public TestOptixStructMatcher<OptixPipelineCompileOptions>
{
};

TEST_F( TestOptixPipelineCompileOptionMatchers, usesMotionBlur )
{
    m_data.usesMotionBlur = 1;

    EXPECT_THAT( &m_data, usesMotionBlur() );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, doesNotUseMotionBlur )
{
    EXPECT_THAT( &m_data, Not( usesMotionBlur() ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, allowAnyTraversableGraph )
{
    m_data.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

    EXPECT_THAT( &m_data, allowAnyTraversableGraph() );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, doesNotAllowAnyTraversableGraph )
{
    m_data.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    EXPECT_NE( 0, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS );
    EXPECT_THAT( &m_data, Not( allowAnyTraversableGraph() ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, hasExpectedPayloadValueCount )
{
    m_data.numPayloadValues = 1;

    EXPECT_THAT( &m_data, hasPayloadValueCount( 1 ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, hasIncorrectPayloadValueCount )
{
    EXPECT_THAT( &m_data, Not( hasAttributeValueCount( 1 ) ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, hasExpectedAttributeValueCount )
{
    m_data.numAttributeValues = 1;

    EXPECT_THAT( &m_data, hasAttributeValueCount( 1 ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, hasIncorrectAttributeValueCount )
{
    EXPECT_THAT( &m_data, Not( hasAttributeValueCount( 1 ) ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, hasExceptionFlags )
{
    m_data.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

    EXPECT_NE( 0U, OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW );
    EXPECT_THAT( &m_data, hasExceptionFlags( OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, doesNotHaveExceptionFlags )
{
    EXPECT_THAT( &m_data, Not( hasExceptionFlags( OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW ) ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, hasParamsName )
{
    m_data.pipelineLaunchParamsVariableName = "foo";

    EXPECT_THAT( &m_data, hasParamsName( "foo" ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, doesNotHaveParamName )
{
    const std::string name{ "foo" };

    EXPECT_THAT( &m_data, Not( hasParamsName( name ) ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, hasPrimitiveTypes )
{
    m_data.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_CUSTOM | OPTIX_PRIMITIVE_TYPE_TRIANGLE;

    EXPECT_THAT( &m_data, hasPrimitiveTypes( OPTIX_PRIMITIVE_TYPE_CUSTOM | OPTIX_PRIMITIVE_TYPE_TRIANGLE ) );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, doesNotHavePrimitiveTypes )
{
    m_data.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_CUSTOM;

    EXPECT_THAT( &m_data, Not( hasPrimitiveTypes( OPTIX_PRIMITIVE_TYPE_CUSTOM | OPTIX_PRIMITIVE_TYPE_TRIANGLE ) ) );
}

#if OPTIX_VERSION >= 70600
TEST_F( TestOptixPipelineCompileOptionMatchers, allowsOpacityMicromaps )
{
    m_data.allowOpacityMicromaps = 1;

    EXPECT_THAT( &m_data, allowsOpacityMicromaps() );
}

TEST_F( TestOptixPipelineCompileOptionMatchers, doesNotAllowOpacityMicromaps )
{
    EXPECT_THAT( &m_data, Not( allowsOpacityMicromaps() ) );
}
#endif

class TestOptixPipelineLinkOptionMatchers : public TestOptixStructMatcher<OptixPipelineLinkOptions>
{
};

TEST_F( TestOptixPipelineLinkOptionMatchers, hasExpectedMaxTraceDepth )
{
    m_data.maxTraceDepth = 2;

    EXPECT_THAT( &m_data, hasMaxTraceDepth( 2U ) );
}

TEST_F( TestOptixPipelineLinkOptionMatchers, doesNotHaveExpectedMaxTraceDepth )
{
    m_data.maxTraceDepth = 1;

    EXPECT_THAT( &m_data, Not( hasMaxTraceDepth( 2U ) ) );
}
