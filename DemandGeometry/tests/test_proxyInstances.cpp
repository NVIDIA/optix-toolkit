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

#include <OptiXToolkit/DemandGeometry/ProxyInstances.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockDemandLoader.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockOptix.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <optix_stubs.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>

using namespace testing;
using namespace demandLoading;
using namespace demandGeometry;
using namespace otk::testing;

namespace {

class TestProxyInstance : public Test
{
  protected:
    void SetUp() override;

    void configureAccelBuildInputs( OptixBuildInput* gasBuildInput, OptixBuildInput* iasBuildInput );
    void configureZeroInstanceIASAccelBuildInput( OptixBuildInput& iasBuildInput );

    MockDemandLoader m_loader;
    ProxyInstances   m_instances{ &m_loader };
    MockOptix        m_optix;

    // We need to use the real default stream because we're not mocking CUDA.
    CUstream m_stream{};

    uint_t    m_startPageId{ 1964 };
    OptixAabb m_proxyBounds{ -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };
    OptixAabb m_proxy2Bounds{ 10.f, 11.f, 12.f, 14.f, 13.f, 13.f };

    OptixDeviceContext     m_fakeDc{ reinterpret_cast<OptixDeviceContext>( 0xd00df00ddeadbeef ) };
    OptixTraversableHandle m_fakeIAS{ 0xdeadbeefULL };
    OptixTraversableHandle m_fakeGAS{ 0xbadf00dULL };
};

void TestProxyInstance::SetUp()
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    initMockOptix( m_optix );
}

void TestProxyInstance::configureAccelBuildInputs( OptixBuildInput* gasBuildInput, OptixBuildInput* iasBuildInput )
{
    const uint_t numBuildInputs{ 1 };
    auto         immutable = AllOf( NotNull(), isBuildOperation(), Not( buildAllowsUpdate() ) );
    auto         isGAS     = AllOf( NotNull(), isCustomPrimitiveBuildInput() );
    auto         isIAS     = AllOf( NotNull(), isInstanceBuildInput() );
    auto&        callMemUsageGAS =
        EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeDc, immutable, isGAS, numBuildInputs, NotNull() ) );
    auto& callMemUsageIAS =
        EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeDc, immutable, isIAS, numBuildInputs, NotNull() ) );
    auto savePtr = []( OptixBuildInput* ptr ) { return DoAll( SaveArgPointee<2>( ptr ), Return( OPTIX_SUCCESS ) ); };
    if( gasBuildInput != nullptr )
        callMemUsageGAS.WillOnce( savePtr( gasBuildInput ) );
    else
        callMemUsageGAS.WillOnce( Return( OPTIX_SUCCESS ) );
    if( iasBuildInput != nullptr )
        callMemUsageIAS.WillOnce( savePtr( iasBuildInput ) );
    else
        callMemUsageIAS.WillOnce( Return( OPTIX_SUCCESS ) );
    EXPECT_CALL( m_optix, accelBuild( m_fakeDc, m_stream, immutable, isGAS, numBuildInputs, _, _, _, _, NotNull(), _, _ ) )
        .WillOnce( DoAll( SetArgPointee<9>( m_fakeGAS ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( m_optix, accelBuild( m_fakeDc, m_stream, immutable, isIAS, numBuildInputs, _, _, _, _, NotNull(), _, _ ) )
        .WillOnce( DoAll( SetArgPointee<9>( m_fakeIAS ), Return( OPTIX_SUCCESS ) ) );
}

void TestProxyInstance::configureZeroInstanceIASAccelBuildInput( OptixBuildInput& iasBuildInput )
{
    const uint_t numBuildInputs{ 1 };
    auto         immutable = AllOf( NotNull(), isBuildOperation(), Not( buildAllowsUpdate() ) );
    auto         isIAS     = AllOf( NotNull(), isInstanceBuildInput(), isZeroInstances() );
    EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeDc, immutable, isIAS, numBuildInputs, NotNull() ) )
        .WillOnce( DoAll( SaveArgPointee<2>( &iasBuildInput ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( m_optix, accelBuild( m_fakeDc, m_stream, immutable, isIAS, numBuildInputs, _, _, _, _, NotNull(), _, _ ) )
        .WillOnce( DoAll( SetArgPointee<9>( m_fakeIAS ), Return( OPTIX_SUCCESS ) ) );
}

}  // namespace

TEST_F( TestProxyInstance, addProxyAllocatesResource )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );

    const uint_t pageId = m_instances.add( m_proxyBounds );

    ASSERT_EQ( m_startPageId, pageId );
}

TEST_F( TestProxyInstance, addMultipleProxiesReturnsDifferentPageIds )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );

    const uint_t pageId1 = m_instances.add( m_proxyBounds );
    const uint_t pageId2 = m_instances.add( m_proxy2Bounds );

    ASSERT_NE( pageId1, pageId2 );
}

static void getDeviceInstances( OptixInstance* dest, const OptixBuildInput& iasBuildInput )
{
    OTK_ERROR_CHECK( cudaMemcpy( dest, reinterpret_cast<const void*>( iasBuildInput.instanceArray.instances ),
                                 iasBuildInput.instanceArray.numInstances * sizeof( OptixInstance ), cudaMemcpyDeviceToHost ) );
}

TEST_F( TestProxyInstance, createAccelsUsesExpectedTransform )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).Times( 1 );
    m_instances.add( m_proxyBounds );
    OptixBuildInput gasBuildInput{}, iasBuildInput{};
    configureAccelBuildInputs( &gasBuildInput, &iasBuildInput );

    OptixTraversableHandle result = m_instances.createTraversable( m_fakeDc, m_stream );

    EXPECT_EQ( m_fakeIAS, result );
    ASSERT_EQ( OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES, gasBuildInput.type );
    EXPECT_EQ( 1U, gasBuildInput.customPrimitiveArray.numPrimitives );
    ASSERT_EQ( OPTIX_BUILD_INPUT_TYPE_INSTANCES, iasBuildInput.type );
    EXPECT_EQ( 1U, iasBuildInput.instanceArray.numInstances );
    // clang-format off
    const float expectedTransform[12]{
        2.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 2.0f, 0.0f, -1.0f,
        0.0f, 0.0f, 2.0f, -1.0f
    };
    // clang-format on
    OptixInstance actualInstance{};
    getDeviceInstances( &actualInstance, iasBuildInput );
    EXPECT_TRUE( isSameTransform( expectedTransform, actualInstance.transform ) );
}

TEST_F( TestProxyInstance, multipleProxiesInstantiatesGASOncePerProxy )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );
    m_instances.add( m_proxyBounds );
    m_instances.add( m_proxy2Bounds );
    OptixBuildInput gasBuildInput{}, iasBuildInput{};
    configureAccelBuildInputs( &gasBuildInput, &iasBuildInput );

    OptixTraversableHandle result = m_instances.createTraversable( m_fakeDc, m_stream );

    EXPECT_EQ( m_fakeIAS, result );
    ASSERT_EQ( OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES, gasBuildInput.type );
    EXPECT_EQ( 1U, gasBuildInput.customPrimitiveArray.numPrimitives );
    ASSERT_EQ( OPTIX_BUILD_INPUT_TYPE_INSTANCES, iasBuildInput.type );
    const uint_t NUM_EXPECTED_INSTANCES{ 2 };
    EXPECT_EQ( NUM_EXPECTED_INSTANCES, iasBuildInput.instanceArray.numInstances );
    OptixInstance actualInstance[NUM_EXPECTED_INSTANCES]{};
    getDeviceInstances( actualInstance, iasBuildInput );
    // clang-format off
    const float expectedTransform1[12]{
        2.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 2.0f, 0.0f, -1.0f,
        0.0f, 0.0f, 2.0f, -1.0f
    };
    const float expectedTransform2[12]{
        4.0f, 0.0f, 0.0f, 10.0f,
        0.0f, 2.0f, 0.0f, 11.0f,
        0.0f, 0.0f, 1.0f, 12.0f
    };
    // clang-format on
    EXPECT_TRUE( isSameTransform( expectedTransform1, actualInstance[0].transform ) );
    EXPECT_TRUE( isSameTransform( expectedTransform2, actualInstance[1].transform ) );
}

TEST_F( TestProxyInstance, proxyInstantiatesCustomPrimitive )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).Times( 1 );
    m_instances.add( m_proxyBounds );
    OptixBuildInput gasBuildInput{}, iasBuildInput{};
    configureAccelBuildInputs( &gasBuildInput, &iasBuildInput );

    m_instances.createTraversable( m_fakeDc, m_stream );

    OptixInstance actualInstance{};
    ASSERT_EQ( OPTIX_BUILD_INPUT_TYPE_INSTANCES, iasBuildInput.type );
    EXPECT_EQ( 1U, iasBuildInput.instanceArray.numInstances );
    getDeviceInstances( &actualInstance, iasBuildInput );
    EXPECT_EQ( m_fakeGAS, actualInstance.traversableHandle );
}

TEST_F( TestProxyInstance, resourceCallbackSavesPageId )
{
    ResourceCallback resourceCallback{};
    void*            context{};
    EXPECT_CALL( m_loader, createResource( _, _, _ ) )
        .WillOnce( DoAll( SaveArg<1>( &resourceCallback ), SaveArg<2>( &context ), Return( m_startPageId ) ) );
    m_instances.add( m_proxyBounds );
    const std::vector<uint_t> beforePageIds = m_instances.requestedProxyIds();

    void*      result;
    const bool satisfied = resourceCallback( m_stream, m_startPageId, context, &result );

    EXPECT_TRUE( satisfied );
    EXPECT_TRUE( beforePageIds.empty() );
    EXPECT_EQ( nullptr, result );
    const std::vector<uint_t> afterPageIds = m_instances.requestedProxyIds();
    EXPECT_FALSE( afterPageIds.empty() );
    EXPECT_NE( afterPageIds.end(), std::find( afterPageIds.begin(), afterPageIds.end(), m_startPageId ) );
}

TEST_F( TestProxyInstance, resourceCallbackDeduplicatesPageId )
{
    ResourceCallback resourceCallback{};
    void*            context{};
    EXPECT_CALL( m_loader, createResource( _, _, _ ) )
        .WillOnce( DoAll( SaveArg<1>( &resourceCallback ), SaveArg<2>( &context ), Return( m_startPageId ) ) );
    m_instances.add( m_proxyBounds );
    const std::vector<uint_t> beforePageIds = m_instances.requestedProxyIds();

    void* pageTableEntry;
    (void)resourceCallback( m_stream, m_startPageId, context, &pageTableEntry );
    (void)resourceCallback( m_stream, m_startPageId, context, &pageTableEntry );

    const std::vector<uint_t> afterPageIds = m_instances.requestedProxyIds();
    EXPECT_EQ( static_cast<size_t>( 1 ), afterPageIds.size() );
}

TEST_F( TestProxyInstance, removingAProxyRemovesTheCustomPrimitiveInstance )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );
    m_instances.add( m_proxyBounds );
    configureAccelBuildInputs( nullptr, nullptr );
    OptixBuildInput instanceInput{};
    configureZeroInstanceIASAccelBuildInput( instanceInput );
    m_instances.createTraversable( m_fakeDc, m_stream );

    m_instances.remove( m_startPageId );
    m_instances.createTraversable( m_fakeDc, m_stream );
}
