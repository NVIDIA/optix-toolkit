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

MATCHER_P3( hasDeviceInstanceTraversable, n, instanceIndex, expectedTraversable, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_INSTANCES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type
                         << ", expected OPTIX_BUILD_INPUT_TYPE_INSTANCES (" << OPTIX_BUILD_INPUT_TYPE_INSTANCES << ')';
        return false;
    }
    const OptixBuildInputInstanceArray& instances = arg[n].instanceArray;
    if( instanceIndex >= instances.numInstances )
    {
        *result_listener << "input " << n << " has " << instances.numInstances << " instances, expected at least "
                         << instanceIndex + 1;
        return false;
    }
    std::vector<OptixInstance> actualInstances;
    actualInstances.resize( instances.numInstances );
    OTK_ERROR_CHECK( cudaMemcpy( actualInstances.data(), reinterpret_cast<const void*>( instances.instances ),
                                 instances.numInstances * sizeof( OptixInstance ), cudaMemcpyDeviceToHost ) );
    if( expectedTraversable != actualInstances[instanceIndex].traversableHandle )
    {
        *result_listener << "input " << n << " instance " << instanceIndex << " has traversable "
                         << actualInstances[instanceIndex].traversableHandle << ", expected " << expectedTraversable;
        return false;
    }

    return true;
}

MATCHER_P2( hasAnyDeviceInstanceId, n, instanceId, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_INSTANCES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type
                         << ", expected OPTIX_BUILD_INPUT_TYPE_INSTANCES (" << OPTIX_BUILD_INPUT_TYPE_INSTANCES << ')';
        return false;
    }
    const OptixBuildInputInstanceArray& instances = arg[n].instanceArray;
    std::vector<OptixInstance>          actualInstances;
    actualInstances.resize( instances.numInstances );
    OTK_ERROR_CHECK( cudaMemcpy( actualInstances.data(), reinterpret_cast<const void*>( instances.instances ),
                                 instances.numInstances * sizeof( OptixInstance ), cudaMemcpyDeviceToHost ) );
    for( uint_t i = 0; i < instances.numInstances; ++i )
    {
        if( instanceId == actualInstances[i].instanceId )
        {
            return true;
        }
    }

    *result_listener << "input " << n << " with " << instances.numInstances << " instances does not contain id " << instanceId;
    return false;
}

static auto isBuildingNumInstances = []( uint_t buildInput, uint_t numInstances ) {
    return AllOf( NotNull(), hasInstanceBuildInput( buildInput, hasNumInstances( numInstances ) ) );
};

static auto immutable = AllOf( NotNull(), isBuildOperation(), Not( buildAllowsUpdate() ) );

static auto setHandle = []( OptixTraversableHandle handle ) {
    return DoAll( SetArgPointee<9>( handle ), Return( OPTIX_SUCCESS ) );
};


namespace {

class TestProxyInstance : public Test
{
  protected:
    void SetUp() override;

    template <typename Matcher>
    Expectation configureAccelComputeMemoryUsage( Matcher& matcher )
    {
        const uint_t numBuildInputs = 1;
        return EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeDc, immutable, matcher, numBuildInputs, NotNull() ) )
            .WillOnce( Return( OPTIX_SUCCESS ) );
    }

    template <typename Matcher>
    Expectation configureAccelBuild( Matcher& matcher, OptixTraversableHandle traversable )
    {
        const uint_t numBuildInputs = 1;
        return EXPECT_CALL( m_optix,
                            accelBuild( m_fakeDc, m_stream, immutable, matcher, numBuildInputs, _, _, _, _, NotNull(), _, _ ) )
            .WillOnce( setHandle( traversable ) );
    }

    template <typename Matcher>
    void configureUpdatedBuild( const ExpectationSet& first, Matcher& isUpdatedIAS, OptixTraversableHandle updatedIAS )
    {
        const uint_t numUpdatedBuildInputs{ 1 };
        EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeDc, immutable, isUpdatedIAS, numUpdatedBuildInputs, NotNull() ) )
            .After( first )
            .WillOnce( Return( OPTIX_SUCCESS ) );
        EXPECT_CALL( m_optix, accelBuild( m_fakeDc, m_stream, immutable, isUpdatedIAS, numUpdatedBuildInputs, _, _, _,
                                          _, NotNull(), _, _ ) )
            .After( first )
            .WillOnce( setHandle( updatedIAS ) );
    }

    MockDemandLoader m_loader;
    ProxyInstances   m_instances{ &m_loader };
    MockOptix        m_optix;

    // We need to use the real default stream because we're not mocking CUDA.
    CUstream m_stream{};

    uint_t    m_startPageId{ 1964 };
    OptixAabb m_proxy1Bounds{ -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };
    // clang-format off
    std::array<float, 12> m_proxy1Transform{
        2.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 2.0f, 0.0f, -1.0f,
        0.0f, 0.0f, 2.0f, -1.0f
    };
    // clang-format on
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

}  // namespace

TEST_F( TestProxyInstance, addProxyAllocatesResource )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );

    const uint_t pageId = m_instances.add( m_proxy1Bounds );

    EXPECT_EQ( m_startPageId, pageId );
}

TEST_F( TestProxyInstance, addMultipleProxiesReturnsDifferentPageIds )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );

    const uint_t pageId1 = m_instances.add( m_proxy1Bounds );
    const uint_t pageId2 = m_instances.add( m_proxy2Bounds );

    EXPECT_NE( pageId1, pageId2 );
}

TEST_F( TestProxyInstance, createAccelsUsesExpectedTransform )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).Times( 1 );
    const uint_t pageId = m_instances.add( m_proxy1Bounds );
    const uint_t numBuildInputs{ 1 };
    auto         isGAS = AllOf( NotNull(), hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) );
    auto         isIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1 ),
                                                 hasDeviceInstances( hasInstance( 0U, hasInstanceId( pageId ),
                                                                                  hasInstanceTransform( m_proxy1Transform ) ) ) ) ) );
    auto setSize = []( const OptixAccelBufferSizes& sizes ) {
        return DoAll( SetArgPointee<4>( sizes ), Return( OPTIX_SUCCESS ) );
    };
    OptixAccelBufferSizes gasSizes{ 1000, 1100, 0 };
    EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeDc, immutable, isGAS, numBuildInputs, NotNull() ) ).WillOnce( setSize( gasSizes ) );
    OptixAccelBufferSizes iasSizes{ 2000, 2200, 0 };
    EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeDc, immutable, isIAS, numBuildInputs, NotNull() ) ).WillOnce( setSize( iasSizes ) );
    EXPECT_CALL( m_optix, accelBuild( m_fakeDc, m_stream, immutable, isGAS, numBuildInputs, Ne( 0 ), gasSizes.tempSizeInBytes,
                                      Ne( 0 ), gasSizes.outputSizeInBytes, NotNull(), nullptr, 0 ) )
        .WillOnce( setHandle( m_fakeGAS ) );
    EXPECT_CALL( m_optix, accelBuild( m_fakeDc, m_stream, immutable, isIAS, numBuildInputs, Ne( 0 ), iasSizes.tempSizeInBytes,
                                      Ne( 0 ), iasSizes.outputSizeInBytes, NotNull(), nullptr, 0 ) )
        .WillOnce( setHandle( m_fakeIAS ) );

    OptixTraversableHandle result = m_instances.createTraversable( m_fakeDc, m_stream );

    EXPECT_EQ( m_fakeIAS, result );
}

TEST_F( TestProxyInstance, multipleProxiesInstantiatesGASOncePerProxy )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );
    m_instances.add( m_proxy1Bounds );
    m_instances.add( m_proxy2Bounds );
    auto isGAS = AllOf( NotNull(), hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) );
    // clang-format off
    const std::array<float, 12> expectedTransform1{
        2.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 2.0f, 0.0f, -1.0f,
        0.0f, 0.0f, 2.0f, -1.0f
    };
    const std::array<float, 12> expectedTransform2{
        4.0f, 0.0f, 0.0f, 10.0f,
        0.0f, 2.0f, 0.0f, 11.0f,
        0.0f, 0.0f, 1.0f, 12.0f
    };
    // clang-format on
    auto isIAS =
        AllOf( NotNull(), hasInstanceBuildInput(
                              0, hasAll( hasNumInstances( 2 ),
                                         hasDeviceInstances( hasInstance( 0U, hasInstanceTransform( expectedTransform1 ) ),
                                                             hasInstance( 1U, hasInstanceTransform( expectedTransform2 ) ) ) ) ) );
    configureAccelComputeMemoryUsage( isGAS );
    configureAccelComputeMemoryUsage( isIAS );
    configureAccelBuild( isGAS, m_fakeGAS );
    configureAccelBuild( isIAS, m_fakeIAS );

    OptixTraversableHandle result = m_instances.createTraversable( m_fakeDc, m_stream );

    EXPECT_EQ( m_fakeIAS, result );
}

TEST_F( TestProxyInstance, proxyInstantiatesCustomPrimitive )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).Times( 1 );
    m_instances.add( m_proxy1Bounds );
    auto isGAS = AllOf( NotNull(), hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) );
    auto isIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1U ),
                                                 hasDeviceInstances( hasInstance( 0U, hasInstanceTraversable( m_fakeGAS ) ) ) ) ) );
    configureAccelComputeMemoryUsage( isGAS );
    configureAccelComputeMemoryUsage( isIAS );
    configureAccelBuild( isGAS, m_fakeGAS );
    configureAccelBuild( isIAS, m_fakeIAS );

    m_instances.createTraversable( m_fakeDc, m_stream );
}

TEST_F( TestProxyInstance, resourceCallbackSavesPageId )
{
    ResourceCallback resourceCallback{};
    void*            context{};
    EXPECT_CALL( m_loader, createResource( _, _, _ ) )
        .WillOnce( DoAll( SaveArg<1>( &resourceCallback ), SaveArg<2>( &context ), Return( m_startPageId ) ) );
    m_instances.add( m_proxy1Bounds );
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
    m_instances.add( m_proxy1Bounds );
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
    const uint_t pageId = m_instances.add( m_proxy1Bounds );
    auto         isGAS  = AllOf( NotNull(), hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) );
    auto         isIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1U ),
                                                 hasDeviceInstances( hasInstance( 0U, hasInstanceId( pageId ),
                                                                                  hasInstanceTransform( m_proxy1Transform ) ) ) ) ) );
    ExpectationSet first;
    first += configureAccelComputeMemoryUsage( isGAS );
    first += configureAccelComputeMemoryUsage( isIAS );
    first += configureAccelBuild( isGAS, m_fakeGAS );
    first += configureAccelBuild( isIAS, m_fakeIAS );
    OptixTraversableHandle updatedIAS{ 7777 };
    auto                   isUpdatedIAS = isBuildingNumInstances( 0, 0U );
    configureUpdatedBuild( first, isUpdatedIAS, updatedIAS );
    m_instances.createTraversable( m_fakeDc, m_stream );
    EXPECT_CALL( m_loader, unloadResource( _ ) ).Times( 0 );

    m_instances.remove( m_startPageId );
    OptixTraversableHandle updatedHandle = m_instances.createTraversable( m_fakeDc, m_stream );

    EXPECT_EQ( updatedHandle, updatedIAS );
}

TEST_F( TestProxyInstance, removingMultipleProxiesKeepsOtherProxies )
{
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( m_startPageId ) );
    uint_t    id1 = m_instances.add( m_proxy1Bounds );
    uint_t    id2 = m_instances.add( m_proxy2Bounds );
    OptixAabb bounds3{ -5.0f, -5.0f, -5.0f, -4.0f, -4.0f, -4.0f };
    uint_t    id3   = m_instances.add( bounds3 );
    auto      isGAS = AllOf( NotNull(), hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) );
    // clang-format off
    const std::array<float, 12> expectedTransform3{
        1.0f, 0.0f, 0.0f, -5.0f,
        0.0f, 1.0f, 0.0f, -5.0f,
        0.0f, 0.0f, 1.0f, -5.0f
    };
    // clang-format on
    auto isIAS = AllOf(
        NotNull(),
        hasInstanceBuildInput(
            0, hasAll( hasNumInstances( 3 ),
                       hasDeviceInstances( hasInstance( 0U, hasInstanceId( id1 ) ), hasInstance( 1U, hasInstanceId( id2 ) ),
                                           hasInstance( 2U, hasInstanceId( id3 ), hasInstanceTransform( expectedTransform3 ) ) ) ) ) );
    ExpectationSet first;
    first += configureAccelComputeMemoryUsage( isGAS );
    first += configureAccelComputeMemoryUsage( isIAS );
    first += configureAccelBuild( isGAS, m_fakeGAS );
    first += configureAccelBuild( isIAS, m_fakeIAS );
    OptixTraversableHandle updatedIAS{ 7777 };
    auto                   isUpdatedIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1 ),
                                                 hasDeviceInstances( hasInstance( 0U, hasInstanceId( id3 ),
                                                                                  hasInstanceTransform( expectedTransform3 ) ) ) ) ) );
    configureUpdatedBuild( first, isUpdatedIAS, updatedIAS );
    OptixTraversableHandle initialHandle = m_instances.createTraversable( m_fakeDc, m_stream );
    EXPECT_CALL( m_loader, unloadResource( _ ) ).Times( 0 );

    m_instances.remove( id2 );
    m_instances.remove( id1 );
    OptixTraversableHandle handle = m_instances.createTraversable( m_fakeDc, m_stream );

    EXPECT_NE( initialHandle, handle );
    EXPECT_NE( m_startPageId, id3 );
}

OptixInstanceVectorPredicate hasNoInstanceWithId( unsigned int instanceId )
{
    return [=]( ::testing::MatchResultListener* listener, const std::vector<OptixInstance>& instances ) {
        for( std::size_t i = 0; i < instances.size(); ++i )
        {
            if( instanceId == instances[i].instanceId )
            {
                *listener << "instance " << i << " has id " << instanceId;
                return false;
            }
        }

        *listener << "instance array with " << instances.size() << " instances does not contain id " << instanceId;
        return true;
    };
}

TEST_F( TestProxyInstance, removeOutOfOrderPageIdProxies )
{
    // Allocate higher ids, then lower ids
    const uint_t   batch1StartId = m_startPageId + 2 * ProxyInstances::PAGE_CHUNK_SIZE;
    ExpectationSet firstBatch;
    firstBatch += EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( batch1StartId ) );
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).After( firstBatch ).WillOnce( Return( m_startPageId ) );
    std::vector<OptixAabb> batch1ProxyBounds;
    std::vector<uint_t>    batch1ProxyIds;
    for( uint_t i = 0; i < ProxyInstances::PAGE_CHUNK_SIZE; ++i )
    {
        const float     minCoord = 10.0f + static_cast<float>( i );
        const float     maxCoord = minCoord + 1.0f;
        const OptixAabb bounds{ minCoord, minCoord, minCoord, maxCoord, maxCoord, maxCoord };
        batch1ProxyBounds.push_back( bounds );
        batch1ProxyIds.push_back( m_instances.add( bounds ) );
        EXPECT_GE( batch1ProxyIds.back(), batch1StartId );
    }
    const uint_t lowerInstanceIndex = 0U;
    const uint_t lowerPageId        = m_instances.add( m_proxy1Bounds );
    EXPECT_LT( lowerPageId, batch1StartId );
    auto         isGAS = AllOf( NotNull(), hasCustomPrimitiveBuildInput( 0, hasNumCustomPrimitives( 1U ) ) );
    const uint_t numInitialInstances = ProxyInstances::PAGE_CHUNK_SIZE + 1;
    auto         isIAS =
        AllOf( NotNull(), hasInstanceBuildInput(
                              0, hasAll( hasNumInstances( numInitialInstances ),
                                         hasDeviceInstances( hasInstance( lowerInstanceIndex, hasInstanceId( lowerPageId ),
                                                                          hasInstanceTransform( m_proxy1Transform ) ) ) ) ) );
    ExpectationSet first;
    first += configureAccelComputeMemoryUsage( isGAS );
    first += configureAccelComputeMemoryUsage( isIAS );
    first += configureAccelBuild( isGAS, m_fakeGAS );
    first += configureAccelBuild( isIAS, m_fakeIAS );
    OptixTraversableHandle updatedIAS{ 7777 };
    auto                   isUpdatedIAS =
        AllOf( NotNull(), hasInstanceBuildInput( 0, hasAll( hasNumInstances( numInitialInstances - 1U ),
                                                            hasDeviceInstances( hasNoInstanceWithId( lowerPageId ) ) ) ) );
    configureUpdatedBuild( first, isUpdatedIAS, updatedIAS );
    OptixTraversableHandle initialHandle = m_instances.createTraversable( m_fakeDc, m_stream );
    EXPECT_CALL( m_loader, unloadResource( _ ) ).Times( 0 );

    m_instances.remove( lowerPageId );
    OptixTraversableHandle handle = m_instances.createTraversable( m_fakeDc, m_stream );

    EXPECT_NE( initialHandle, handle );
    EXPECT_EQ( updatedIAS, handle );
}

TEST_F( TestProxyInstance, removedPageIdsAreRecycled )
{
    m_instances.setRecycleProxyIds( true );
    const OptixAabb bounds1{ 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
    const OptixAabb bounds2{ 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f };
    const uint_t    firstId{ 1010 };
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( firstId ) );
    const uint_t id1 = m_instances.add( bounds1 );
    const uint_t id2 = m_instances.add( bounds2 );
    EXPECT_CALL( m_loader, unloadResource( id1 ) );
    m_instances.remove( id1 );

    const uint_t id3 = m_instances.add( bounds1 );

    EXPECT_EQ( firstId, id1 );
    EXPECT_EQ( firstId + 1, id2 );
    EXPECT_EQ( id1, id3 );
    EXPECT_NE( id1, id2 );
}

TEST_F( TestProxyInstance, alwaysNewPageIdWhenNotRecycling )
{
    m_instances.setRecycleProxyIds( false ); // the default
    const OptixAabb bounds1{ 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
    const OptixAabb bounds2{ 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f };
    const uint_t    firstId{ 1010 };
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( firstId ) );
    EXPECT_CALL( m_loader, unloadResource( _ ) ).Times( 0 );
    const uint_t id1 = m_instances.add( bounds1 );
    const uint_t id2 = m_instances.add( bounds2 );
    m_instances.remove( id1 );

    const uint_t id3 = m_instances.add( bounds1 );

    EXPECT_EQ( firstId, id1 );
    EXPECT_EQ( firstId + 1, id2 );
    EXPECT_EQ( firstId + 2, id3 );
    EXPECT_NE( id1, id2 );
    EXPECT_NE( id1, id3 );
}

TEST_F( TestProxyInstance, removingTwiceThrowsException )
{
    const OptixAabb bounds1{ 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
    const uint_t    firstId{ 1010 };
    EXPECT_CALL( m_loader, createResource( _, _, _ ) ).WillOnce( Return( firstId ) );
    const uint_t id1 = m_instances.add( bounds1 );
    EXPECT_CALL( m_loader, unloadResource( _ ) ).Times( 0 );
    m_instances.remove( id1 );

    EXPECT_THROW( m_instances.remove( id1 ), std::runtime_error );
}
