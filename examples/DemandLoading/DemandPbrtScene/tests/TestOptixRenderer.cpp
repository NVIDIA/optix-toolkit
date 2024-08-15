// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptixRenderer.h>

#include "MockGeometryLoader.h"

#include <Options.h>
#include <Params.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockOptix.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/BitCast.h>
#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <ios>
#include <vector>

using namespace demandPbrtScene;
using namespace otk::testing;
using namespace testing;

namespace {

using StrictMockOptix = StrictMock<MockOptix>;

using OptixRendererPtr = std::shared_ptr<OptixRenderer>;

constexpr int NUM_ATTRIBUTES = 4;

inline OptixProgramGroup PG( unsigned int id )
{
    return otk::bit_cast<OptixProgramGroup>( static_cast<std::intptr_t>( id ) );
};

class TestOptixRenderer : public Test
{
  protected:
    void SetUp() override;
    void TearDown() override;

    ExpectationSet expectInitialize();
    ExpectationSet expectBeforeLaunchAfter( const ExpectationSet& before );
    ExpectationSet expectLaunchAfter( const ExpectationSet& before );

    Options                        m_options{};
    MockGeometryLoaderPtr          m_geometryLoader{ createMockGeometryLoader() };
    StrictMockOptix                m_optix{};
    CUstream                       m_stream{};
    uchar4                         m_image[1]{};
    OptixDeviceContext             m_fakeDeviceContext{ otk::bit_cast<OptixDeviceContext>( 0xf00df00dULL ) };
    OptixRendererPtr               m_renderer{ std::make_shared<OptixRenderer>( m_options, NUM_ATTRIBUTES ) };
    std::vector<OptixProgramGroup> m_fakeProgramGroups{ PG( 111100U ), PG( 2222000U ), PG( 333300U ), PG( 444400U ),
                                                        PG( 555500U ), PG( 666600U ),  PG( 777700U ) };
    OptixPipeline                  fakePipeline{ otk::bit_cast<OptixPipeline>( 0xbaadf00dULL ) };
};

void TestOptixRenderer::SetUp()
{
    Test::SetUp();
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0 ) );
    initMockOptix( m_optix );
}

void TestOptixRenderer::TearDown()
{
    OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
}

ExpectationSet TestOptixRenderer::expectInitialize()
{
    ExpectationSet expect;
    expect += EXPECT_CALL( *m_geometryLoader, getNumAttributes() ).WillRepeatedly( Return( 1 ) );
    expect +=
        EXPECT_CALL( m_optix, deviceContextCreate( _, _, _ ) ).WillOnce( DoAll( SetArgPointee<2>( m_fakeDeviceContext ), Return( OPTIX_SUCCESS ) ) );
    m_renderer->initialize( m_stream );
    return expect;
}

ExpectationSet TestOptixRenderer::expectBeforeLaunchAfter( const ExpectationSet& before )
{
    ExpectationSet expect;
    expect += EXPECT_CALL( m_optix, pipelineCreate( m_fakeDeviceContext, _, _, _, m_fakeProgramGroups.size(), NotNull(),
                                                    NotNull(), NotNull() ) )
                  .After( before )
                  .WillOnce( DoAll( SetArgPointee<7>( fakePipeline ), Return( OPTIX_SUCCESS ) ) );
    const OptixStackSizes stackSizes{ 1U, 2U, 3U, 4U, 5U, 6U, 7U };
    for( OptixProgramGroup group : m_fakeProgramGroups )
    {
#if OPTIX_VERSION >= 70700
        expect += EXPECT_CALL( m_optix, programGroupGetStackSize( group, NotNull(), fakePipeline ) )
#else
        expect += EXPECT_CALL( m_optix, programGroupGetStackSize( group, NotNull() ) )
#endif
                      .After( before )
                      .WillOnce( DoAll( SetArgPointee<1>( stackSizes ), Return( OPTIX_SUCCESS ) ) );
    }
    expect += EXPECT_CALL( m_optix, pipelineSetStackSize( fakePipeline, _, _, _, _ ) ).After( before ).WillOnce( Return( OPTIX_SUCCESS ) );
    for( OptixProgramGroup group : m_fakeProgramGroups )
    {
        expect += EXPECT_CALL( m_optix, sbtRecordPackHeader( group, NotNull() ) ).After( before ).WillOnce( Return( OPTIX_SUCCESS ) );
    }
    m_renderer->setProgramGroups( m_fakeProgramGroups );
    m_renderer->beforeLaunch( m_stream );
    return expect;
}

ExpectationSet TestOptixRenderer::expectLaunchAfter( const ExpectationSet& before )
{
    ExpectationSet expect;
    expect += EXPECT_CALL( m_optix, launch( fakePipeline, m_stream, _, _, _, _, _, _ ) ).After( before ).WillOnce( Return( OPTIX_SUCCESS ) );
    m_renderer->launch( m_stream, m_image );
    return expect;
}

}  // namespace

TEST_F( TestOptixRenderer, initializeCreatesOptixResources )
{
    EXPECT_CALL( m_optix, deviceContextCreate( nullptr, NotNull(), NotNull() ) )
        .WillOnce( DoAll( SetArgPointee<2>( m_fakeDeviceContext ), Return( OPTIX_SUCCESS ) ) );

    m_renderer->initialize( m_stream );
}

TEST_F( TestOptixRenderer, initializeDebugLocationFromOptions )
{
    EXPECT_CALL( m_optix, deviceContextCreate( _, _, _ ) ).WillRepeatedly( Return( OPTIX_SUCCESS ) );
    m_options.debug      = true;
    m_options.debugPixel = make_int2( 20, 30 );

    m_renderer->initialize( m_stream );

    const Params& params = m_renderer->getParams();
    EXPECT_TRUE( params.debug.enabled );
    EXPECT_TRUE( params.debug.debugIndexSet );
    EXPECT_EQ( make_uint3( 20, 30, 0 ), params.debug.debugIndex );
}

TEST_F( TestOptixRenderer, initializeOneShotDebuggingFromOptions )
{
    EXPECT_CALL( m_optix, deviceContextCreate( _, _, _ ) ).WillRepeatedly( Return( OPTIX_SUCCESS ) );
    m_options.debug        = true;
    m_options.debugPixel   = make_int2( 20, 30 );
    m_options.oneShotDebug = true;

    m_renderer->initialize( m_stream );

    const Params& params = m_renderer->getParams();
    EXPECT_TRUE( params.debug.enabled );
    EXPECT_TRUE( params.debug.debugIndexSet );
    EXPECT_FALSE( params.debug.dumpSuppressed );
    EXPECT_EQ( make_uint3( 20, 30, 0 ), params.debug.debugIndex );
}

TEST_F( TestOptixRenderer, initializeUseFaceForwardFromOptions )
{
    EXPECT_CALL( m_optix, deviceContextCreate( _, _, _ ) ).WillRepeatedly( Return( OPTIX_SUCCESS ) );
    m_options.faceForward = true;

    m_renderer->initialize( m_stream );

    const Params& params = m_renderer->getParams();
    EXPECT_TRUE( params.useFaceForward );
}

TEST_F( TestOptixRenderer, cleanupDestroysOptixResources )
{
    expectInitialize();
    EXPECT_CALL( m_optix, deviceContextDestroy( m_fakeDeviceContext ) ).WillOnce( Return( OPTIX_SUCCESS ) );

    m_renderer->cleanup();
}

template <typename T, typename U>
AssertionResult isBitSet( T value, U bit )
{
    if( ( value & bit ) == static_cast<T>( bit ) )
        return AssertionSuccess() << "bit " << std::hex << bit << " set in " << value;

    return AssertionFailure() << "bit " << std::hex << bit << " not set in " << value;
}

TEST_F( TestOptixRenderer, pipelineCompileOptionsUsesTriangles )
{
    expectInitialize();

    const OptixPipelineCompileOptions& options = m_renderer->getPipelineCompileOptions();

    EXPECT_TRUE( isBitSet( options.usesPrimitiveTypeFlags, OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ) );
}

TEST_F( TestOptixRenderer, pipelineCompileOptionsUsesSpheres )
{
    expectInitialize();

    const OptixPipelineCompileOptions& options = m_renderer->getPipelineCompileOptions();

    EXPECT_TRUE( isBitSet( options.usesPrimitiveTypeFlags, OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE ) );
}

MATCHER( Always, "" )
{
    *result_listener << "always matches";
    return true;
}

TEST_F( TestOptixRenderer, beforeFirstLaunchCreatesPipelineAndPacksSbtRecords )
{
    ExpectationSet init = expectInitialize();
#if OPTIX_VERSION >= 70600
    auto opacityMaps = Not( allowsOpacityMicromaps() );
#else
    auto opacityMaps = Always();
#endif
    auto compileOptions = AllOf( NotNull(), Not( usesMotionBlur() ), allowAnyTraversableGraph(),
                                 hasPayloadValueCount( 3 ), hasAttributeValueCount( 4 ),
                                 hasExceptionFlags( OPTIX_EXCEPTION_FLAG_NONE ), hasParamsName( PARAMS_STRING_NAME ),
                                 hasPrimitiveTypes( OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE
                                                    | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ),
                                 opacityMaps );
    auto linkOptions    = AllOf( NotNull(), hasMaxTraceDepth( 1 ) );
    auto programGroups  = NotNull();
    EXPECT_CALL( m_optix, pipelineCreate( m_fakeDeviceContext, compileOptions, linkOptions, programGroups,
                                          m_fakeProgramGroups.size(), NotNull(), NotNull(), NotNull() ) )
        .After( init )
        .WillOnce( DoAll( SetArgPointee<7>( fakePipeline ), Return( OPTIX_SUCCESS ) ) );
    const OptixStackSizes stackSizes{ 1U, 2U, 3U, 4U, 5U, 6U, 7U };
    for( OptixProgramGroup group : m_fakeProgramGroups )
    {
#if OPTIX_VERSION >= 70700
        EXPECT_CALL( m_optix, programGroupGetStackSize( group, NotNull(), fakePipeline ) )
#else
        EXPECT_CALL( m_optix, programGroupGetStackSize( group, NotNull() ) )
#endif
            .After( init )
            .WillOnce( DoAll( SetArgPointee<1>( stackSizes ), Return( OPTIX_SUCCESS ) ) );
    }
    EXPECT_CALL( m_optix, pipelineSetStackSize( fakePipeline, _, _, _, _ ) ).After( init ).WillOnce( Return( OPTIX_SUCCESS ) );
    for( OptixProgramGroup group : m_fakeProgramGroups )
    {
        EXPECT_CALL( m_optix, sbtRecordPackHeader( group, NotNull() ) ).After( init ).WillOnce( Return( OPTIX_SUCCESS ) );
    }
    m_renderer->setProgramGroups( m_fakeProgramGroups );

    m_renderer->beforeLaunch( m_stream );
}

ListenerPredicate<Params> hasImage( uchar4* image )
{
    return [=]( MatchResultListener* listener, const Params& params ) {
        if( params.image != image )
        {
            *listener << "expected image " << static_cast<void*>( image ) << ", got " << static_cast<void*>( params.image );
            return false;
        }

        *listener << "has image " << static_cast<void*>( image );
        return true;
    };
}

MATCHER_P( hasParams, predicate, "" )
{
    Params params{};
    OTK_ERROR_CHECK( cuMemcpyDtoH( &params, arg, sizeof( Params ) ) );
    return predicate( result_listener, params );
}

TEST_F( TestOptixRenderer, launchSetsParamsImage )
{
    ExpectationSet init         = expectInitialize();
    ExpectationSet beforeLaunch = expectBeforeLaunchAfter( init );
    EXPECT_CALL( m_optix, launch( fakePipeline, m_stream, hasParams( hasImage( m_image ) ), sizeof( Params ),
                                  NotNull() /*sbt*/, _ /*m_options.width*/, _ /*m_options.height*/, 1 /*depth*/ ) )
        .WillOnce( Return( OPTIX_SUCCESS ) );

    m_renderer->launch( m_stream, m_image );
}

TEST_F( TestOptixRenderer, afterLaunchSuppressesDumpWithOneShotDebugging )
{
    m_options.debug        = true;
    m_options.debugPixel   = make_int2( 20, 30 );
    m_options.oneShotDebug = true;
    expectInitialize();

    m_renderer->afterLaunch();

    const Params& params = m_renderer->getParams();
    EXPECT_TRUE( params.debug.enabled );
    EXPECT_TRUE( params.debug.debugIndexSet );
    EXPECT_TRUE( params.debug.dumpSuppressed );
    EXPECT_EQ( make_uint3( 20, 30, 0 ), params.debug.debugIndex );
}

TEST_F( TestOptixRenderer, beforeLaunchResetsDebugInfo )
{
    m_options.debug                  = true;
    m_options.debugPixel             = make_int2( 20, 30 );
    m_options.oneShotDebug           = true;
    ExpectationSet init              = expectInitialize();
    ExpectationSet beforeFirstLaunch = expectBeforeLaunchAfter( init );
    ExpectationSet firstLaunch       = expectLaunchAfter( beforeFirstLaunch );
    m_renderer->afterLaunch();
    m_options.oneShotDebug = false;

    m_renderer->beforeLaunch( m_stream );

    const Params& params = m_renderer->getParams();
    EXPECT_TRUE( params.debug.enabled );
    EXPECT_TRUE( params.debug.debugIndexSet );
    EXPECT_FALSE( params.debug.dumpSuppressed );
    EXPECT_EQ( make_uint3( 20, 30, 0 ), params.debug.debugIndex );
}
