// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <Options.h>

#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace demandPbrtScene {

std::ostream& operator<<( std::ostream& str, RenderMode value )
{
    switch( value )
    {
        case RenderMode::PRIMARY_RAY:
            return str << "primary";
        case RenderMode::NEAR_AO:
            return str << "near";
        case RenderMode::DISTANT_AO:
            return str << "distant";
        case RenderMode::PATH_TRACING:
            return str << "path";
    }
    return str << "? (" << +value << ')';
}

}

namespace {

using namespace testing;

class TestOptions : public Test
{
  protected:
    demandPbrtScene::Options getOptions( std::initializer_list<const char*> args ) const;

    StrictMock<MockFunction<demandPbrtScene::UsageFn>> m_mockUsage;
    std::function<demandPbrtScene::UsageFn>            m_usage{ m_mockUsage.AsStdFunction() };
};

demandPbrtScene::Options TestOptions::getOptions( std::initializer_list<const char*> args ) const
{
    std::vector<char*> argv;
    std::transform( args.begin(), args.end(), std::back_inserter( argv ),
                    []( const char* arg ) { return const_cast<char*>( arg ); } );
    return demandPbrtScene::parseOptions( static_cast<int>( argv.size() ), argv.data(), m_usage );
}

}  // namespace

TEST_F( TestOptions, programNameParsed )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_EQ( "DemandPbrtScene", options.program );
}

TEST_F( TestOptions, missingSceneFile )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "missing scene file argument" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene" } );
}

TEST_F( TestOptions, sceneFileArgumentParsed )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_EQ( "scene.pbrt", options.sceneFile );
}

TEST_F( TestOptions, sceneFileBetweenOptions )
{
    const demandPbrtScene::Options options =
        getOptions( { "DemandPbrtScene", "--dim=128x256", "scene.pbrt", "--file", "output.png" } );

    EXPECT_EQ( 128, options.width );
    EXPECT_EQ( 256, options.height );
    EXPECT_EQ( "scene.pbrt", options.sceneFile );
    EXPECT_EQ( "output.png", options.outFile );
}

TEST_F( TestOptions, fileArgumentParsed )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "-f", "outfile.png", "scene.pbrt" } );

    EXPECT_EQ( "outfile.png", options.outFile );
}

TEST_F( TestOptions, fileArgumentMissingValue )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "missing filename argument" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "-f" } );
}

TEST_F( TestOptions, longFormFileArgument )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--file", "outfile.png", "scene.pbrt" } );

    EXPECT_EQ( "outfile.png", options.outFile );
}

TEST_F( TestOptions, dimensionsDefaultTo768x512 )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_EQ( 768, options.width );
    EXPECT_EQ( 512, options.height );
}

TEST_F( TestOptions, dimensionsParsed )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--dim=256x512", "scene.pbrt" } );

    EXPECT_EQ( 256, options.width );
    EXPECT_EQ( 512, options.height );
}

TEST_F( TestOptions, defaultBackgroundIsBlack )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_EQ( make_float3( 0.0f, 0.0f, 0.0f ), options.background );
}

TEST_F( TestOptions, parseBackgroundColor )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--bg=0.1/0.2/0.3", "scene.pbrt" } );

    EXPECT_EQ( make_float3( 0.1f, 0.2f, 0.3f ), options.background );
}

TEST_F( TestOptions, oneShotGeometry )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--oneshot-geometry", "scene.pbrt" } );

    EXPECT_TRUE( options.oneShotGeometry );
}

TEST_F( TestOptions, oneShotMaterial )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--oneshot-material", "scene.pbrt" } );

    EXPECT_TRUE( options.oneShotMaterial );
}

TEST_F( TestOptions, noProxyResolutionLoggingByDefault )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_FALSE( options.verboseProxyGeometryResolution );
    EXPECT_FALSE( options.verboseProxyMaterialResolution );
}

TEST_F( TestOptions, verboseProxyResolution )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--proxy-resolution", "scene.pbrt" } );

    EXPECT_TRUE( options.verboseProxyGeometryResolution );
    EXPECT_TRUE( options.verboseProxyMaterialResolution );
}

TEST_F( TestOptions, verboseProxyGeometryResolution )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--proxy-geometry", "scene.pbrt" } );

    EXPECT_TRUE( options.verboseProxyGeometryResolution );
    EXPECT_FALSE( options.verboseProxyMaterialResolution );
}

TEST_F( TestOptions, verboseProxyMaterialResolution )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--proxy-material", "scene.pbrt" } );

    EXPECT_FALSE( options.verboseProxyGeometryResolution );
    EXPECT_TRUE( options.verboseProxyMaterialResolution );
}

TEST_F( TestOptions, noSceneDecompositionByDefault )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_FALSE( options.verboseSceneDecomposition );
}

TEST_F( TestOptions, verboseSceneDecomposition )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--scene-decomposition", "scene.pbrt" } );

    EXPECT_TRUE( options.verboseSceneDecomposition );
}

TEST_F( TestOptions, noTextureCreationByDefault )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_FALSE( options.verboseTextureCreation );
}

TEST_F( TestOptions, verboseTextureCreation )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--texture-creation", "scene.pbrt" } );

    EXPECT_TRUE( options.verboseTextureCreation );
}

TEST_F( TestOptions, verboseLogging )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--verbose", "scene.pbrt" } );

    EXPECT_TRUE( options.verboseProxyGeometryResolution );
    EXPECT_TRUE( options.verboseProxyMaterialResolution );
    EXPECT_TRUE( options.verboseSceneDecomposition );
    EXPECT_TRUE( options.verboseTextureCreation );
}

TEST_F( TestOptions, proxiesNotSortedByDefault )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_FALSE( options.sortProxies );
}

TEST_F( TestOptions, sortProxies )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--sort-proxies", "scene.pbrt" } );

    EXPECT_TRUE( options.sortProxies );
}

TEST_F( TestOptions, sync )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--sync", "scene.pbrt" } );

    EXPECT_TRUE( options.sync );
}

TEST_F( TestOptions, faceForward )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--face-forward", "scene.pbrt" } );

    EXPECT_TRUE( options.faceForward );
}

TEST_F( TestOptions, backgroundMissing3rdValue )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad background color value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--bg=0.1/0.2", "scene.pbrt" } );
}

TEST_F( TestOptions, backgroundWithNegativeRedValue )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad background color value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--bg=-1/2/3", "scene.pbrt" } );
}

TEST_F( TestOptions, backgroundWithNegativeGreenValue )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad background color value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--bg=1/-2/3", "scene.pbrt" } );
}

TEST_F( TestOptions, backgroundWithNegativeBlueValue )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad background color value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--bg=1/2/-3", "scene.pbrt" } );
}

TEST_F( TestOptions, warmupFrameCount )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--warmup=10", "scene.pbrt" } );

    EXPECT_EQ( 10, options.warmupFrames );
}

TEST_F( TestOptions, negativeWarmupCountInvalid )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad warmup frame count value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--warmup=-10", "scene.pbrt" } );
}

TEST_F( TestOptions, missingWarmupCountInvalid )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad warmup frame count value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--warmup=", "scene.pbrt" } );
}

TEST_F( TestOptions, parseDebugPixel )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=384/256", "scene.pbrt" } );

    EXPECT_TRUE( options.debug );
    EXPECT_EQ( make_int2( 384, 256 ), options.debugPixel );
}

TEST_F( TestOptions, oneShotDebug )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--oneshot-debug", "scene.pbrt" } );

    EXPECT_TRUE( options.oneShotDebug );
}

TEST_F( TestOptions, negativeDebugPixelX )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad debug pixel value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=-1/256", "scene.pbrt" } );
}

TEST_F( TestOptions, negativeDebugPixelY )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad debug pixel value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=384/-1", "scene.pbrt" } );
}

TEST_F( TestOptions, missingDebugPixelY )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad debug pixel value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=384/", "scene.pbrt" } );
}

TEST_F( TestOptions, missingDebugPixelSeparator )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad debug pixel value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=384", "scene.pbrt" } );
}

TEST_F( TestOptions, missingDebugPixelX )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad debug pixel value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=", "scene.pbrt" } );
}

TEST_F( TestOptions, tooLargeDebugPixelX )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad debug pixel value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=384/128", "--dim=256x256", "scene.pbrt" } );
}

TEST_F( TestOptions, tooLargeDebugPixelY )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad debug pixel value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--debug=128/384", "--dim=256x256", "scene.pbrt" } );
}

TEST_F( TestOptions, defaultRenderModePrimaryRay )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt" } );

    EXPECT_EQ( demandPbrtScene::RenderMode::PRIMARY_RAY, options.renderMode );
}

TEST_F( TestOptions, renderModePrimaryRay )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "scene.pbrt", "--render-mode=primary" } );

    EXPECT_EQ( demandPbrtScene::RenderMode::PRIMARY_RAY, options.renderMode );
}

TEST_F( TestOptions, renderModeNearAmbientOcclusion )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--render-mode=near", "scene.pbrt" } );

    EXPECT_EQ( demandPbrtScene::RenderMode::NEAR_AO, options.renderMode );
}

TEST_F( TestOptions, renderModeDistantAmbientOcclusion )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--render-mode=distant" , "scene.pbrt"} );

    EXPECT_EQ( demandPbrtScene::RenderMode::DISTANT_AO, options.renderMode );
}

TEST_F( TestOptions, renderModePathTracing )
{
    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--render-mode=path" , "scene.pbrt"} );

    EXPECT_EQ( demandPbrtScene::RenderMode::PATH_TRACING, options.renderMode );
}

TEST_F( TestOptions, missingRenderMode )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "missing render mode value" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--render-mode=" , "scene.pbrt"} );
}

TEST_F( TestOptions, unknownRenderMode )
{
    EXPECT_CALL( m_mockUsage, Call( StrEq( "DemandPbrtScene" ), StrEq( "bad render mode value: foo" ) ) ).Times( 1 );

    const demandPbrtScene::Options options = getOptions( { "DemandPbrtScene", "--render-mode=foo" , "scene.pbrt"} );
}
