// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// gtest has to come before pbrt stuff
#include <gtest/gtest.h>

#include <DemandPbrtScene/CameraMath.h>
#include <DemandPbrtScene/SceneAdapters.h>

#include <OptiXToolkit/ShaderUtil/vec_math.h>

namespace demandPbrtScene {
namespace {

void expectNear( const pbrt::Point3f& expected, const float3& actual, float tolerance = 1.0e-6f )
{
    EXPECT_NEAR( expected.x, actual.x, tolerance );
    EXPECT_NEAR( expected.y, actual.y, tolerance );
    EXPECT_NEAR( expected.z, actual.z, tolerance );
}

void expectNear( const float3& expected, const float3& actual, float tolerance = 1.0e-6f )
{
    EXPECT_NEAR( expected.x, actual.x, tolerance );
    EXPECT_NEAR( expected.y, actual.y, tolerance );
    EXPECT_NEAR( expected.z, actual.z, tolerance );
}

void expectScreenWindow( const float4& actual, float xMin, float xMax, float yMin, float yMax )
{
    EXPECT_FLOAT_EQ( xMin, actual.x );
    EXPECT_FLOAT_EQ( xMax, actual.y );
    EXPECT_FLOAT_EQ( yMin, actual.z );
    EXPECT_FLOAT_EQ( yMax, actual.w );
}

pbrt::Transform makeReferenceRasterToCamera( const pbrt::Transform& cameraToScreen,
                                             const float4&          screenWindow,
                                             const float2&          imageSize )
{
    const pbrt::Transform screenToRaster =
        pbrt::Scale( imageSize.x, imageSize.y, 1.0f )
        * pbrt::Scale( 1.0f / ( screenWindow.y - screenWindow.x ), 1.0f / ( screenWindow.z - screenWindow.w ), 1.0f )
        * pbrt::Translate( pbrt::Vector3f( -screenWindow.x, -screenWindow.w, 0.0f ) );
    return Inverse( cameraToScreen ) * Inverse( screenToRaster );
}

otk::Transform4 toTransform4( const pbrt::Transform& transform )
{
    otk::Transform4 result;
    toFloat4Transform( result.m, transform );
    return result;
}

pbrt::Transform landscapeCameraToWorld()
{
    const float           transform[16]{ 0.798635483f,  -0.0210030414f, 0.601448417f,  0.0f,
                                         0.0f,          0.999390841f,   0.0348994955f, 0.0f,
                                         -0.601815045f, -0.0278719775f, 0.79814899f,   0.0f,
                                         594.658691f,   -171.667648f,   3053.42725f,   1.0f };
    const pbrt::Matrix4x4 matrix( transform[0], transform[4], transform[8], transform[12],   //
                                  transform[1], transform[5], transform[9], transform[13],   //
                                  transform[2], transform[6], transform[10], transform[14],  //
                                  transform[3], transform[7], transform[11], transform[15] );
    return Inverse( pbrt::Transform( matrix ) );
}

TEST( TestCameraMath, defaultLandscapeScreenWindowUsesImageAspectRatio )
{
    const float4 screenWindow{ getPbrtScreenWindow( 16.0f / 9.0f, false, 0.0f, false, float4{} ) };

    expectScreenWindow( screenWindow, -16.0f / 9.0f, 16.0f / 9.0f, -1.0f, 1.0f );
}

TEST( TestCameraMath, defaultPortraitScreenWindowExpandsVertically )
{
    const float4 screenWindow{ getPbrtScreenWindow( 0.5f, false, 0.0f, false, float4{} ) };

    expectScreenWindow( screenWindow, -1.0f, 1.0f, -2.0f, 2.0f );
}

TEST( TestCameraMath, explicitFrameAspectRatioOverridesImageAspectRatio )
{
    const float4 screenWindow{ getPbrtScreenWindow( 16.0f / 9.0f, true, 1.5f, false, float4{} ) };

    expectScreenWindow( screenWindow, -1.5f, 1.5f, -1.0f, 1.0f );
}

TEST( TestCameraMath, explicitScreenWindowOverridesFrameAspectRatio )
{
    const float4 explicitWindow{ make_float4( -2.0f, 3.0f, -4.0f, 5.0f ) };
    const float4 screenWindow{ getPbrtScreenWindow( 16.0f / 9.0f, true, 1.5f, true, explicitWindow ) };

    EXPECT_EQ( explicitWindow, screenWindow );
}

TEST( TestCameraMath, centerRasterPointMatchesPbrtTransform )
{
    const float2          imageSize{ make_float2( 512.0f, 288.0f ) };
    const float4          screenWindow{ make_float4( -16.0f / 9.0f, 16.0f / 9.0f, -1.0f, 1.0f ) };
    const pbrt::Transform cameraToScreen{ pbrt::Perspective( 55.0f, 1.0e-2f, 1000.0f ) };
    const otk::Transform4 rasterToCamera{
        makePbrtRasterToCamera( toTransform4( cameraToScreen ), screenWindow, imageSize ) };
    const float3 filmPosition{ make_float3( 256.0f, 144.0f, 0.0f ) };

    const pbrt::Point3f expected{ makeReferenceRasterToCamera( cameraToScreen, screenWindow, imageSize )(
        pbrt::Point3f( filmPosition.x, filmPosition.y, filmPosition.z ) ) };
    expectNear( expected, transformPbrtPoint( rasterToCamera, filmPosition ) );
}

TEST( TestCameraMath, cornerRasterPointMatchesPbrtTransform )
{
    const float2          imageSize{ make_float2( 512.0f, 288.0f ) };
    const float4          screenWindow{ make_float4( -16.0f / 9.0f, 16.0f / 9.0f, -1.0f, 1.0f ) };
    const pbrt::Transform cameraToScreen{ pbrt::Perspective( 55.0f, 1.0e-2f, 1000.0f ) };
    const otk::Transform4 rasterToCamera{
        makePbrtRasterToCamera( toTransform4( cameraToScreen ), screenWindow, imageSize ) };
    const float3 filmPosition{ make_float3( 0.0f, 0.0f, 0.0f ) };

    const pbrt::Point3f expected{ makeReferenceRasterToCamera( cameraToScreen, screenWindow, imageSize )(
        pbrt::Point3f( filmPosition.x, filmPosition.y, filmPosition.z ) ) };
    expectNear( expected, transformPbrtPoint( rasterToCamera, filmPosition ) );
}

TEST( TestCameraMath, pointTransformPerformsProjectiveDivide )
{
    otk::Transform4 transform{ otk::identity() };
    transform.m[3].w = 2.0f;

    EXPECT_EQ( make_float3( 1.0f, 2.0f, 3.0f ), transformPbrtPoint( transform, make_float3( 2.0f, 4.0f, 6.0f ) ) );
}

TEST( TestCameraMath, cameraTranslationDoesNotChangeRayDirection )
{
    const float3 filmPosition{ make_float3( 1.0f, 2.0f, 3.0f ) };
    const PerspectiveCameraRay untranslated{
        makePbrtPerspectiveCameraRay( otk::identity(), otk::identity(), filmPosition ) };
    const PerspectiveCameraRay translated{
        makePbrtPerspectiveCameraRay( otk::translate( 10.0f, 20.0f, 30.0f ), otk::identity(), filmPosition ) };

    expectNear( untranslated.direction, translated.direction );
    EXPECT_EQ( make_float3( 10.0f, 20.0f, 30.0f ), translated.origin );
}

TEST( TestCameraMath, landscapeCenterRayMatchesPbrtCamera )
{
    const float2 imageSize{ make_float2( 512.0f, 288.0f ) };
    const float4 screenWindow{ getPbrtScreenWindow( imageSize.x / imageSize.y, false, 0.0f, false, float4{} ) };
    const otk::Transform4 cameraToWorld{ toTransform4( landscapeCameraToWorld() ) };
    const otk::Transform4 cameraToScreen{
        toTransform4( pbrt::Perspective( 16.46069f, 1.0e-2f, 1000.0f ) ) };
    const otk::Transform4 rasterToCamera{ makePbrtRasterToCamera( cameraToScreen, screenWindow, imageSize ) };

    const float3 centerFilmPosition{ make_float3( 0.5f * imageSize.x, 0.5f * imageSize.y, 0.0f ) };
    const PerspectiveCameraRay ray{
        makePbrtPerspectiveCameraRay( cameraToWorld, rasterToCamera, centerFilmPosition ) };

    expectNear( make_float3( -2315.0f, 65.0f, -2084.0f ), ray.origin, 1.0e-3f );
    expectNear( make_float3( 0.601448f, 0.0348995f, 0.798149f ), ray.direction, 1.0e-6f );
}

}  // namespace
}  // namespace demandPbrtScene
