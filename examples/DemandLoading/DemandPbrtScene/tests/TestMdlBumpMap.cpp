// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <gtest/gtest.h>

#include "DemandPbrtScene/MdlBumpMap.h"

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <vector_functions.h>

namespace demandPbrtScene {
namespace {

constexpr float TOLERANCE{ 1.0e-6f };

void expectNear( const float3& actual, const float3& expected )
{
    EXPECT_NEAR( expected.x, actual.x, TOLERANCE );
    EXPECT_NEAR( expected.y, actual.y, TOLERANCE );
    EXPECT_NEAR( expected.z, actual.z, TOLERANCE );
}

MdlBumpDifferentialGeometry unitSquareGeometry()
{
    const float3 vertices[3] = {
        make_float3( 0.0f, 0.0f, 0.0f ),
        make_float3( 1.0f, 0.0f, 0.0f ),
        make_float3( 0.0f, 1.0f, 0.0f ),
    };
    const float2 uvs[3] = {
        make_float2( 0.0f, 0.0f ),
        make_float2( 1.0f, 0.0f ),
        make_float2( 0.0f, 1.0f ),
    };
    const float3 normals[3] = {
        make_float3( 0.0f, 0.0f, 1.0f ),
        make_float3( 0.0f, 0.0f, 1.0f ),
        make_float3( 0.0f, 0.0f, 1.0f ),
    };
    return makeMdlBumpDifferentialGeometry( vertices, uvs, normals );
}

TEST( TestMdlBumpMap, calculatesSurfaceDerivativesFromPositionsAndUvs )
{
    const MdlBumpDifferentialGeometry geometry{ unitSquareGeometry() };

    expectNear( geometry.dpdu, make_float3( 1.0f, 0.0f, 0.0f ) );
    expectNear( geometry.dpdv, make_float3( 0.0f, 1.0f, 0.0f ) );
    expectNear( geometry.dndu, make_float3( 0.0f, 0.0f, 0.0f ) );
    expectNear( geometry.dndv, make_float3( 0.0f, 0.0f, 0.0f ) );
}

TEST( TestMdlBumpMap, usesPbrtTextureDifferentialOffsets )
{
    EXPECT_FLOAT_EQ( 0.25f, mdlBumpOffset( 0.2f, -0.3f ) );
    EXPECT_FLOAT_EQ( 0.0005f, mdlBumpOffset( 0.0f, 0.0f ) );
}

TEST( TestMdlBumpMap, constantHeightDoesNotChangeAFlatSurfaceNormal )
{
    const MdlBumpDifferentialGeometry geometry{ unitSquareGeometry() };
    const float3                      normal{ make_float3( 0.0f, 0.0f, 1.0f ) };

    for( const float height : { 0.0f, 0.5f, 1.0f } )
    {
        const MdlBumpDifferentialGeometry bumped{ applyMdlBumpMap( geometry, normal, height, height, height, 0.1f, 0.1f ) };
        expectNear( mdlBumpNormal( bumped, normal ), normal );
    }
}

TEST( TestMdlBumpMap, heightDerivativeTiltsNormalInUAndV )
{
    const MdlBumpDifferentialGeometry geometry{ unitSquareGeometry() };
    const float3                      normal{ make_float3( 0.0f, 0.0f, 1.0f ) };

    const MdlBumpDifferentialGeometry bumpedU{ applyMdlBumpMap( geometry, normal, 0.0f, 0.1f, 0.0f, 0.1f, 0.1f ) };
    const MdlBumpDifferentialGeometry bumpedV{ applyMdlBumpMap( geometry, normal, 0.0f, 0.0f, 0.1f, 0.1f, 0.1f ) };

    expectNear( mdlBumpNormal( bumpedU, normal ), otk::normalize( make_float3( -1.0f, 0.0f, 1.0f ) ) );
    expectNear( mdlBumpNormal( bumpedV, normal ), otk::normalize( make_float3( 0.0f, -1.0f, 1.0f ) ) );
}

TEST( TestMdlBumpMap, addingAConstantHeightDoesNotChangeAFlatSurfaceNormal )
{
    const MdlBumpDifferentialGeometry geometry{ unitSquareGeometry() };
    const float3                      normal{ make_float3( 0.0f, 0.0f, 1.0f ) };
    const MdlBumpDifferentialGeometry first{ applyMdlBumpMap( geometry, normal, 0.1f, 0.2f, 0.3f, 0.1f, 0.1f ) };
    const MdlBumpDifferentialGeometry shifted{ applyMdlBumpMap( geometry, normal, 0.6f, 0.7f, 0.8f, 0.1f, 0.1f ) };

    expectNear( mdlBumpNormal( shifted, normal ), mdlBumpNormal( first, normal ) );
}

}  // namespace
}  // namespace demandPbrtScene
