// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>
#include <stdio.h>
#include <gtest/gtest.h>
#include <vector_types.h>
#include <OptiXToolkit/ShaderUtil/ray_cone.h>

float EPS = 0.0001f;

class TestRayCone : public testing::Test
{
};

TEST_F( TestRayCone, projectToRayDifferentialsOnSurface_parallel )
{
    // Parallel ray and normal
    float rayConeWidth = 0.1f;
    float3 D = float3{0,0,-1};
    float3 N = float3{0,0,1};
    float3 dPdx, dPdy;
    float invMaxAnisotropy = 1.0f / 64.0f;
    projectToRayDifferentialsOnSurface( rayConeWidth, D, N, dPdx, dPdy, invMaxAnisotropy );

    // The differentials should have the same length as rayConeWidth
    EXPECT_NEAR( length( dPdx ), rayConeWidth, EPS );
    EXPECT_NEAR( length( dPdy ), rayConeWidth, EPS );

    // The differentials should be perpendicular to N
    EXPECT_NEAR( fabsf( dot( dPdx, N ) ), 0.0f, EPS );
    EXPECT_NEAR( fabsf( dot( dPdy, N ) ), 0.0f, EPS );

    // The differentials should be perpendicular to each-other
    EXPECT_NEAR( fabsf( dot( dPdx, dPdy ) ), 0.0f, EPS );
}

