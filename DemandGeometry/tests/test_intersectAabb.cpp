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

#include <gtest/gtest.h>

#include <vector_functions.h>

#include "LaunchIntersectAabb.h"

static const OptixAabb unitCube{ 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f };

// Start at ray origin and extend for 100 units in ray direction.
static const float rayTMin{ 0.f };
static const float rayTMax{ 100.f };

// Vectors in the 6 directions
static const float3 posX{ make_float3( 1.0f, 0.0f, 0.0f ) };
static const float3 negX{ make_float3( -1.0f, 0.0f, 0.0f ) };
static const float3 posY{ make_float3( 0.0f, 1.0f, 0.0f ) };
static const float3 negY{ make_float3( 0.0f, -1.0f, 0.0f ) };
static const float3 posZ{ make_float3( 0.0f, 0.0f, 1.0f ) };
static const float3 negZ{ make_float3( 0.0f, 0.0f, -1.0f ) };

std::ostream& operator<<( std::ostream& str, const float3& val )
{
    return str << '(' << val.x << ", " << val.y << ", " << val.z << ')';
}

std::ostream& operator<<( std::ostream& str, const OptixAabb& val )
{
    return str << "{ min=(" << val.minX << ", " << val.minY << ", " << val.minZ << "), max=("
               << val.maxX << ", " << val.maxY << ", " << val.maxZ << ") }";
}

std::ostream& operator<<( std::ostream& os, const Intersection& data )
{
    return os << "{\n  rayOrigin=" << data.rayOrigin << "\n  rayDir=" << data.rayDir
              << "\n  rayTMin=" << data.rayTMin << "\n  rayTMax=" << data.rayTMax
              << "\n  aabb=" << data.aabb << "\n  tIntersect=" << data.tIntersect
              << "\n  normal=" << data.normal << "\n  face=" << data.face
              << "\n  intersected=" << ( data.intersected ? "yes" : "no" ) << "\n}";
}

// Set intersected to the opposite of intersects so we can verify that it is set correctly on return.
static Intersection intersects( const float3& origin, const float3& dir, bool intersects )
{
    Intersection intersection{};
    intersection.rayOrigin   = origin;
    intersection.rayDir      = dir;
    intersection.rayTMin     = rayTMin;
    intersection.rayTMax     = rayTMax;
    intersection.aabb        = unitCube;
    intersection.intersected = !intersects;
    return intersection;
}

static Intersection miss( const float3& origin, const float3& dir )
{
    return intersects( origin, dir, false );
}

TEST( IntersectAabb, pointsAwayFromNearXYFace )
{
    Intersection intersection{ miss( make_float3( 1.5f, 1.5f, -5.0f ), negZ ) };

    launchIntersectAabb( intersection );

    ASSERT_FALSE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAwayFromFarXYFace )
{
    Intersection intersection{ miss( make_float3( 1.5f, 1.5f, 5.0f ), posZ ) };

    launchIntersectAabb( intersection );

    ASSERT_FALSE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAwayFromLeftYZFace )
{
    Intersection intersection{ miss( make_float3( -5.0f, 1.5f, 1.5f ), negX ) };

    launchIntersectAabb( intersection );

    ASSERT_FALSE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAwayFromRightYZFace )
{
    Intersection intersection{ miss( make_float3( 5.0f, 1.5f, 1.5f ), posX ) };

    launchIntersectAabb( intersection );

    ASSERT_FALSE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAwayFromBottomXZFace )
{
    Intersection intersection{ miss( make_float3( 1.5f, -5.0f, 1.5f ), negY ) };

    launchIntersectAabb( intersection );

    ASSERT_FALSE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAwayFromTopXZFace )
{
    Intersection intersection{ miss( make_float3( 1.5f, 5.0f, 1.5f ), posY ) };

    launchIntersectAabb( intersection );

    ASSERT_FALSE( intersection.intersected ) << intersection;
}

static Intersection hit( const float3& origin, const float3& dir )
{
    return intersects( origin, dir, true );
}

TEST( IntersectAabb, pointsAtNearXYFace )
{
    Intersection intersection{ hit( make_float3( 1.5f, 1.5f, -5.0f ), posZ ) };

    launchIntersectAabb( intersection );

    EXPECT_EQ( 0, intersection.face ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.x ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.y ) << intersection;
    EXPECT_FLOAT_EQ( 1.0f, intersection.normal.z ) << intersection;
    EXPECT_TRUE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAtRightYZFace )
{
    Intersection intersection{ hit( make_float3( 5.0f, 1.5f, 1.5f ), negX ) };

    launchIntersectAabb( intersection );

    ASSERT_EQ( 1, intersection.face ) << intersection;
    EXPECT_FLOAT_EQ( 1.0f, intersection.normal.x ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.y ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.z ) << intersection;
    ASSERT_TRUE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAtFarXYFace )
{
    Intersection intersection{ hit( make_float3( 1.5f, 1.5f, 5.0f ), negZ ) };

    launchIntersectAabb( intersection );

    EXPECT_EQ( 2, intersection.face ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.x ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.y ) << intersection;
    EXPECT_FLOAT_EQ( -1.0f, intersection.normal.z ) << intersection;
    EXPECT_TRUE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAtLeftYZFace )
{
    Intersection intersection{ hit( make_float3( -5.0f, 1.5f, 1.5f ), posX ) };

    launchIntersectAabb( intersection );

    EXPECT_EQ( 3, intersection.face ) << intersection;
    EXPECT_FLOAT_EQ( -1.0f, intersection.normal.x ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.y ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.z ) << intersection;
    ASSERT_TRUE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAtBottomXZFace )
{
    Intersection intersection{ hit( make_float3( 1.5f, -5.0f, 1.5f ), posY ) };

    launchIntersectAabb( intersection );

    ASSERT_EQ( 4, intersection.face ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.x ) << intersection;
    EXPECT_FLOAT_EQ( -1.0f, intersection.normal.y ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.z ) << intersection;
    ASSERT_TRUE( intersection.intersected ) << intersection;
}

TEST( IntersectAabb, pointsAtTopXZFace )
{
    Intersection intersection{ hit( make_float3( 1.5f, 5.0f, 1.5f ), negY ) };

    launchIntersectAabb( intersection );

    ASSERT_EQ( 5, intersection.face ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.x ) << intersection;
    EXPECT_FLOAT_EQ( 1.0f, intersection.normal.y ) << intersection;
    EXPECT_FLOAT_EQ( 0.0f, intersection.normal.z ) << intersection;
    ASSERT_TRUE( intersection.intersected ) << intersection;
}
