// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>

using namespace otkApp;

extern "C" __global__ void __miss__OTKApp()
{
    getRayPayload()->occluded = false;
}

extern "C" __global__ void __closesthit__OTKApp()
{
    OTKAppRayPayload* payload = getRayPayload();
    payload->occluded = true;
    //if( isOcclusionRay() ) // for occlusion query, just return
    //    return;

    // Get hit info
    const OTKAppTriangleHitGroupData* hg_data = reinterpret_cast<OTKAppTriangleHitGroupData*>( optixGetSbtDataPointer() );
    const int vidx = optixGetPrimitiveIndex() * 3;
    const float3 bary = getTriangleBarycentrics();
    const float3 D = optixGetWorldRayDirection();
    const float rayDistance = optixGetRayTmax();

    // Get triangle geometry
    float4* vertices = &hg_data->vertices[vidx];
    float3& Va = *reinterpret_cast<float3*>( &vertices[0] );
    float3& Vb = *reinterpret_cast<float3*>( &vertices[1] );
    float3& Vc = *reinterpret_cast<float3*>( &vertices[2] );
    const float3* normals = &hg_data->normals[vidx];
    const float2* tex_coords = &hg_data->tex_coords[vidx];

    // Compute Surface geometry for hit point
    SurfaceGeometry& geom = payload->geometry;
    geom.P = bary.x * Va + bary.y * Vb + bary.z * Vc; 
    geom.Ng = normalize( cross( Vb-Va, Vc-Va ) ); //geometric normal
    geom.N = normalize( bary.x * normals[0] + bary.y * normals[1] + bary.z * normals[2] ); // shading normal
    makeOrthoBasis( geom.N, geom.S, geom.T );
    geom.uv = bary.x * tex_coords[0] + bary.y * tex_coords[1] + bary.z * tex_coords[2];
    geom.curvature = minTriangleCurvature( Va, Vb, Vc, normals[0], normals[1], normals[2] );

    // Flip normal for local geometry if needed
    geom.flipped = false;
    if( dot( D, geom.Ng ) > 0.0f )
    {
        geom.Ng *= -1.0f;
        geom.N *= -1.0f;
        geom.curvature *= -1.0f;
        geom.flipped = true;
    }

    // Must be done before propagating the ray cones, since sign of cone angles may change
    bool pinholeSpecialCase = (payload->rayDepth == 0) && (payload->rayCone1.angle >= 0);

    // Propagate the ray cone
    float3 dPdx, dPdy;
    payload->rayCone1 = propagate( payload->rayCone1, rayDistance );
    payload->rayCone2 = propagate( payload->rayCone2, rayDistance );

    // Compute ray differentials on surface
    float rayConeWidth = fabsf( payload->rayCone1.width );
    if( !pinholeSpecialCase )
        rayConeWidth = maxf( rayConeWidth, fabsf( payload->rayCone2.width ) );
    projectToRayDifferentialsOnSurface( rayConeWidth, D, geom.N, dPdx, dPdy );
    
    // Compute texture gradients for triangle
    computeTexGradientsForTriangle( Va, Vb, Vc, tex_coords[0], tex_coords[1], tex_coords[2], dPdx, dPdy, geom.ddx, geom.ddy );

    // Get the surface texture
    payload->material = (void*)&hg_data->tex;
}
