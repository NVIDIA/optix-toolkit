// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>

#include <OptiXToolkit/DemandLoading/Texture2DCubic.h>

extern "C" {
__constant__ OTKAppLaunchParams params;
}

//------------------------------------------------------------------------------
// OptiX programs
//------------------------------------------------------------------------------
 
//#define CAST_RAYS // uncomment to turn on ray casting (slower)

extern "C" __global__ void __raygen__rg()
{
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) )
        return;

    // Make eye ray
    float3 origin, direction;
    makeEyeRayOrthographic( params.camera, params.image_dim, float2{px.x+0.5f, px.y+0.5f}, origin, direction );
    float4 color = params.background_color;

    // Don't cast a ray. Calculate texture coords directly.
    #ifndef CAST_RAYS
    if( origin.x >= 0.0f && origin.x <= 1.0f && origin.y >= 0.0f && origin.y <= 1.0f )
    {
        float2* uvdim = reinterpret_cast<float2*>( &params.extraData );
        RayCone rayCone = initRayConeOrthoCamera( params.camera.U, params.camera.V, params.image_dim );

        float u = origin.x * uvdim->x;
        float v = origin.y * uvdim->y;
        float2 ddx = float2{rayCone.width * uvdim->x, 0.0f};
        float2 ddy = float2{0.0f, rayCone.width * uvdim->y};
        bool resident = false;

        // Standard bilinear filtering
        color = tex2DGradUdimBlend<float4>( params.demand_texture_context, params.display_texture_id,
                                            u, v, ddx, ddy, &resident );

        // Cubic filtering
        //resident = textureUdim<float4>( params.demand_texture_context, params.display_texture_id,
        //                                u, v, ddx, ddy, &color, nullptr, nullptr, float2{0.0f, 0.0f} );
    }
    #endif

    // Cast a ray to hit the square
    #ifdef CAST_RAYS
    OTKAppRayPayload prd{};
    prd.rayCone1 = initRayConeOrthoCamera( params.camera.U, params.camera.V, params.image_dim );
    prd.rayCone2 = prd.rayCone1;
    traceRay( params.traversable_handle, origin, direction, 0.0f, 1e16f, OPTIX_RAY_FLAG_NONE, &prd );

    if( prd.occluded ) // The square was hit
    {
        bool resident = false;
        const SurfaceGeometry& g = prd.geometry;
        color = tex2DGradUdimBlend<float4>( params.demand_texture_context, params.display_texture_id,
                                            g.uv.x, g.uv.y, g.ddx, g.ddy, &resident );
    }
    #endif

    // Put color in the output buffer
    params.result_buffer[px.y * params.image_dim.x + px.x] = make_color( color );
}
