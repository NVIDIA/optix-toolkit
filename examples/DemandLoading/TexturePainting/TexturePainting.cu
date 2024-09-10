// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>
#include "TexturePaintingParams.h"

extern "C" {
__constant__ OTKAppLaunchParams params;
}

//------------------------------------------------------------------------------
// Overlay buttons and brush
//------------------------------------------------------------------------------

__device__ __forceinline__ float4 superellipse( uint2 px, int cx, int cy, int width, int height, float4 color, int exp )
{
    float dx = fabs( cx - static_cast<float>(px.x) ) / (width * 0.5f);
    float dy = fabs( cy - static_cast<float>(px.y) ) / (height * 0.5f);
    if( dx > 1.0f || dy > 1.0f )
        return float4{0.0f, 0.0f, 0.0f, 0.0f};
    float d2 = powf(dx, exp) + powf(dy, exp);
    if( d2 > 1.0f )
        return float4{0.0f, 0.0f, 0.0f, 0.0f};
    float4 rval = color;
    rval.w *= ( 1.0f - d2 * d2 );
    return rval;
}

__device__ __forceinline__ float4 overlayColor( uint2 px )
{
    TexturePaintingParams* tp = reinterpret_cast<TexturePaintingParams*>( params.extraData );

    // Draw buttons
    float4 rval = float4{0.0f, 0.0f, 0.0f, 0.0f};
    for( int i = 0; i < tp->numCanvases; ++i )
    {
        const int cx = BUTTON_SPACING * (i+1) + BUTTON_SIZE * i + BUTTON_SIZE / 2;
        const int cy = BUTTON_SPACING + BUTTON_SIZE / 2;
        const int bsize = ( i == tp->activeCanvas ) ? BUTTON_SIZE + 4 : BUTTON_SIZE;
        const float4 bcolor = ( i == tp->activeCanvas ) ? float4{ 0.3f, 0.0f, 0.0f, 1.0f } : float4{ 0.0f, 0.0f, 0.3f, 1.0f };
        rval += superellipse( px, cx, cy, bsize, bsize, bcolor, 6 );
    }

    // Draw brush
    const int cx = params.image_dim.x - BUTTON_SPACING / 2 - tp->brushWidth / 2;
    const int cy = BUTTON_SPACING / 2 + tp->brushHeight / 2;
    rval += superellipse( px, cx, cy, tp->brushWidth, tp->brushHeight, tp->brushColor, 2 );
    
    return rval;
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
        RayCone rayCone = initRayConeOrthoCamera( params.camera.U, params.camera.V, params.image_dim );
        float2 ddx = float2{rayCone.width, 0.0f};
        float2 ddy = float2{0.0f, rayCone.width};
        bool resident = false;
        color = tex2DGrad<float4>( params.demand_texture_context, params.display_texture_id, 
                                   origin.x, origin.y, ddx, ddy, &resident );
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
        color = tex2DGrad<float4>( params.demand_texture_context, params.display_texture_id, 
                                   g.uv.x, g.uv.y, g.ddx, g.ddy, &resident );
    }
    #endif

    // Blend color with tile overlay color, and put result in the output buffer
    float4 tcolor = overlayColor( px );
    color = ( 1.0f - tcolor.w ) * color + tcolor.w * tcolor;
    params.result_buffer[px.y * params.image_dim.x + px.x] = make_color( color );
}
