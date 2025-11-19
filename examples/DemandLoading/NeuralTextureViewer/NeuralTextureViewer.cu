// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// Suppress deprecation warnings from CUDA vector types in ShaderUtil headers
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>

extern "C" {
    __constant__ OTKAppLaunchParams params;
}

#include <OptiXToolkit/NeuralTextures/InferenceOptix.h>
#include <OptiXToolkit/NeuralTextures/Texture2DNeural.h>
using namespace neuralTextures;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <class T_VEC_OUT>
static __forceinline__ __device__ float4 getSubtextureValue( NtcTextureSet* nts, T_VEC_OUT& out, unsigned int texNum )
{
    texNum = min( texNum, nts->numTextures - 1 );
    const int start = nts->texFirstChannel[texNum];
    const int numChannels = nts->texNumChannels[texNum];
    float4 color;
    color.x = out[start];
    color.y = (numChannels > 1) ? out[start+1] : 0.0f;
    color.z = (numChannels > 2) ? out[start+2] : 0.0f;
    color.w = (numChannels > 3) ? out[start+3] : 0.0f;
    return color;
}

//------------------------------------------------------------------------------
// OptiX programs
//------------------------------------------------------------------------------
 
extern "C" __global__ void __raygen__rg()
{
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) )
        return;

    // Make eye ray
    float3 origin, direction;
    makeEyeRayOrthographic( params.camera, params.image_dim, float2{px.x+0.5f, px.y+0.5f}, origin, direction );
    float4 color = params.background_color;

    // Don't cast the ray. Calculate texture coords directly from it.
    if( origin.x >= 0.0f && origin.x <= 1.0f && origin.y >= 0.0f && origin.y <= 1.0f )
    {
        // Compute texture coordinates
        RayCone rayCone = initRayConeOrthoCamera( params.camera.U, params.camera.V, params.image_dim );
        float scale = params.extraData[0];
        float2 uv = float2{origin.x, origin.y} * scale;
        float2 ddx = float2{rayCone.width, 0.0f} * scale;
        float2 ddy = float2{0.0f, rayCone.width} * scale;
        
        unsigned int seed = srand( px.x, px.y, params.subframe );
        float2 xi = rnd2( seed );

        // Sample the texture set
        T_VEC_OUT_FLOAT out;
        const unsigned int textureId = params.display_texture_id;
        const DeviceContext dtContext = params.demand_texture_context;

        // Extract the current texture value
        unsigned int texNum = params.render_mode - 1u;
        NtcTextureSet* nts;
        bool resident = ntcTex2DGradUdim<T_VEC_OUT_FLOAT>( out, nts, dtContext, textureId, uv.x, uv.y, ddx, ddy, xi );
        color = ( resident ) ? getSubtextureValue( nts, out, texNum ) : float4{0.2f, 0.0f, 0.0f, 0.0f};
    }

    // Accumulate sample in the accumulation buffer
    unsigned int image_idx = px.y * params.image_dim.x + px.x;
    float4 accum_color = float4{color.x, color.y, color.z, 1.0f};
    if( params.subframe != 0 )
        accum_color += params.accum_buffer[image_idx];
    params.accum_buffer[image_idx] = accum_color;

    // Blend result of ray trace with tile display and put in result buffer
    accum_color *= ( 1.0f / accum_color.w );
    float4 tcolor = tileDisplayColor( params.demand_texture_context, params.display_texture_id, 10, 10, px );
    color  = ( 1.0f - tcolor.w ) * accum_color + tcolor.w * tcolor;
    params.result_buffer[image_idx] = make_color( color );
}
