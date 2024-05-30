//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "../../src/Util/VecMath.h"
#include "OptiXKernels.h"

extern "C" {
__constant__ Params params;
}

__forceinline__ __device__ float3 toSRGB( const float3& c )
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3( c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f, c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f, c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
{
    x = clamp( x, 0.0f, 1.0f );
    enum
    {
        N   = ( 1 << 8 ) - 1,
        Np1 = ( 1 << 8 )
    };
    return (unsigned char)min( (unsigned int)( x * (float)Np1 ), (unsigned int)N );
}

__forceinline__ __device__ uchar3 make_color( const float3& c )
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( make_float3( c.x, c.y, c.z ) );
    return make_uchar3( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ) );
}

__device__ unsigned int getLaneMaskLt( void )
{
    unsigned int r;
    asm( "mov.u32 %0, %lanemask_lt;" : "=r"( r ) );
    return r;
}

extern "C" __global__ void __raygen__rg()
{
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

    // shoot off-center so we don't report precision issues on the edge of texture filters as visibility mismatches
    float x = (float)(idx.x + 0.499f) / (float)dim.x;
    float y = (float)(idx.y + 0.501f) / (float)dim.x;

    float3       origin    = make_float3( 
        params.options.windowMin.x + x * ( params.options.windowMax.x - params.options.windowMin.x), 
        params.options.windowMin.y + y * ( params.options.windowMax.y - params.options.windowMin.y ), 1.0f );
    float3       direction = make_float3( 0.0f, 0.0f, -1.0f );
    float        tmin      = 0.0f;
    float        tmax      = 100.0f;
    float        rayTime   = 0.0f;
    unsigned int rayFlags  = params.options.force2state ? ( OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE ) : OPTIX_RAY_FLAG_NONE;
    unsigned int SBTstride = 2;

    unsigned int pr = 0, pg = 0, pb = 0, ppi = 0, pii = 0;
    optixTrace( params.handle, origin, direction, tmin, tmax, rayTime, /*visibilityMask=*/VISIBILITY_MASK_OMM_ENABLED, rayFlags, /*SBToffset=*/0, SBTstride, /*missSBTIndex=*/0, pr, pg, pb, ppi, pii );

    // verrify that the OOM is conservative and rendering without OMM results in identical nearest hit.
    if( params.options.validate_opacity )
    {
        unsigned int ppi_ref = 0, pii_ref = 0;
        optixTrace( params.handle, origin, direction, tmin, tmax, rayTime, /*visibilityMask=*/VISIBILITY_MASK_OMM_DISABLED, rayFlags, /*SBToffset=*/1, SBTstride, /*missSBTIndex=*/1, ppi_ref, pii_ref );

        if( ppi_ref != ppi || pii_ref != pii )
        {
            const unsigned int activeMask = __activemask();

            const unsigned int ltMask = ( activeMask & getLaneMaskLt() );
            if( ltMask == 0 )
            {
                unsigned int count = __popc( activeMask );
                atomicAdd( params.error_count, count );
            }

            // flag broken pixels red
            pr = __float_as_int( 1.f );
            pg = pb = 0;
        }
    }

    uint3 launch_index                                                 = optixGetLaunchIndex();
    params.image[launch_index.y * params.image_width + launch_index.x] = make_color( { __int_as_float( pr ), __int_as_float( pg ), __int_as_float( pb ) } );
}

__device__ float2 getTextureUV()
{
    HitSbtData* hitData = (HitSbtData*)optixGetSbtDataPointer();

    uint32_t pidx = optixGetPrimitiveIndex();

    uint3 idx3;

    switch( hitData->desc.indexFormat )
    {
        break;
        case cuOmmBaking::IndexFormat::NONE:
            idx3 = make_uint3( 3 * pidx + 0, 3 * pidx + 1, 3 * pidx + 2 );
        break;
        case cuOmmBaking::IndexFormat::I16_UINT: {
            uint32_t stride = ( hitData->desc.indexTripletStrideInBytes ? hitData->desc.indexTripletStrideInBytes : sizeof( ushort3 ) );
            ushort3  sidx3  = *reinterpret_cast<ushort3*>( hitData->desc.indexBuffer + stride * pidx );
            idx3            = { sidx3.x, sidx3.y, sidx3.z };
        }
        break;
        case cuOmmBaking::IndexFormat::I32_UINT: {
            uint32_t stride = ( hitData->desc.indexTripletStrideInBytes ? hitData->desc.indexTripletStrideInBytes : sizeof( uint3 ) );
            idx3            = *reinterpret_cast<uint3*>( hitData->desc.indexBuffer + stride * pidx );
        }
    };

    float2 uv0, uv1, uv2;
    switch( hitData->desc.texCoordFormat )
    {
        break;
        case cuOmmBaking::TexCoordFormat::UV32_FLOAT2: {
            uint32_t stide = hitData->desc.texCoordStrideInBytes ? hitData->desc.texCoordStrideInBytes : sizeof( float2 );
            uv0            = *reinterpret_cast<float2*>( hitData->desc.texCoordBuffer + stide * idx3.x );
            uv1            = *reinterpret_cast<float2*>( hitData->desc.texCoordBuffer + stide * idx3.y );
            uv2            = *reinterpret_cast<float2*>( hitData->desc.texCoordBuffer + stide * idx3.z );
        }
    };

    float2 barys = optixGetTriangleBarycentrics();
    float2 uv    = { ( 1.0f - barys.x - barys.y ) * uv0.x + barys.x * uv1.x + barys.y * uv2.x, ( 1.0f - barys.x - barys.y ) * uv0.y + barys.x * uv1.y + barys.y * uv2.y };

    if( hitData->desc.transformFormat == cuOmmBaking::UVTransformFormat::MATRIX_FLOAT2X3 )
    {
        const float* transform = (const float*)hitData->desc.transform;
        uv                     = { transform[0] * uv.x + transform[1] * uv.y + transform[2], transform[3] * uv.x + transform[4] * uv.y + transform[5] };
    }

    return uv;
}

inline __device__ float4 make_color4( uint4 v )
{
    return make_float4( (float)v.x, ( float )v.y, ( float )v.z, ( float )v.w );
}

inline __device__ float4 make_color4( int4 v )
{
    return make_float4( ( float )v.x, ( float )v.y, ( float )v.z, ( float )v.w );
}

inline __device__ float rscale( uint32_t bits )
{
    return ( 1.f / ( ( 1llu << bits ) - 1 ) );
}

inline __device__ float4 eval()
{
    float2 uv = getTextureUV();

    HitSbtData* hitData = (HitSbtData*)optixGetSbtDataPointer();

    float4 color = {};
    if( hitData->texture.tex )
    {
        if( params.options.textureLayer )
        {
            if( hitData->texture.readMode == cudaReadModeNormalizedFloat || hitData->texture.chanDesc.f == cudaChannelFormatKindFloat )
                color = tex2DLayered<float4>( hitData->texture.tex, uv.x, uv.y, params.options.textureLayer );
            else if( hitData->texture.chanDesc.f == cudaChannelFormatKindUnsigned )
                color = make_color4( tex2DLayered<uint4>( hitData->texture.tex, uv.x, uv.y, params.options.textureLayer ) );
            else
                color = make_color4( tex2DLayered<int4>( hitData->texture.tex, uv.x, uv.y, params.options.textureLayer ) );
        }
        else
        {
            if( hitData->texture.readMode == cudaReadModeNormalizedFloat || hitData->texture.chanDesc.f == cudaChannelFormatKindFloat )
                color = tex2D<float4>( hitData->texture.tex, uv.x, uv.y );
            else if( hitData->texture.chanDesc.f == cudaChannelFormatKindUnsigned )
                color = make_color4( tex2D<uint4>( hitData->texture.tex, uv.x, uv.y ) );
            else
                color = make_color4( tex2D<int4>( hitData->texture.tex, uv.x, uv.y ) );
        }
    }

    float4 alpha = color;

    if( hitData->texture.readMode != cudaReadModeNormalizedFloat )
    {
        if( hitData->texture.chanDesc.f == cudaChannelFormatKindUnsigned )
        {
            color.x *= rscale( hitData->texture.chanDesc.x );
            color.y *= rscale( hitData->texture.chanDesc.y );
            color.z *= rscale( hitData->texture.chanDesc.z );
            color.w *= rscale( hitData->texture.chanDesc.w );
        }
        else if( hitData->texture.chanDesc.f == cudaChannelFormatKindSigned )
        {
            color.x *= rscale( hitData->texture.chanDesc.x - 1 );
            color.y *= rscale( hitData->texture.chanDesc.y - 1 );
            color.z *= rscale( hitData->texture.chanDesc.z - 1 );
            color.w *= rscale( hitData->texture.chanDesc.w - 1 );
        }
    }

    cuOmmBaking::CudaTextureAlphaMode alphaMode = hitData->texture.alphaMode;

    if( alphaMode == cuOmmBaking::CudaTextureAlphaMode::DEFAULT )
    {
        if( hitData->texture.chanDesc.w )
            alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W;
        else if( hitData->texture.chanDesc.z )
            alphaMode = cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY;
        else if( hitData->texture.chanDesc.y )
            alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y;
        else
            alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X;
    }

    switch( alphaMode )
    {
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X:
        color.w = alpha.x;
        break;
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y:
        color.w = alpha.y;
        break;
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z:
        color.w = alpha.z;
        break;
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W:
        color.w = alpha.w;
        break;
    case cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY:
        color.w = ( alpha.x + alpha.y + alpha.z) / 3;
        break;
    }

    return color;
}

extern "C" __global__ void __anyhit__validate__ah()
{
    float4 color = eval();

    HitSbtData* hitData = ( HitSbtData* )optixGetSbtDataPointer();

    if( color.w <= hitData->texture.transparencyCutoff )
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__ah()
{
    unsigned int unkown_opaque      = optixGetPayload_0();
    unsigned int unkown_alpha       = optixGetPayload_1();
    unsigned int unkown_transparent = optixGetPayload_2();

    float4 color = eval();

    HitSbtData* hitData = ( HitSbtData* )optixGetSbtDataPointer();

    bool ignore = false;
    if( color.w <= hitData->texture.transparencyCutoff )
    {
        unkown_transparent++;
        ignore = true;
    }
    else if( color.w >= hitData->texture.opacityCutoff )
    {
        unkown_opaque++;
    }
    else
    {
        unkown_alpha++;
    }

    optixSetPayload_0( unkown_opaque );
    optixSetPayload_1( unkown_alpha );
    optixSetPayload_2( unkown_transparent );

    if( ignore )
        optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__validate__ch()
{
    optixSetPayload_0( optixGetPrimitiveIndex() );
    optixSetPayload_1( optixGetInstanceIndex() / 2 );  // instances come in OMM on/off pairs
}

extern "C" __global__ void __closesthit__ch()
{
    float4 color = eval();

    const unsigned int unkown_opaque      = optixGetPayload_0();
    const unsigned int unkown_alpha       = optixGetPayload_1();
    const unsigned int unkown_transparent = optixGetPayload_2();

    const unsigned int sum = unkown_opaque + unkown_alpha + unkown_transparent;

    float4 shade = {};

    if( params.options.opacity_shading )
    {
        if( sum )
        {
            if( sum == unkown_opaque )
            {
                shade = { 246.f / 255.f, 213.f / 255.f, 92.f / 255.f, 1.f };
            }
            else if( sum == unkown_alpha )
            {
                shade = { 60.f / 255.f, 174.f / 255.f, 163.f / 255.f, 1.f };
            }
            else if( sum == unkown_transparent )
            {
                shade = { 32 / 255.f, 99 / 255.f, 155 / 255.f, 1.f };
            }
            else  // mixed
            {
                shade = { 1.f, 1.f, 1.f, 1.f };
            }
        }
        else  // opaque
        {
            shade = { 237.f / 255.f, 85.f / 255.f, 59.f / 255.f, 1.f };
        }

        shade = ( 0.5f + ( color.x + color.y + color.z ) / 6.f ) * shade;
    }
    else
    {
        shade = color;
    }

    optixSetPayload_0( __float_as_int( shade.x ) );
    optixSetPayload_1( __float_as_int( shade.y ) );
    optixSetPayload_2( __float_as_int( shade.z ) );

    optixSetPayload_3( optixGetPrimitiveIndex() );
    optixSetPayload_4( optixGetInstanceIndex() / 2 );  // instances come in OMM on/off pairs
}

extern "C" __global__ void __miss__validate__ms()
{
    optixSetPayload_0( ~0u );
    optixSetPayload_1( ~0u );
}

extern "C" __global__ void __miss__ms()
{
    const unsigned int unkown_opaque      = optixGetPayload_0();
    const unsigned int unkown_alpha       = optixGetPayload_1();
    const unsigned int unkown_transparent = optixGetPayload_2();

    const unsigned int sum = unkown_opaque + unkown_alpha + unkown_transparent;

    float4 shade = {};

    if( params.options.opacity_shading )
    {
        if( sum )
        {
            if( sum == unkown_opaque )
            {
                shade = { 246.f / 255.f, 213.f / 255.f, 92.f / 255.f, 1.f };
            }
            else if( sum == unkown_alpha )
            {
                shade = { 60.f / 255.f, 174.f / 255.f, 163.f / 255.f, 1.f };
            }
            else if( sum == unkown_transparent )
            {
                shade = { 32.f / 255.f, 99.f / 255.f, 155.f / 255.f, 1.f };
            }
            else  // mixed
            {
                shade = { 1.f, 1.f, 1.f, 1.f };
            }
        }
        else  // transparent
        {
            shade = { 23.f / 255.f, 63.f / 255.f, 95.f / 255.f, 1.f };
        }
    }

    optixSetPayload_0( __float_as_int( shade.x ) );
    optixSetPayload_1( __float_as_int( shade.y ) );
    optixSetPayload_2( __float_as_int( shade.z ) );

    optixSetPayload_3( ~0u );
    optixSetPayload_4( ~0u );
}
