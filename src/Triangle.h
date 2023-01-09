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

#pragma once

#include "CuOmmBakingImpl.h"
#include "Util/XXH.h"
#include "Util/VecMath.h"

namespace {

    __device__ bool operator!=( float2 a0, float2 a1 )
    {
        return a0.x != a1.x || a0.y != a1.y;
    }
};

struct Triangle
{
    __device__ bool operator!=( Triangle key )
    {
        return uv0 != key.uv0 || uv1 != key.uv1 || uv2 != key.uv2 || texture != key.texture;
    }

    __device__ float Area() const
    {
        const float2 e0 = uv1 - uv0;
        const float2 e1 = uv2 - uv0;

        return 0.5f * fabsf( e0.x * e1.y - e0.y * e1.x );
    }

    float2   uv0, uv1, uv2;
    uint64_t texture;
};

__device__ uint3 loadIndices( const cuOmmBaking::BakeInputDesc& inputDesc, unsigned int index )
{
    const void* indexPtr = ( const char* )inputDesc.indexBuffer + index * inputDesc.indexTripletStrideInBytes;

    uint3 idx3;
    if( inputDesc.indexFormat == cuOmmBaking::IndexFormat::I8_UINT )
    {
        uchar3 b3 = *( ( const uchar3* )indexPtr );
        idx3 = { b3.x, b3.y, b3.z };
    }
    else if( inputDesc.indexFormat == cuOmmBaking::IndexFormat::I16_UINT )
    {
        ushort3 s3 = *( ( const ushort3* )indexPtr );
        idx3 = { s3.x, s3.y, s3.z };
    }
    else if( inputDesc.indexFormat == cuOmmBaking::IndexFormat::I32_UINT )
    {
        idx3 = *( ( const uint3* )indexPtr );
    }
    else  // cuOmmBaking::IndexFormat::NONE
    {
        idx3 = make_uint3( 3 * index, 3 * index + 1, 3 * index + 2 );
    }

    return idx3;
}

__device__ float2 loadTexcoord( const cuOmmBaking::BakeInputDesc& inputDesc, unsigned int index )
{
    const float* texcoord = ( const float* )( ( const char* )inputDesc.texCoordBuffer + index * inputDesc.texCoordStrideInBytes );
    return make_float2( texcoord[0], texcoord[1] );
}

__device__ Triangle loadTriangle( const cuOmmBaking::BakeInputDesc& inputDesc, unsigned int index )
{
    Triangle tri;

    uint3 idx3 = loadIndices( inputDesc, index );

    tri.uv0 = loadTexcoord( inputDesc, idx3.x );
    tri.uv1 = loadTexcoord( inputDesc, idx3.y );
    tri.uv2 = loadTexcoord( inputDesc, idx3.z );

    if( inputDesc.transform )
    {
        float2 m0 = ( ( const float2* )inputDesc.transform )[0];
        float2 m1 = ( ( const float2* )inputDesc.transform )[1];
        float2 m2 = ( ( const float2* )inputDesc.transform )[2];

        tri.uv0 = make_float2( tri.uv0.x * m0.x + tri.uv0.y * m0.y + m1.x, tri.uv0.x * m1.y + tri.uv0.y * m2.x + m2.y );
        tri.uv1 = make_float2( tri.uv1.x * m0.x + tri.uv1.y * m0.y + m1.x, tri.uv1.x * m1.y + tri.uv1.y * m2.x + m2.y );
        tri.uv2 = make_float2( tri.uv2.x * m0.x + tri.uv2.y * m0.y + m1.x, tri.uv2.x * m1.y + tri.uv2.y * m2.x + m2.y );
    }

    tri.texture = 0;
    if( inputDesc.textureIndexBuffer )
    {
        const char* addr = ( ( const char* )inputDesc.textureIndexBuffer + inputDesc.textureIndexStrideInBytes * index );

        switch( inputDesc.textureIndexFormat )
        {
        case cuOmmBaking::IndexFormat::I8_UINT:
            tri.texture = *( const unsigned char* )addr;
            break;
        case cuOmmBaking::IndexFormat::I16_UINT:
            tri.texture = *( const unsigned short* )addr;
        break;
        case cuOmmBaking::IndexFormat::I32_UINT:
            tri.texture = *( const unsigned int* )addr;
        break;
        default:
            tri.texture = 0;
        }
    }

    return tri;
}

// convert uv to quantized (integer) representation
__device__ float2 quantizeUV( float2 uv, const TextureInput& textureInput )
{
    float2 f = textureInput.quantizationFrequency;

    float x = uv.x, y = uv.y;
    if( f.x )
        x = floorf( uv.x * f.x + 0.5f );
    if( f.y )
        y = floorf( uv.y * f.y + 0.5f );

    return { x, y };
}

// snap uv to the nearest quantizable value
__device__ float2 snapUV( float2 uv, const TextureInput& textureInput )
{
    float2 f = textureInput.quantizationFrequency;
    float2 p = textureInput.quantizationPeriod;

    float x = uv.x, y = uv.y;
    if( f.x )
        x = p.x * floorf( uv.x * f.x + 0.5f );
    if( f.y )
        y = p.y * floorf( uv.y * f.y + 0.5f );

    return { x, y };
}

// unwrap and quantize triangle coordinates
__device__ Triangle canonicalizeTriangle( Triangle in, const TextureInput* textureInputs )
{
    const TextureInput& textureInput = textureInputs[in.texture];

    // quantize the UVs for near-duplicate matching
    const float2 uv0 = quantizeUV( in.uv0, textureInput );
    const float2 uv1 = quantizeUV( in.uv1, textureInput );
    const float2 uv2 = quantizeUV( in.uv2, textureInput );

    // shift the quantized coordinates by half a unit before canonicalize the period to prevent precision issues around zero
    float baseU = 0.f;
    if( textureInput.quantizedPeriod.x )
        baseU = textureInput.quantizedPeriod.x * floorf( textureInput.quantizedFrequency.x * ( uv0.x + 0.5f ) );

    // shift the quantized coordinates by half a unit before canonicalize the period to prevent precision issues around zero
    float baseV = 0.f;
    if( textureInput.quantizedPeriod.y )
        baseV = textureInput.quantizedPeriod.y * floorf( textureInput.quantizedFrequency.y * ( uv0.y + 0.5f ) );

    Triangle out;

    out.uv0.x = uv0.x - baseU;
    out.uv1.x = uv1.x - baseU;
    out.uv2.x = uv2.x - baseU;

    out.uv0.y = uv0.y - baseV;
    out.uv1.y = uv1.y - baseV;
    out.uv2.y = uv2.y - baseV;

    out.texture = textureInput.data.id;

    return out;
}

__device__ uint32_t hash( Triangle key )
{
    const uint32_t data[8] = { __float_as_uint( key.uv0.x ), __float_as_uint( key.uv0.y ), __float_as_uint( key.uv1.x ), __float_as_uint( key.uv1.y ), __float_as_uint( key.uv2.x ), __float_as_uint( key.uv2.y ), ( uint32_t )( key.texture & 0xFFFFFFFF ), ( uint32_t )( key.texture >> 32 ) };

    return XXH( { data[0], data[1], data[2], data[3] }, { data[4], data[5], data[6], data[7] } );
}