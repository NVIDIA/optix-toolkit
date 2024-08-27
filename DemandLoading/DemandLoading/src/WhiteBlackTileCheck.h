//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/ImageSource/ImageHelpers.h>
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>
using namespace otk;

namespace demandLoading
{

// 0/1 tile types for float4, float2, float, half4, half2, half, uchar4, uchar2, uchar
enum WhiteBlackTileType { F4_0000=0, F4_0001, F4_1110, F4_1111, F2_00, F2_11, F_0, F_1,
                          H4_0000, H4_0001, H4_1110, H4_1111, H2_00, H2_11, H_0, H_1,
                          UB4_0000, UB4_0001, UB4_1110, UB4_1111, UB2_00, UB2_11, UB_0, UB_1,
                          WB_NONE };

// Check whether a tile has all uniform color
template<class TYPE> bool tileHasUniformColor( TYPE* tile, TYPE val )
{
    // Make a small array to compare against so memcmp is not called so many times.
    const int testGroupSize = 8;
    TYPE valBuff[testGroupSize] = {val, val, val, val, val, val, val, val};

    int numTests = TILE_SIZE_IN_BYTES / ( testGroupSize * sizeof( TYPE ) );
    for( int i = 0; i < numTests; ++i )
    {
        if( memcmp( &tile[i*testGroupSize], valBuff, testGroupSize * sizeof( TYPE ) ) )
            return false;
    }
    return true;
}

/// Check whether a tile is all white or all black for reuse.
inline WhiteBlackTileType classifyTileAsWhiteOrBlack( char* data, CUarray_format format, int numChannels )
{
    // float formats
    if( format == CU_AD_FORMAT_FLOAT )
    {
        if( numChannels == 4 && tileHasUniformColor<float4>( (float4*)data, float4{0,0,0,0} ) )
            return F4_0000;
        if( numChannels == 4 && tileHasUniformColor<float4>( (float4*)data, float4{0,0,0,1} ) )
            return F4_0001;
        if( numChannels == 4 && tileHasUniformColor<float4>( (float4*)data, float4{1,1,1,0} ) )
            return F4_1110;
        if( numChannels == 4 && tileHasUniformColor<float4>( (float4*)data, float4{1,1,1,1} ) )
            return F4_1111;
        if( numChannels == 2 && tileHasUniformColor<float2>( (float2*)data, float2{0,0} ) )
            return F2_00;
        if( numChannels == 2 && tileHasUniformColor<float2>( (float2*)data, float2{1,1} ) )
            return F2_11;
        if( numChannels == 1 && tileHasUniformColor<float>( (float*)data, 0.0f ) )
            return F_0;
        if( numChannels == 1 && tileHasUniformColor<float>( (float*)data, 1.0f ) )
            return F_1;
        return WB_NONE;
    }

    // half formats
    if( format == CU_AD_FORMAT_HALF )
    {
        if( numChannels == 4 && tileHasUniformColor<half4>( (half4*)data, half4{0.f,0.f,0.f,0.f} ) )
            return H4_0000;
        if( numChannels == 4 && tileHasUniformColor<half4>( (half4*)data, half4{0.f,0.f,0.f,1.f} ) )
            return H4_0001;
        if( numChannels == 4 && tileHasUniformColor<half4>( (half4*)data, half4{1.f,1.f,1.f,0.f} ) )
            return H4_1110;
        if( numChannels == 4 && tileHasUniformColor<half4>( (half4*)data, half4{1.f,1.f,1.f,1.f} ) )
            return H4_1111;
        if( numChannels == 2 && tileHasUniformColor<half2>( (half2*)data, half2{0.f,0.f} ) )
            return H2_00;
        if( numChannels == 2 && tileHasUniformColor<half2>( (half2*)data, half2{1.f,1.f} ) )
            return H2_11;
        if( numChannels == 1 && tileHasUniformColor<half>( (half*)data, (half)0.0f ) )
            return H_0;
        if( numChannels == 1 && tileHasUniformColor<half>( (half*)data, (half)1.0f ) )
            return H_1;
        return WB_NONE;
    }

    // uchar formats
    if( format == CU_AD_FORMAT_UNSIGNED_INT8 )
    {
        if( numChannels == 4 && tileHasUniformColor<uchar4>( (uchar4*)data, uchar4{0,0,0,0} ) )
            return UB4_0000;
        if( numChannels == 4 && tileHasUniformColor<uchar4>( (uchar4*)data, uchar4{0,0,0,255} ) )
            return UB4_0001;
        if( numChannels == 4 && tileHasUniformColor<uchar4>( (uchar4*)data, uchar4{255,255,255,0} ) )
            return UB4_1110;
        if( numChannels == 4 && tileHasUniformColor<uchar4>( (uchar4*)data, uchar4{255,255,255,255} ) )
            return UB4_1111;
        if( numChannels == 2 && tileHasUniformColor<uchar2>( (uchar2*)data, uchar2{0,0} ) )
            return UB2_00;
        if( numChannels == 2 && tileHasUniformColor<uchar2>( (uchar2*)data, uchar2{255,255} ) )
            return UB2_11;
        if( numChannels == 1 && tileHasUniformColor<uchar>( (uchar*)data, 0 ) )
            return UB_0;
        if( numChannels == 1 && tileHasUniformColor<uchar>( (uchar*)data, 255 ) )
            return UB_1;
        return WB_NONE;
    }

    return WB_NONE;
}

}  // namespace demandLoading
