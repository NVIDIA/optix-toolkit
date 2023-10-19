//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_fp16.h>

namespace imageSource {

#if defined( __CUDACC__ ) || defined( OPTIX_PAGING_BIT_OPS )
#define INLINE static __forceinline__ __device__
#else
#define INLINE inline
#endif

using ubyte = unsigned char;

struct ubyte4
{
    ubyte x, y, z, w;
};

struct ubyte2 
{
    ubyte x, y;
};

struct half4
{
    half x, y, z, w;
};

// clang-format off
INLINE unsigned int getNumChannels( float4& /*x*/ ) { return 4; }
INLINE unsigned int getNumChannels( float2& /*x*/ ) { return 2; }
INLINE unsigned int getNumChannels( float&  /*x*/ ) { return 1; }
INLINE unsigned int getNumChannels( half4&  /*x*/ ) { return 4; }
INLINE unsigned int getNumChannels( half2&  /*x*/ ) { return 2; }
INLINE unsigned int getNumChannels( half&   /*x*/ ) { return 1; }
INLINE unsigned int getNumChannels( ubyte4& /*x*/ ) { return 4; }
INLINE unsigned int getNumChannels( ubyte2& /*x*/ ) { return 2; }
INLINE unsigned int getNumChannels( ubyte&  /*x*/ ) { return 1; }
INLINE unsigned int getNumChannels( unsigned int& /*x*/ ) { return 1; }

INLINE CUarray_format_enum getFormat( float4& /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
INLINE CUarray_format_enum getFormat( float2& /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
INLINE CUarray_format_enum getFormat( float&  /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
INLINE CUarray_format_enum getFormat( half4&  /*x*/ ) { return CU_AD_FORMAT_HALF; }
INLINE CUarray_format_enum getFormat( half2&  /*x*/ ) { return CU_AD_FORMAT_HALF; }
INLINE CUarray_format_enum getFormat( half&   /*x*/ ) { return CU_AD_FORMAT_HALF; }
INLINE CUarray_format_enum getFormat( ubyte4& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
INLINE CUarray_format_enum getFormat( ubyte2& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
INLINE CUarray_format_enum getFormat( ubyte&  /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
INLINE CUarray_format_enum getFormat( unsigned int& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT32; }

INLINE unsigned char toUChar( float value)
{
    return static_cast<unsigned char>( 255.0f * value );
}
INLINE unsigned char toUChar( half value )
{
    return toUChar( static_cast<float>( value ) );
}
INLINE unsigned int toUInt( float value )
{
    return static_cast<unsigned int>( value );
}
INLINE unsigned int toUInt( half value )
{
    return static_cast<unsigned int>( value );
}

// clang-format off
INLINE void convertType( const float4& a, float4& b )           { b = a; }
INLINE void convertType( const float4& a, float2& b )           { b = float2{ a.x, a.y }; }
INLINE void convertType( const float4& a, float& b )            { b = a.x; }
INLINE void convertType( const float4& a, half4& b )            { b = half4{ a.x, a.y, a.z, a.w }; }
INLINE void convertType( const float4& a, half2& b )            { b = half2{ a.x, a.y }; }
INLINE void convertType( const float4& a, half& b )             { b = a.x; }
INLINE void convertType( const float4& a, uchar4& b )           { b = uchar4{ toUChar( a.x ), toUChar( a.y ), toUChar( a.z ), toUChar( a.w ) }; }
INLINE void convertType( const float4& a, uchar2& b )           { b = uchar2{ toUChar( a.x ), toUChar( a.y ) }; }
INLINE void convertType( const float4& a, unsigned char& b )    { b = toUChar( a.x ); }
INLINE void convertType( const float4& a, uint4& b )            { b = uint4{ toUInt( a.x ), toUInt( a.y ), toUInt( a.z ), toUInt( a.w ) }; }
INLINE void convertType( const float4& a, uint2& b )            { b = uint2{ toUInt( a.x ), toUInt( a.y ) }; }
INLINE void convertType( const float4& a, unsigned int& b )     { b = toUInt( a.x ); }

INLINE void convertType( const half4& a, float4& b )            { b = float4{ a.x, a.y, a.z, a.w }; }
INLINE void convertType( const half4& a, float2& b )            { b = float2{ a.x, a.y }; }
INLINE void convertType( const half4& a, float& b )             { b = a.x; }
INLINE void convertType( const half4& a, half4& b )             { b = a; }
INLINE void convertType( const half4& a, half2& b )             { b = half2{ a.x, a.y }; }
INLINE void convertType( const half4& a, half& b )              { b = a.x; }
INLINE void convertType( const half4& a, uchar4& b )            { b = uchar4{ toUChar( a.x ), toUChar( a.y ), toUChar( a.z ), toUChar( a.w )}; }
INLINE void convertType( const half4& a, uchar2& b )            { b = uchar2{ toUChar( a.x ), toUChar( a.y ) }; }
INLINE void convertType( const half4& a, unsigned char& b )     { b = toUChar( a.x ); }
INLINE void convertType( const half4& a, uint4& b )             { b = uint4{ toUInt( a.x ), toUInt( a.y ), toUInt( a.z ), toUInt( a.w ) }; }
INLINE void convertType( const half4& a, uint2& b )             { b = uint2{ toUInt( a.x ), toUInt( a.y ) }; }
INLINE void convertType( const half4& a, unsigned int& b )      { b = toUInt( a.x ); }
// clang-format on

}  // namespace imageSource
