// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_fp16.h>
#include <vector_types.h>

struct half4
{
    half x, y, z, w;
};

using uchar = unsigned char;

namespace imageSource {

#if defined( __CUDACC__ ) || defined( OPTIX_PAGING_BIT_OPS )
#define INLINE static __forceinline__ __device__
#else
#define INLINE inline
#endif

// clang-format off
INLINE unsigned int getNumChannels( float4& /*x*/ ) { return 4; }
INLINE unsigned int getNumChannels( float2& /*x*/ ) { return 2; }
INLINE unsigned int getNumChannels( float&  /*x*/ ) { return 1; }
INLINE unsigned int getNumChannels( half4&  /*x*/ ) { return 4; }
INLINE unsigned int getNumChannels( half2&  /*x*/ ) { return 2; }
INLINE unsigned int getNumChannels( half&   /*x*/ ) { return 1; }
INLINE unsigned int getNumChannels( uchar4& /*x*/ ) { return 4; }
INLINE unsigned int getNumChannels( uchar2& /*x*/ ) { return 2; }
INLINE unsigned int getNumChannels( uchar&  /*x*/ ) { return 1; }
INLINE unsigned int getNumChannels( unsigned int& /*x*/ ) { return 1; }

INLINE CUarray_format_enum getFormat( float4& /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
INLINE CUarray_format_enum getFormat( float2& /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
INLINE CUarray_format_enum getFormat( float&  /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
INLINE CUarray_format_enum getFormat( half4&  /*x*/ ) { return CU_AD_FORMAT_HALF; }
INLINE CUarray_format_enum getFormat( half2&  /*x*/ ) { return CU_AD_FORMAT_HALF; }
INLINE CUarray_format_enum getFormat( half&   /*x*/ ) { return CU_AD_FORMAT_HALF; }
INLINE CUarray_format_enum getFormat( uchar4& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
INLINE CUarray_format_enum getFormat( uchar2& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
INLINE CUarray_format_enum getFormat( uchar&  /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
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
