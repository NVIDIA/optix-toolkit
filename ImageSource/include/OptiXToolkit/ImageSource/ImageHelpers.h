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
inline unsigned int getNumChannels( float4& /*x*/ ) { return 4; }
inline unsigned int getNumChannels( float2& /*x*/ ) { return 2; }
inline unsigned int getNumChannels( float&  /*x*/ ) { return 1; }
inline unsigned int getNumChannels( ubyte4& /*x*/ ) { return 4; }
inline unsigned int getNumChannels( ubyte2& /*x*/ ) { return 2; }
inline unsigned int getNumChannels( ubyte&  /*x*/ ) { return 1; }
inline unsigned int getNumChannels( unsigned int& /*x*/ ) { return 1; }

inline CUarray_format_enum getFormat( float4& /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
inline CUarray_format_enum getFormat( float2& /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
inline CUarray_format_enum getFormat( float&  /*x*/ ) { return CU_AD_FORMAT_FLOAT; }
inline CUarray_format_enum getFormat( ubyte4& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
inline CUarray_format_enum getFormat( ubyte2& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
inline CUarray_format_enum getFormat( ubyte&  /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT8; }
inline CUarray_format_enum getFormat( unsigned int& /*x*/ ) { return CU_AD_FORMAT_UNSIGNED_INT32; }

inline void convertType( float4 a, float4& b ) { b = a; }
inline void convertType( float4 a, float2& b ) { b = {a.x, (a.y+a.z)}; }
inline void convertType( float4 a, float& b  ) { b = (a.x + a.y + a.z) / 3.0f; }
inline void convertType( float4 a, ubyte4& b ) { b = {ubyte(a.x*255.0f), ubyte(a.y*255.0f), ubyte(a.z*255.0f), ubyte(a.w*255.0f)}; }
inline void convertType( float4 a, ubyte2& b ) { b = {ubyte(a.x*255.0f), ubyte(a.y*255.0f)}; }
inline void convertType( float4 a, ubyte& b ) { b = ubyte(255.0f * (a.x + a.y + a.z) / 3.0f); }
inline void convertType( float4 a, unsigned int& b ) { b = (a.x+a.y+a.z > 0.1f) ? (1<<30) : 0; }

inline unsigned int getNumChannels( half4& /*x*/ ) { return 4; }
inline unsigned int getNumChannels( half2& /*x*/ ) { return 2; }
inline unsigned int getNumChannels( half&  /*x*/ ) { return 1; }
inline CUarray_format_enum getFormat( half4& /*x*/ ) { return CU_AD_FORMAT_HALF; }
inline CUarray_format_enum getFormat( half2& /*x*/ ) { return CU_AD_FORMAT_HALF; }
inline CUarray_format_enum getFormat( half&  /*x*/ ) { return CU_AD_FORMAT_HALF; }
inline void convertType( float4 a, half4& b ) { b = half4{half(a.x), half(a.y), half(a.z), half(a.w)}; }
inline void convertType( float4 a, half2& b ) { b = half2{half(a.x), half(a.y)}; }
inline void convertType( float4 a, half& b ) { b = half((a.x + a.y + a.z) / 3.0f); }
// clang-format on


}  // namespace imageSource
