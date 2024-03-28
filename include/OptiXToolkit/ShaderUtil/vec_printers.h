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

#include <vector_types.h>

#include <iostream>

namespace otk {

template <typename T>
std::ostream& printer2( std::ostream& str, const T& value )
{
    return str << '(' << value.x << ", " << value.y << ")";
}

template <typename T>
std::ostream& printer3( std::ostream& str, const T& value )
{
    return str << '(' << value.x << ", " << value.y << ", " << value.z << ")";
}

template <typename T>
std::ostream& printer4( std::ostream& str, const T& value )
{
    return str << '(' << value.x << ", " << value.y << ", " << value.z << ", " << value.w << ")";
}

}  // namespace otk

// clang-format off
inline std::ostream& operator<<( std::ostream& str, const short2& value )       { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ushort2& value )      { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const int2& value )         { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const uint2& value )        { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const long2& value )        { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ulong2& value )       { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const longlong2& value )    { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ulonglong2& value )   { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const float2& value )       { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const double2& value )      { return otk::printer2( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const short3& value )       { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ushort3& value )      { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const int3& value )         { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const uint3& value )        { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const long3& value )        { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ulong3& value )       { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const longlong3& value )    { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ulonglong3& value )   { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const float3& value )       { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const double3& value )      { return otk::printer3( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const short4& value )       { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ushort4& value )      { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const int4& value )         { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const uint4& value )        { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const long4& value )        { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ulong4& value )       { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const longlong4& value )    { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const ulonglong4& value )   { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const float4& value )       { return otk::printer4( str, value ); }
inline std::ostream& operator<<( std::ostream& str, const double4& value )      { return otk::printer4( str, value ); }
// clang-format on
