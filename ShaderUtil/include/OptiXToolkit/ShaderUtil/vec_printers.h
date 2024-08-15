// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
