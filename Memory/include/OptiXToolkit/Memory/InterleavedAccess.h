// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef __CUDACC__

namespace otk {

#define WS 32 // Warp size
#define INLINE __forceinline__ __device__
#define FPTR float* f = ( float* )offsetPointer( ptr, base )
#define IPTR int* i = ( int* )offsetPointer( ptr, base )
#define UPTR unsigned int* u = ( unsigned int* )offsetPointer( ptr, base )

// Get interleaved pointer for an offset from a base pointer
INLINE char* offsetPointer( void* ptr, void* base ) { return ( char* )base + ( WS * ( ( char* )ptr - ( char* )base ) ); }

class InterleavedAccess
{
  public:
    void* base;

    INLINE InterleavedAccess( void* base_ ) : base( base_ ) {}
    INLINE operator void*() { return base; }

    // float access
    INLINE float getFloat( void* ptr ) { FPTR; return f[0]; }
    INLINE float2 getFloat2( void* ptr ) { FPTR; return float2{f[0], f[WS]}; }
    INLINE float3 getFloat3( void* ptr ) { FPTR; return float3{f[0], f[WS], f[2 * WS]}; }
    INLINE float4 getFloat4( void* ptr ) { FPTR; return float4{f[0], f[WS], f[2 * WS], f[3 * WS]}; }

    INLINE void setFloat( void* ptr, float v ) { FPTR; f[0] = v; }
    INLINE void setFloat2( void* ptr, float2 v ) { FPTR; f[0] = v.x; f[WS] = v.y; }
    INLINE void setFloat3( void* ptr, float3 v ) { FPTR; f[0] = v.x; f[WS] = v.y; f[2 * WS] = v.z; }
    INLINE void setFloat4( void* ptr, float4 v ) { FPTR; f[0] = v.x; f[WS] = v.y; f[2 * WS] = v.z; f[3 * WS] = v.w; }

    // int access
    INLINE int getInt( void* ptr ) { IPTR; return i[0]; }
    INLINE int2 getInt2( void* ptr ) { IPTR; return int2{i[0], i[WS]}; }
    INLINE int3 getInt3( void* ptr ) { IPTR; return int3{i[0], i[WS], i[2 * WS]}; }
    INLINE int4 getInt4( void* ptr ) { IPTR; return int4{i[0], i[WS], i[2 * WS], i[3 * WS]}; }

    INLINE void setInt( void* ptr, int v ) { IPTR; i[0] = v; }
    INLINE void setInt2( void* ptr, int2 v ) { IPTR; i[0] = v.x; i[WS] = v.y; }
    INLINE void setInt3( void* ptr, int3 v ) { IPTR; i[0] = v.x; i[WS] = v.y; i[2 * WS] = v.z; }
    INLINE void setInt4( void* ptr, int4 v ) { IPTR; i[0] = v.x; i[WS] = v.y; i[2 * WS] = v.z; i[3 * WS] = v.w; }

    // unsigned int access
    INLINE unsigned int getUint( void* ptr ) { UPTR; return u[0]; }
    INLINE uint2 getUint2( void* ptr ) { UPTR; return uint2{u[0], u[WS]}; }
    INLINE uint3 getUint3( void* ptr ) { UPTR; return uint3{u[0], u[WS], u[2 * WS]}; }
    INLINE uint4 getUint4( void* ptr ) { UPTR; return uint4{u[0], u[WS], u[2 * WS], u[3 * WS]}; }

    INLINE void setUint( void* ptr, unsigned int v ) { UPTR; u[0] = v; }
    INLINE void setUint2( void* ptr, uint2 v ) { UPTR; u[0] = v.x; u[WS] = v.y; }
    INLINE void setUint3( void* ptr, uint3 v ) { UPTR; u[0] = v.x; u[WS] = v.y; u[2 * WS] = v.z; }
    INLINE void setUint4( void* ptr, uint4 v ) { UPTR; u[0] = v.x; u[WS] = v.y; u[2 * WS] = v.z; u[3 * WS] = v.w; }
};

}  // namespace otk

#endif
