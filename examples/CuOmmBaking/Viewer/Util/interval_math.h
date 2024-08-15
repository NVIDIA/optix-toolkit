// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <cfloat>

template <typename T>
__device__ __forceinline__ void swap( T& a, T& b )
{
    T tmp = a;
    a = b;
    b = tmp;
}


// interval algebra

struct interval_t
{
    float lo, hi;

    __device__ __forceinline__ interval_t( float f )
        : lo( f )
        , hi( f ) {};

    __device__ __forceinline__ interval_t( float _lo, float _hi )
        : lo( _lo )
        , hi( _hi ) {};

    __device__ __forceinline__ static interval_t make( float v0 ) { return { v0, v0 }; }

    __device__ __forceinline__ static interval_t make( float v0, float v1 ) { return { fminf( v0, v1 ), fmaxf( v0, v1 ) }; }

    __device__ __forceinline__ static interval_t make( float v0, float v1, float v2 ) { return { fminf( fminf( v0, v1 ), v2 ), fmaxf( fmaxf( v0, v1 ), v2 ) }; }

    __device__ __forceinline__ interval_t& operator*=( float s )
    {
        if( s < 0.f )
            swap( lo, hi );
        lo *= s;
        hi *= s;
        return *this;
    }

    __device__ __forceinline__ interval_t& operator+=( float s )
    {
        lo += s;
        hi += s;
        return *this;
    }

    __device__ __forceinline__ interval_t& operator-=( float s )
    {
        lo -= s;
        hi -= s;
        return *this;
    }

    __device__ __forceinline__ interval_t operator*( float s )
    {
        if( s < 0.f )
            return { hi * s, lo * s };
        return { lo * s, hi * s };
    }

    __device__ __forceinline__ interval_t operator+( float s ) const { return { lo + s, hi + s }; }

    __device__ __forceinline__ interval_t operator-( float s ) const { return { lo - s, hi - s }; }

    __device__ __forceinline__ interval_t operator-() const { return { -hi, -lo }; }

    __device__ __forceinline__ interval_t operator+( interval_t v ) const { return { lo + v.lo, hi + v.hi }; }

    __device__ __forceinline__ interval_t operator-( interval_t v ) const { return { lo - v.hi, hi - v.lo }; }

    __device__ __forceinline__ interval_t operator*( interval_t v ) const
    {
        interval_t a = v * lo;
        interval_t b = v * hi;

        return { fminf( a.lo, b.lo ), fmaxf( a.hi, b.hi ) };
    }
};

__device__ __forceinline__ interval_t operator+( float s, interval_t v )
{
    return v + s;
}

__device__ __forceinline__ interval_t operator-( float s, interval_t v )
{
    return { s - v.hi, s - v.lo };
}

__device__ __forceinline__ interval_t operator*( float s, interval_t v )
{
    return v * s;
}

__device__ __forceinline__ interval_t operator/( float s, interval_t v )
{
    if( s == 0 )
        return { 0,0 };

    interval_t r( 1.f / v.hi, 1.f / v.lo );

    if( v.lo < 0 && 0 < v.hi )
    {
        r.lo = -FLT_MAX;
        r.hi = FLT_MAX;
    }

    return r * s;
}

__device__ __forceinline__ interval_t sqrtf( interval_t t )
{
    return interval_t( sqrtf( fmaxf( t.lo, 0.f ) ), sqrtf( fmaxf( t.hi, 0.f ) ) );
}

template <typename T>
struct vec2
{
    T x, y;

    __device__ __forceinline__ vec2( T _x, T _y )
        : x( _x )
        , y( _y ) {};

    template <typename U>
    __device__ __forceinline__ vec2<T>& operator=( U s )
    {
        x = s.x;
        y = s.y;
        return *this;
    }

    template <typename U>
    __device__ __forceinline__ vec2<T>& operator*=( U s )
    {
        x *= s;
        y *= s;
        return *this;
    }

    template <typename U>
    __device__ __forceinline__ vec2<T>& operator+=( U s )
    {
        x += s.x;
        y += s.y;
        return *this;
    }

    template <typename U>
    __device__ __forceinline__ vec2<T>& operator-=( U s )
    {
        x -= s.x;
        y -= s.y;
        return *this;
    }

    template <typename U>
    __device__ __forceinline__ vec2<T> operator+( U s )
    {
        return vec2<T>( x + s.x, y + s.y );
    }

    template <typename U>
    __device__ __forceinline__ vec2<T> operator-( U s )
    {
        return vec2<T>( x - s.x, y - s.y );
    }

    template <typename U>
    __device__ __forceinline__ vec2<T> operator*( U s )
    {
        return vec2<T>( x * s.x, y * s.y );
    }
};

template <typename T>
__device__ __forceinline__ vec2<T> operator*( T a, float2 b )
{
    return vec2<T>( a * b.x, a * b.y );
}

template <typename T, typename U>
__device__ __forceinline__ vec2<T> operator*( U a, vec2<T> b )
{
    return vec2<T>( a * b.x, a * b.y );
}

template <typename T>
__device__ __forceinline__ vec2<T> operator+( float2 s, vec2<T> v )
{
    return vec2<T>( v.x + s.x, v.y + s.y );
}

template <typename T>
__device__ __forceinline__ static T dot( vec2<T> a, vec2<T> b )
{
    return a.x * b.x + a.y * b.y;
}

template <typename T>
__device__ __forceinline__ static T length( vec2<T> a )
{
    return sqrtf( dot( a, a ) );
}

__device__ __forceinline__ static vec2<interval_t> make_vec2( float2 a )
{
    return vec2<interval_t>( interval_t::make( a.x ), interval_t::make( a.y ) );
}

__device__ __forceinline__ static vec2<interval_t> make_vec2( float2 a, float2 b )
{
    return vec2<interval_t>( interval_t::make( a.x, b.x ), interval_t::make( a.y, b.y ) );
}

__device__ __forceinline__ static vec2<interval_t> make_vec2( float2 a, float2 b, float2 c )
{
    return vec2<interval_t>( interval_t::make( a.x, b.x, c.x ), interval_t::make( a.y, b.y, c.y ) );
}

template <typename T>
struct vec3
{
    T x, y, z;

    __device__ __forceinline__ vec3( T _x, T _y, T _z )
        : x( _x )
        , y( _y )
        , z( _z ) {};

    __device__ __forceinline__ vec3<T>& operator*=( float s )
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    template <typename U>
    __device__ __forceinline__ vec3<T>& operator+=( U s )
    {
        x += s.x;
        y += s.y;
        z += s.z;
        return *this;
    }

    template <typename U>
    __device__ __forceinline__ vec3<T>& operator-=( U s )
    {
        x -= s.x;
        y -= s.y;
        z -= s.z;
        return *this;
    }

    template <typename U>
    __device__ __forceinline__ vec3<T> operator+( U s )
    {
        return vec3<T>( x + s.x, y + s.y, z + s.z );
    }

    template <typename U>
    __device__ __forceinline__ vec3<T> operator-( U s )
    {
        return vec3<T>( x - s.x, y - s.y, z - s.z );
    }

    __device__ __forceinline__ vec3<T> operator*( vec3<T> s ) { return vec3<T>( x * s.x, y * s.y, z * s.z ); }

    __device__ __forceinline__ vec3<T> operator*( T s ) { return vec3<T>( x * s, y * s, z * s ); }
};

template <typename T>
__device__ __forceinline__ vec3<T> operator*( T d, float3 v )
{
    return vec3<T>( d * v.x, d * v.y, d * v.z );
}

template <typename T>
__device__ __forceinline__ vec3<T> operator*( T d, vec3<T> v )
{
    return v * d;
}

template <typename T>
__device__ __forceinline__ vec3<T> operator*( vec3<T> a, float3 b )
{
    return vec3<T>( a * b.x, a * b.y, a * b.z );
}

template <typename T>
__device__ __forceinline__ vec3<T> operator+( float3 s, vec3<T> v )
{
    return vec3<T>( v.x + s.x, v.y + s.y, v.z + s.z );
}

__device__ __forceinline__ static interval_t dot( vec3<interval_t> a, vec3<interval_t> b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ static vec3<interval_t> make_vec3( float3 a )
{
    return vec3<interval_t>( interval_t::make( a.x ), interval_t::make( a.y ), interval_t::make( a.z ) );
}

__device__ __forceinline__ static vec3<interval_t> make_vec3( float3 a, float3 b )
{
    return vec3<interval_t>( interval_t::make( a.x, b.x ), interval_t::make( a.y, b.y ), interval_t::make( a.z, b.z ) );
}

__device__ __forceinline__ static vec3<interval_t> make_vec3( float3 a, float3 b, float3 c )
{
    return vec3<interval_t>( interval_t::make( a.x, b.x, c.x ), interval_t::make( a.y, b.y, c.y ), interval_t::make( a.z, b.z, c.z ) );
}

__device__ __forceinline__ interval_t fminf( interval_t a, interval_t b )
{
    return { fminf( a.lo, b.lo ), fminf( a.hi, b.hi ) };
}

__device__ __forceinline__ interval_t fmaxf( interval_t a, interval_t b )
{
    return { fmaxf( a.lo, b.lo ), fmaxf( a.hi, b.hi ) };
}

__device__ __forceinline__ interval_t clamp( interval_t a, float lo, float hi )
{
    return { fminf( fmaxf( a.lo, lo ), hi ), fminf( fmaxf( a.hi, lo ), hi ) };
}

__device__ __forceinline__ interval_t fabsf( interval_t t )
{
    interval_t r( fminf( fabsf( t.lo ), fabsf( t.hi ) ), fmaxf( fabsf( t.lo ), fabsf( t.hi ) ) );

    if( t.lo < 0 && t.hi > 0 )
        r.lo = 0;

    return r;
}

__device__ __forceinline__ interval_t cosf( interval_t t )
{
    if( t.hi - t.lo > ( 2.f * M_PIf ) )
        return interval_t( -1.f, 1.f );

    const float clo = cosf( t.lo );
    const float chi = cosf( t.hi );

    float hi = fmaxf( clo, chi );
    float lo = fminf( clo, chi );

    if( ceilf( t.lo * ( 1.f / ( 2.f * M_PIf ) ) ) * ( 2.f * M_PIf ) <= t.hi )
        hi = 1.f;

    if( ceilf( ( t.lo - M_PIf ) * ( 1.f / ( 2.f * M_PIf ) ) ) * ( 2.f * M_PIf ) + M_PIf <= t.hi )
        lo = -1.f;

    return interval_t( lo, hi );
}

__device__ __forceinline__ interval_t sinf( interval_t t )
{
    if( t.hi - t.lo > ( 2.f * M_PIf ) )
        return interval_t( -1.f, 1.f );

    const float slo = sinf( t.lo );
    const float shi = sinf( t.hi );

    float hi = fmaxf( slo, shi );
    float lo = fminf( slo, shi );

    if( ceilf( ( t.lo - 0.5f * M_PIf ) * ( 1.f / ( 2.f * M_PIf ) ) ) * ( 2.f * M_PIf ) + 0.5f * M_PIf <= t.hi )
        hi = 1.f;

    if( ceilf( ( t.lo - 1.5f * M_PIf ) * ( 1.f / ( 2.f * M_PIf ) ) ) * ( 2.f * M_PIf ) + 1.5f * M_PIf <= t.hi )
        lo = -1.f;

    return interval_t( lo, hi );
}
