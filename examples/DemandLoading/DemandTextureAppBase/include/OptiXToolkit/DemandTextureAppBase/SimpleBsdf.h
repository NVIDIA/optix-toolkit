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

#include <OptiXToolkit/DemandTextureAppBase/LaunchParams.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureAppDeviceUtil.h>

using namespace demandLoading;
using namespace demandTextureApp;

OTK_DEVICE const float MIN_ROUGHNESS = 0.0000001f;

static __forceinline__ __device__
float3 evalPhongBsdf( float3 Kd, float3 Ks, float roughness, SurfaceGeometry g, float3 D, float3 R, 
                      float& prob, float curvature )
{
    using namespace otk;
    float kd = ( Kd.x + Kd.y + Kd.z ) / 3.0f;
    float ks = ( Ks.x + Ks.y + Ks.z ) / 3.0f;
    float ksum = kd + ks;

    // Diffuse term
    float3 Fd = Kd / M_PIf;
    float pd = dot( R, g.N ) / M_PIf;

    // Specular term
    float3 refl = reflect( D, g.N );
    float cosTheta = maxf( 0.0f, dot( refl, R ) );
    float s = 1.0f / roughness; 
    float ps = ( s + 2.0f ) * powf( cosTheta, s ) / ( 2.0f * M_PIf );
    float correction = 0.5f * ( fabsf( dot( R, g.N ) ) + fabsf( dot( D, g.N ) ) );
    float3 Fs = Ks * ps / correction;

    // Calculate the combined probability of sampling this ray when sampling the bsdf
    prob = ( pd * kd / ksum ) + ( ps * ks / ksum );
    return Fd + Fs;
}

static __forceinline__ __device__
float3 sampleDiffuseLobe( float2 xi, float3 N, float3 S, float3 T )
{
    using namespace otk;
    const float2 st = concentricMapping( xi );
    return ( st.x * S ) + ( st.y * T ) + ( sqrtf( 1.0f - dot( st, st ) ) * N );
}

static __forceinline__ __device__ 
float3 samplePhongLobe( float2 xi, float s, float3 R, float3 U, float3 V )
{
    using namespace otk;
    float cosTheta = pow( 1.0f-xi.x, 1.0f/(2.0f+s) );
    float sinTheta = sqrtf( 1.0f - cosTheta*cosTheta );
    float phi = 2.0f * M_PIf * xi.y;
    return ( cosTheta * R ) + ( cosf( phi ) * sinTheta * U ) + ( sin( phi ) * sinTheta * V );
}

static __forceinline__ __device__
bool samplePhongBsdf( float2 xi, float3 Kd, float3 Ks, float roughness, SurfaceGeometry g, float3 D, float3& R )
{
    using namespace otk;
    float kd = ( Kd.x + Kd.y + Kd.z ) / 3.0f;
    float ks = ( Ks.x + Ks.y + Ks.z ) / 3.0f;
    float ksum = kd + ks;

    float x = xi.x * ksum;
    if ( x < kd ) // diffuse term
    {
        xi.x = ( x / kd );
        R = sampleDiffuseLobe( xi, g.N, g.S, g.T );
    }
    else // specular term
    {
        xi.x = ( x - kd ) / ks;  
        R = reflect( D, g.N );
        if( roughness > MIN_ROUGHNESS )
        {
            float3 U, V;
            makeOrthoBasis( R, U, V );
            R = samplePhongLobe( xi, 1.0f / roughness, R, U, V );
        }
    }

    // Failed to sample above surface, return false
    if( dot( R, g.N ) < 0.0f || dot( R, g.Ng ) < 0.0f )
        return false;

    return true;
}

static __forceinline__ __device__ float schlick( float ior, float cosTheta )
{
    float r0 = ( ior - 1.0f ) / ( ior + 1.0f );
    r0 *= r0;
    return r0 + ( 1.0f - r0 ) * powf( 1.0f - cosTheta, 5.0f );
}

static __forceinline__ __device__ bool refract( float3 D, float3 N, float ior, float3& T )
{
    using namespace otk;
    float dt = dot( D, N );
    float disc = 1.0f - ior * ior * ( 1.0f - dt * dt );
    T = ( disc >= 0.0f ) ? ior * (D - dt*N) - sqrt( disc ) * N : float3{0.0f};
    return ( disc < 0.0f );
}

static __forceinline__ __device__ 
float3 evalGlassBsdf( float3 Ks, float3 Kt, float ior, float roughness, SurfaceGeometry g, float3 D, float3 R, 
                      float& prob, float curvature )
{
    using namespace otk;
    float fr = schlick( ior, fabsf( dot( g.N, D ) ) );
    float ks = fr * (Ks.x + Ks.y + Ks.z) / 3.0f;
    float kt = ( 1.0f - fr ) * ( Kt.x + Kt.y + Kt.z ) / 3.0f;
    float ksum = ks + kt;

    float s = 1.0f / roughness; // phong exponent
    if( dot( g.N, R ) > 0.0f )  // reflect
    {
        float3 refl = reflect( D, g.N );
        float cosTheta = fabs( dot( refl, R ) );
        float ps = ( s + 2.0f ) * powf( cosTheta, s ) / ( 2.0f * M_PIf );
        prob = ps * ks / ksum; 
        return ps * fr * Ks;
    }
    else  // transmit
    {
        float3 refr;
        bool tir = refract( D, g.N, 1.0f/ior, refr );
        if( tir ) 
        {
            prob = 0.0f;
            return float3{0.0f};
        }
        float cosTheta = fabs( dot( refr, R ) );
        float pt = ( s + 2.0f ) * powf( cosTheta, s ) / (2.0f * M_PIf);
        prob = pt * kt / ksum;
        return pt * ( 1.0f - fr ) * Kt;
    }
}

static __forceinline__ __device__
bool sampleGlassBsdf( float2 xi, float3 Ks, float3 Kt, float ior, float roughness, SurfaceGeometry g, float3 D, float3& R )
{
    using namespace otk;
    float fr = schlick( ior, fabsf( dot( g.N, D ) ) );
    float ks = fr * ( Ks.x + Ks.y + Ks.z ) / 3.0f;
    float kt = ( 1.0f - fr ) * ( Kt.x + Kt.y + Kt.z ) / 3.0f;
    float ksum = ks + kt;

    float x = xi.x * ksum;
    if( x < ks ) // reflection
    {
        xi.x = x / ks;
        R = reflect( D, g.N );
        if( roughness > MIN_ROUGHNESS ) // treat min roughness as perfect reflection
        {
            float3 U, V;
            makeOrthoBasis( R, U, V );
            R = samplePhongLobe( xi, 1.0f / roughness, R, U, V );
        }
        if( dot( R, g.N ) < 0.0f || dot( R, g.Ng ) < 0.0f )
            return false;
    }
    else // transmission
    {
        xi.x = ( x - ks ) / kt;
        bool tir = refract( D, g.N, 1.0f/ior, R );
        if( tir )
        {
            R = float3{0.0f};
            return false;
        }
        if( roughness > MIN_ROUGHNESS ) // treat min roughness as perfect reflection
        {
            float3 U, V;
            makeOrthoBasis( R, U, V );
            R = samplePhongLobe( xi, 1.0f / roughness, R, U, V );
        }
        if( dot( R, g.N ) > 0.0f || dot( R, g.Ng ) > 0.0f )
            return false;
    }
    return true;
}

static __forceinline__ __device__ 
float3 evalBsdf( float3 Kd, float3 Ks, float3 Kt, float ior, float roughness, SurfaceGeometry geom, float3 D, float3 R, 
                 float& prob, float curvature )
{
    const bool isPhongMaterial = ( ior == 0.0f );
    if( isPhongMaterial )
        return evalPhongBsdf( Kd, Ks, roughness, geom, D, R, prob, curvature );
    else // glass
        return evalGlassBsdf( Ks, Kt, ior, roughness, geom, D, R, prob, curvature );
}

static __forceinline__ __device__
bool sampleBsdf( float2 xi, float3 Kd, float3 Ks, float3 Kt, float ior, float roughness, SurfaceGeometry geom, float3 D, float3& R )
{
    const bool isPhongMaterial = ( ior == 0.0f );
    if( isPhongMaterial )
        return samplePhongBsdf( xi, Kd, Ks, roughness, geom, D, R );
    else // glass
        return sampleGlassBsdf( xi, Ks, Kt, ior, roughness, geom, D, R );
}
