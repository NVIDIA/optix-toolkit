// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

using SelfIntersectionAvoidance::Matrix3x4;

struct RayGenData
{
};

struct MissData
{
};

struct InstanceData
{
};

#define MAX_GRAPH_DEPTH 16

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<InstanceData> InstanceSbtRecord;

struct Stats
{
    unsigned long long frontMissBackMiss;
    unsigned long long frontHitBackMiss;
    unsigned long long frontMissBackHit;
    unsigned long long frontHitBackHit;
    unsigned long long contextContextFreeMissmatch;
    unsigned long long optixCudaMissmatch;
};

struct SpawnPoint
{
#ifdef __CUDACC__
    __device__ unsigned int ulps( float a, float b )
    {
        return abs( ( int )( __float_as_uint( a ) - __float_as_uint( b ) ) );
    }

    __device__ unsigned int ulps( float3 a, float3 b )
    {
        return max( max( ulps( a.x, b.x ), ulps( a.y, b.y ) ), ulps( a.z, b.z ) );
    }

    __device__ bool operator!=( const SpawnPoint& s )
    {
        // miss-matching threshold in ulps for non-critical errors
        const unsigned int threshold = 16;

        if( !set && !s.set )
            return false;
        else if( set != s.set )
            return true;

        // small errors on the error estimate are ok
        if( ulps( objOffset, s.objOffset ) > threshold )
            return true;

        if( objPos != s.objPos )
            return true;

        // small errors on the normal are ok
        if( ulps( objNorm, s.objNorm ) > threshold )
            return true;

        // small errors on the error estimate are ok
        if( ulps( wldOffset, s.wldOffset ) > threshold )
            return true;

        if( wldPos != s.wldPos )
            return true;

        // small errors on the normal are ok
        if( ulps( wldNorm, s.wldNorm ) > threshold )
            return true;

        if( ulps( wldFront, s.wldFront ) > threshold )
            return true;

        if( ulps( wldBack, s.wldBack ) > threshold )
            return true;

        return false;
}
#endif

    bool   set;
    float  objOffset;
    float3 objPos;
    float3 objNorm;
    float  wldOffset;
    float3 wldPos;
    float3 wldNorm;
    float3 wldFront;
    float3 wldBack;
};

static __device__ __host__ double detAffine3x4( double *m )
{
    double d = m[0] * m[5] * m[10] -
        m[0] * m[9] * m[6] -
        m[4] * m[1] * m[10] +
        m[4] * m[9] * m[2] +
        m[8] * m[1] * m[6] -
        m[8] * m[5] * m[2];

    return d;
}

inline __device__ __host__ Matrix3x4 computeInverseAffine3x4( Matrix3x4 mtrx )
{
    double m[12], o[12];

    m[0] = mtrx.row0.x; m[1] = mtrx.row0.y; m[2] = mtrx.row0.z; m[3] = mtrx.row0.w;
    m[4] = mtrx.row1.x; m[5] = mtrx.row1.y; m[6] = mtrx.row1.z; m[7] = mtrx.row1.w;
    m[8] = mtrx.row2.x; m[9] = mtrx.row2.y; m[10] = mtrx.row2.z; m[11] = mtrx.row2.w;

    const double d = 1.0f / detAffine3x4( m );

    o[0] = d * ( m[5] * ( m[10] * 1.0 - 0.0 * m[11] ) + m[9] * ( 0.0 * m[7] - m[6] * 1.0 ) + 0.0 * ( m[6] * m[11] - m[10] * m[7] ) );
    o[4] = d * ( m[6] * ( m[8] * 1.0 - 0.0 * m[11] ) + m[10] * ( 0.0 * m[7] - m[4] * 1.0 ) + 0.0 * ( m[4] * m[11] - m[8] * m[7] ) );
    o[8] = d * ( m[7] * ( m[8] * 0.0 - 0.0 * m[9] ) + m[11] * ( 0.0 * m[5] - m[4] * 0.0 ) + 1.0 * ( m[4] * m[9] - m[8] * m[5] ) );
    o[1] = d * ( m[9] * ( m[2] * 1.0 - 0.0 * m[3] ) + 0.0 * ( m[10] * m[3] - m[2] * m[11] ) + m[1] * ( 0.0 * m[11] - m[10] * 1.0f ) );
    o[5] = d * ( m[10] * ( m[0] * 1.0 - 0.0 * m[3] ) + 0.0 * ( m[8] * m[3] - m[0] * m[11] ) + m[2] * ( 0.0 * m[11] - m[8] * 1.0f ) );
    o[9] = d * ( m[11] * ( m[0] * 0.0 - 0.0 * m[1] ) + 1.0 * ( m[8] * m[1] - m[0] * m[9] ) + m[3] * ( 0.0 * m[9] - m[8] * 0.0f ) );
    o[2] = d * ( 0.0f * ( m[2] * m[7] - m[6] * m[3] ) + m[1] * ( m[6] * 1.0 - 0.0f * m[7] ) + m[5] * ( 0.0 * m[3] - m[2] * 1.0f ) );
    o[6] = d * ( 0.0f * ( m[0] * m[7] - m[4] * m[3] ) + m[2] * ( m[4] * 1.0 - 0.0f * m[7] ) + m[6] * ( 0.0 * m[3] - m[0] * 1.0f ) );
    o[10] = d * ( 1.0f * ( m[0] * m[5] - m[4] * m[1] ) + m[3] * ( m[4] * 0.0 - 0.0f * m[5] ) + m[7] * ( 0.0 * m[1] - m[0] * 0.0f ) );
    o[3] = d * ( m[1] * ( m[10] * m[7] - m[6] * m[11] ) + m[5] * ( m[2] * m[11] - m[10] * m[3] ) + m[9] * ( m[6] * m[3] - m[2] * m[7] ) );
    o[7] = d * ( m[2] * ( m[8] * m[7] - m[4] * m[11] ) + m[6] * ( m[0] * m[11] - m[8] * m[3] ) + m[10] * ( m[4] * m[3] - m[0] * m[7] ) );
    o[11] = d * ( m[3] * ( m[8] * m[5] - m[4] * m[9] ) + m[7] * ( m[0] * m[9] - m[8] * m[1] ) + m[11] * ( m[4] * m[1] - m[0] * m[5] ) );

    mtrx.row0.x = static_cast<float>( o[0] );
    mtrx.row0.y = static_cast<float>( o[1] );
    mtrx.row0.z = static_cast<float>( o[2] );
    mtrx.row0.w = static_cast<float>( o[3] );
    
    mtrx.row1.x = static_cast<float>( o[4] );
    mtrx.row1.y = static_cast<float>( o[5] );
    mtrx.row1.z = static_cast<float>( o[6] );
    mtrx.row1.w = static_cast<float>( o[7] );
    
    mtrx.row2.x = static_cast<float>( o[8] );
    mtrx.row2.y = static_cast<float>( o[9] );
    mtrx.row2.z = static_cast<float>( o[10] );
    mtrx.row2.w = static_cast<float>( o[11] );

    return mtrx;
}

struct TransformPtr
{
    OptixTransformType type;
    union
    {
        OptixSRTMotionTransform*      srt;
        OptixStaticTransform*        mtrx;
        OptixMatrixMotionTransform* mmtrx;
        OptixInstance*               inst;
    };

#ifdef __CUDACC__
    __device__ OptixTransformType getTransformTypeFromHandle() const { return type; }

    __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const
    {
        return mmtrx;
    }

    __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const
    {
        return srt;
    }

    __device__ const OptixStaticTransform* getStaticTransformFromHandle() const
    {
        return mtrx;
    }

    __device__ Matrix3x4 getInstanceTransformFromHandle() const
    {
        return optix_impl::optixLoadReadOnlyAlign16( reinterpret_cast< const Matrix3x4* >( inst->transform ) );
    }

    __device__ Matrix3x4 getInstanceInverseTransformFromHandle() const
    {
        return computeInverseAffine3x4( getInstanceTransformFromHandle() );
    }
#endif
};


struct Params
{
    unsigned int           size;

    TransformPtr           transforms[MAX_GRAPH_DEPTH];
    OptixTraversableHandle handles[MAX_GRAPH_DEPTH];
    OptixTraversableHandle root;
    unsigned int           depth;
    float                  time;

    float2* barys;
    Stats*  stats;

    float3* vertices;
    SpawnPoint* spawn;
};
