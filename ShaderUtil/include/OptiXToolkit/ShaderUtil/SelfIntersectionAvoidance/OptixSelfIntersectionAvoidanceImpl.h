// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file OptixSelfIntersectionAvoidanceImpl.h
/// Optix implementation of Self Intersection Avoidance library.

#include "SelfIntersectionAvoidanceImpl.h"
#include <assert.h>

namespace SelfIntersectionAvoidance {

class OptixTransform
{
  public:
    OTK_INLINE __device__ OptixTransform( OptixTraversableHandle handle )
        : m_handle( handle ){};

    OTK_INLINE __device__ OptixTransformType getTransformTypeFromHandle() const
    {
        return optixGetTransformTypeFromHandle( m_handle );
    }

    OTK_INLINE __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const
    {
        return optixGetSRTMotionTransformFromHandle( m_handle );
    }

    OTK_INLINE __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const
    {
        return optixGetMatrixMotionTransformFromHandle( m_handle );
    }

    OTK_INLINE __device__ const OptixStaticTransform* getStaticTransformFromHandle() const
    {
        return optixGetStaticTransformFromHandle( m_handle );
    }

    OTK_INLINE __device__ const Matrix3x4 getInstanceTransformFromHandle() const
    {
        return optix_impl::optixLoadReadOnlyAlign16( reinterpret_cast< const Matrix3x4* >( optixGetInstanceTransformFromHandle( m_handle ) ) );
    }

    OTK_INLINE __device__ const Matrix3x4 getInstanceInverseTransformFromHandle() const
    {
        return optix_impl::optixLoadReadOnlyAlign16( reinterpret_cast< const Matrix3x4* >( optixGetInstanceInverseTransformFromHandle( m_handle ) ) );
    }

  private:
    OptixTraversableHandle m_handle;
};

class OptixLocalTransformList
{
  public:
    typedef OptixTransform value_t;

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return optixGetTransformListSize(); }

    OTK_INLINE __device__ OptixTransform getTransform( unsigned int index ) const
    {
        return OptixTransform( optixGetTransformListHandle( index ) );
    }
};

class OptixTransformList
{
  public:
    typedef OptixTransform value_t;

    OTK_INLINE __device__ OptixTransformList( int size, const OptixTraversableHandle* handles )
        : m_size( size )
        , m_handles( handles ){};

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return m_size; }

    OTK_INLINE __device__ OptixTransform getTransform( unsigned int index ) const
    {
        return OptixTransform( m_handles[index] );
    }

  private:
    unsigned int m_size;
    const OptixTraversableHandle* __restrict m_handles;
};

OTK_INLINE __device__ void getSafeTriangleSpawnOffset( float3& outPosition, float3& outNormal, float& outOffset )
{
    assert( optixIsTriangleHit() );

    float3 data[3];
    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
                                optixGetRayTime(), data );

    getSafeTriangleSpawnOffset( outPosition, outNormal, outOffset, data[0], data[1], data[2], optixGetTriangleBarycentrics() );
}

OTK_INLINE __device__ void transformSafeSpawnOffset( float3&            outPosition,
                                                     float3&            outNormal,
                                                     float&             outOffset,
                                                     const float3&      inPosition,
                                                     const float3&      inNormal,
                                                     const float        inOffset,
                                                     const float        time,
                                                     const unsigned int numTransforms,
                                                     const OptixTraversableHandle* const __restrict transformHandles )
{
    safeInstancedSpawnOffsetImpl<OptixTransformList>( outPosition, outNormal, outOffset, inPosition, inNormal, inOffset,
                                                      time, OptixTransformList( numTransforms, transformHandles ) );
}

OTK_INLINE __device__ void transformSafeSpawnOffset( float3&       outPosition,
                                                     float3&       outNormal,
                                                     float&        outOffset,
                                                     const float3& inPosition,
                                                     const float3& inNormal,
                                                     const float   inOffset )
{
    safeInstancedSpawnOffsetImpl<OptixLocalTransformList>( outPosition, outNormal, outOffset, inPosition, inNormal,
                                                           inOffset, optixGetRayTime(), OptixLocalTransformList() );
}

}  // namespace SelfIntersectionAvoidance
