// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file CudaSelfIntersectionAvoidanceImpl.h
/// Cuda implementation of Self Intersection Avoidance library.
///

#include "SelfIntersectionAvoidanceImpl.h"

namespace SelfIntersectionAvoidance {

// Encapsulate the custom transform. This is needed so a custom transform in gmem is not forced into lmem before access. The offsetting code assumes the transform data lives in gmem.
template <typename T>
class Transform
{
  public:
    OTK_INLINE __device__ Transform( const T* __restrict transform )
        : m_transform( transform ){};

    OTK_INLINE __device__ OptixTransformType getTransformTypeFromHandle() const
    {
        return m_transform->getTransformTypeFromHandle();
    };
    OTK_INLINE __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const
    {
        return m_transform->getMatrixMotionTransformFromHandle();
    };
    OTK_INLINE __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const
    {
        return m_transform->getSRTMotionTransformFromHandle();
    };
    OTK_INLINE __device__ const OptixStaticTransform* getStaticTransformFromHandle() const
    {
        return m_transform->getStaticTransformFromHandle();
    };
    OTK_INLINE __device__ Matrix3x4 getInstanceTransformFromHandle() const
    {
        return m_transform->getInstanceTransformFromHandle();
    };
    OTK_INLINE __device__ Matrix3x4 getInstanceInverseTransformFromHandle() const
    {
        return m_transform->getInstanceInverseTransformFromHandle();
    };

  private:
    const T* __restrict m_transform;
};

// List of generic transforms
template<typename T>
class TransformList
{
  public:
    typedef Transform<T> value_t;

    OTK_INLINE __device__ TransformList( int size, const T* transforms )
        : m_size( size )
        , m_transforms( transforms ){};

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return m_size; }

    OTK_INLINE __device__ value_t getTransform( unsigned int index ) const { return value_t( m_transforms + index ); }

  private:
    unsigned int m_size;
    const T* __restrict m_transforms;
};

// Specialized instance transform
class InstanceTransform
{
  public:
    OTK_INLINE __device__ InstanceTransform( const Matrix3x4* o2w, const Matrix3x4* w2o )
        : m_o2w( o2w )
        , m_w2o( w2o ){};

    OTK_INLINE __device__ OptixTransformType getTransformTypeFromHandle() const
    {
        return OPTIX_TRANSFORM_TYPE_INSTANCE;
    }

#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES
    OTK_INLINE __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const { return 0; }

    OTK_INLINE __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const { return 0; }

    OTK_INLINE __device__ const OptixStaticTransform* getStaticTransformFromHandle() const { return 0; }
#endif

    OTK_INLINE __device__ Matrix3x4 getInstanceTransformFromHandle() const { return optix_impl::optixLoadReadOnlyAlign16( m_o2w ); }

    OTK_INLINE __device__ Matrix3x4 getInstanceInverseTransformFromHandle() const { return optix_impl::optixLoadReadOnlyAlign16( m_w2o ); }

  private:
    const Matrix3x4* __restrict m_o2w;
    const Matrix3x4* __restrict m_w2o;
};

// Specialized instance transform list
class InstanceTransformSingleton
{
  public:
    typedef InstanceTransform value_t;

    OTK_INLINE __device__ InstanceTransformSingleton( const Matrix3x4* o2w, const Matrix3x4* w2o )
        : m_o2w( o2w )
        , m_w2o( w2o ){};

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return 1; }

    OTK_INLINE __device__ InstanceTransform getTransform( unsigned int index ) const
    {
        return InstanceTransform( m_o2w , m_w2o );
    }

  private:
    unsigned int m_size;
    const Matrix3x4* __restrict m_o2w;
    const Matrix3x4* __restrict m_w2o;
};

OTK_INLINE __device__ void transformSafeSpawnOffset( float3&            outPosition,
                                                     float3&            outNormal,
                                                     float&             outOffset,
                                                     const float3&      obj_p,
                                                     const float3&      obj_n,
                                                     const float        obj_offset,
                                                     const Matrix3x4* const __restrict o2w,
                                                     const Matrix3x4* const __restrict w2o )
{
    safeInstancedSpawnOffsetImpl<InstanceTransformSingleton>( outPosition, outNormal, outOffset, obj_p, obj_n, obj_offset,
                                                              0.f, InstanceTransformSingleton( o2w, w2o ) );
}

template <typename T>
OTK_INLINE __device__ void transformSafeSpawnOffset( float3& outPosition,
    float3&       outNormal,
    float&        outOffset,
    const float3& obj_p,
    const float3& obj_n,
    const float   obj_offset,
    const float   time,
    const unsigned int numTransforms,
    const T* const __restrict transforms )
{
    safeInstancedSpawnOffsetImpl<TransformList<T>>( outPosition, outNormal, outOffset, obj_p, obj_n, obj_offset, time, TransformList<T>( numTransforms, transforms ) );
}

}  // namespace SelfIntersectionAvoidance
