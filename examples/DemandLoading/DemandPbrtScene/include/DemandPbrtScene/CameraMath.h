// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ShaderUtil/Preprocessor.h>
#include <OptiXToolkit/ShaderUtil/Transform4.h>

#include <vector_types.h>

namespace demandPbrtScene {

struct PerspectiveCameraRay
{
    float3 origin;
    float3 direction;
    float3 cameraPoint;
};

OTK_INLINE OTK_HOSTDEVICE float4 getPbrtScreenWindow( float         imageAspectRatio,
                                                      bool          frameAspectRatioSpecified,
                                                      float         frameAspectRatio,
                                                      bool          screenWindowSpecified,
                                                      const float4& screenWindow )
{
    if( screenWindowSpecified )
    {
        return screenWindow;
    }

    const float aspectRatio{ frameAspectRatioSpecified ? frameAspectRatio : imageAspectRatio };
    if( aspectRatio > 1.0f )
    {
        return make_float4( -aspectRatio, aspectRatio, -1.0f, 1.0f );
    }
    return make_float4( -1.0f, 1.0f, -1.0f / aspectRatio, 1.0f / aspectRatio );
}

OTK_INLINE OTK_HOSTDEVICE otk::Transform4 makePbrtScreenToRaster( const float4& screenWindow, const float2& imageSize )
{
    return otk::scale( imageSize.x, imageSize.y, 1.0f )
           * otk::scale( 1.0f / ( screenWindow.y - screenWindow.x ), 1.0f / ( screenWindow.z - screenWindow.w ), 1.0f )
           * otk::translate( -screenWindow.x, -screenWindow.w, 0.0f );
}

OTK_INLINE OTK_HOSTDEVICE otk::Transform4 makePbrtRasterToCamera( const otk::Transform4& cameraToScreen,
                                                                  const float4&          screenWindow,
                                                                  const float2&          imageSize )
{
    return inverse( cameraToScreen ) * inverse( makePbrtScreenToRaster( screenWindow, imageSize ) );
}

OTK_INLINE OTK_HOSTDEVICE float3 transformPbrtPoint( const otk::Transform4& transform, const float3& point )
{
    const float4 result{ transform * make_float4( point.x, point.y, point.z, 1.0f ) };
    if( result.w == 1.0f )
    {
        return make_float3( result.x, result.y, result.z );
    }
    return make_float3( result.x, result.y, result.z ) / result.w;
}

OTK_INLINE OTK_HOSTDEVICE float3 transformPbrtVector( const otk::Transform4& transform, const float3& vector )
{
    const float4 result{ transform * make_float4( vector.x, vector.y, vector.z, 0.0f ) };
    return make_float3( result.x, result.y, result.z );
}

OTK_INLINE OTK_HOSTDEVICE PerspectiveCameraRay makePbrtPerspectiveCameraRay( const otk::Transform4& cameraToWorld,
                                                                             const otk::Transform4& rasterToCamera,
                                                                             const float3&          filmPosition )
{
    PerspectiveCameraRay ray;
    ray.cameraPoint = transformPbrtPoint( rasterToCamera, filmPosition );
    ray.origin      = transformPbrtPoint( cameraToWorld, make_float3( 0.0f, 0.0f, 0.0f ) );
    ray.direction   = transformPbrtVector( cameraToWorld, otk::normalize( ray.cameraPoint ) );
    return ray;
}

}  // namespace demandPbrtScene
