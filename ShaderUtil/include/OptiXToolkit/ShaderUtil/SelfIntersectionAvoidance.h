// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

// #define OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES // Disables support for optix transform traversables (motion matrix, motion srt and static matrix). Only instances are supported.

/// \file SelfIntersectionAvoidance.h
/// Primary interface of Self Intersection Avoidance library.
///
/// Example use:
///
///     ...
///
///     float3 objPos, objNorm;
///     float objOffset;
///
///     // generate object space spawn point and offset
///     getSafeTriangleSpawnOffset( objPos, objNorm, objOffset, ... );
///
///     float3 wldPos, wldNorm;
///     float wldOffset;
///
///     // convert object space spawn point and offset to world space
///     transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset, ... );
///
///     float3 front, back;
///     // offset world space spawn point to generate self intersection safe front and back spawn points
///     offsetSpawnPoint( front, back, wldPos, wldNorm, wldOffset );
///
///     // flip normal to point towards incoming direction
///     if( dot( wldNorm, wldInDir ) > 0.f )
///     {
///         wldNorm = -wldNorm;
///         swap( front, back );
///     }
///     ...
///
///     // pick safe spawn point for secondary scatter ray
///     float3 wldOutPos = ( dot( wldOutDir, wldNorm ) > 0.f ) ? front : back
/// 

#include <OptiXToolkit/ShaderUtil/CudaSelfIntersectionAvoidance.h>
#include <OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h>
