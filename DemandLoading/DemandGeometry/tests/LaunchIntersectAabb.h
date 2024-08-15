// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>

#include <vector_types.h>

struct Intersection
{
    float3 rayOrigin;
    float3 rayDir;
    float rayTMin;
    float rayTMax;
    OptixAabb aabb;
    float tIntersect;
    float3 normal;
    int face;
    bool intersected;
};

__host__ void launchIntersectAabb( Intersection& data );
