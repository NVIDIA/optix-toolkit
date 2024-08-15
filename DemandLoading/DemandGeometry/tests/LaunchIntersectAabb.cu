// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "LaunchIntersectAabb.h"

#include <OptiXToolkit/DemandGeometry/intersectAabb.h>

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>

__global__ void deviceLaunchIntersectAabb( Intersection* data )
{
    data->intersected = demandGeometry::intersectAabb( data->rayOrigin, data->rayDir, data->rayTMin, data->rayTMax,
                                                       data->aabb, data->tIntersect, data->normal, data->face );
}

__host__ void launchIntersectAabb( Intersection& data )
{
    void* devData{};
    cudaMalloc( &devData, sizeof( Intersection ) );
    cudaMemcpy( devData, &data, sizeof( Intersection ), cudaMemcpyHostToDevice );
    const int numBlocks          = 1;
    const int numThreadsPerBlock = 1;
    deviceLaunchIntersectAabb<<<numBlocks, numThreadsPerBlock>>>(
        static_cast<Intersection*>( devData ) );
    cudaMemcpy( &data, devData, sizeof( Intersection ), cudaMemcpyDeviceToHost );
    cudaFree( devData );
}
