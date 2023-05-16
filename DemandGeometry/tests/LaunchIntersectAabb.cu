//
//  Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
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
