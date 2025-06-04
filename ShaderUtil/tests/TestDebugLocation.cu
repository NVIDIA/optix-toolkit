// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/DebugLocation.h>

#include "TestDebugLocationParams.h"

#include <optix.h>

#include <vector_functions.h>

extern "C" {
__constant__ otk::shaderUtil::testing::Params g_params;
}

__device__ __forceinline__ void setPixel( float r, float g, float b )
{
    const uint3  launchIndex = optixGetLaunchIndex();
    const uint_t index       = launchIndex.y * g_params.width + launchIndex.x;
    g_params.image[index]    = make_float3( r, g, b );
}

struct Callback
{
    __device__ __forceinline__ void dump( const uint3& pos ) const
    {
        ++g_params.dumpIndicator[0];
    }

    __device__ __forceinline__ void setColor( float r, float g, float b ) const
    {
        setPixel( r, g, b );
    }
};

extern "C" __global__ void __raygen__debugLocationTest()
{
    if( otk::debugInfoDump( g_params.debug, Callback{} ) )
        return;

    setPixel( g_params.miss.x, g_params.miss.y, g_params.miss.z );
}

extern "C" __global__ void __closesthit__debugLocationTest() {}

extern "C" __global__ void __intersection__debugLocationTest() {}

extern "C" __global__ void __miss__debugLocationTest() {}
