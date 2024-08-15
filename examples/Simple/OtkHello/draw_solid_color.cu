// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "hello.h"

#include <OptiXToolkit/ShaderUtil/color.h>

#include <optix.h>

extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] =
        make_color( make_float3( rtData->r, rtData->g, rtData->b ) );
}
