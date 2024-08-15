// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DeferredImageLoadingKernels.h"

#include <OptiXToolkit/DemandLoading/Texture2D.h>

#include <optix.h>

#include <vector_types.h>

__constant__ Params params;

extern "C" __global__ void __raygen__sampleTexture()
{
    const uint3 launchIndex = optixGetLaunchIndex();
    if( launchIndex.x >= params.m_width || launchIndex.y >= params.m_height )
    {
        printf("launchIndex %u,%u %ux%u\n", launchIndex.x, launchIndex.y, params.m_width, params.m_height );
        return;
    }

    const float s = launchIndex.x / (float)params.m_width;
    const float t = launchIndex.y / (float)params.m_height;

    bool         isResident = false;
    const float4 pixel      = demandLoading::tex2DLod<float4>( params.m_context, params.m_textureId, s, t, 0.0f, &isResident );

    const RayGenData* rayGenData = reinterpret_cast<RayGenData*>( optixGetSbtDataPointer() );
    params.m_output[launchIndex.y * params.m_width + launchIndex.x] = isResident ? pixel : rayGenData->m_nonResidentColor;
}
