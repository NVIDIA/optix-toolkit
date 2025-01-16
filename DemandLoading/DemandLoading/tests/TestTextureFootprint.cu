// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/DemandLoading/Texture2D.h>

#include "TestTextureFootprint.h"

#include <optix.h>

#if __CUDA_ARCH__ >= 600
#define SPARSE_TEX_SUPPORT true
#endif

using namespace demandLoading;

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__testFootprintGrad()
{
#ifdef SPARSE_TEX_SUPPORT
    uint3                  launchDim   = optixGetLaunchDimensions();
    uint3                  launchIndex = optixGetLaunchIndex();
    const FootprintInputs& inputs      = params.inputs[launchIndex.x];
    unsigned int           desc        = *reinterpret_cast<unsigned int*>( &params.sampler.desc );
    unsigned int           singleMipLevel;

    uint4 fineFootprint = optixTexFootprint2DGrad( params.sampler.texture, desc, inputs.x, inputs.y, inputs.dPdx_x, inputs.dPdx_y,
                                                   inputs.dPdy_x, inputs.dPdy_y, FINE_MIP_LEVEL, &singleMipLevel );
    params.outputs[launchIndex.x] = fineFootprint;

    uint4 coarseFootprint = optixTexFootprint2DGrad( params.sampler.texture, desc, inputs.x, inputs.y, inputs.dPdx_x, inputs.dPdx_y,
                                                     inputs.dPdy_x, inputs.dPdy_y, COARSE_MIP_LEVEL, &singleMipLevel );
    params.outputs[launchIndex.x + launchDim.x] = coarseFootprint;

    requestTexFootprint2DGrad( params.sampler, params.referenceBits, params.residenceBits, inputs.x, inputs.y, inputs.dPdx_x,
                               inputs.dPdx_y, inputs.dPdy_x, inputs.dPdy_y, 0.0f, 0.0f );
#endif
}

extern "C" __global__ void __raygen__testFootprintLod()
{
#ifdef SPARSE_TEX_SUPPORT
    uint3           launchDim   = optixGetLaunchDimensions();
    uint3           launchIndex = optixGetLaunchIndex();
    FootprintInputs inputs      = params.inputs[launchIndex.x];
    unsigned int    desc        = *reinterpret_cast<unsigned int*>( &params.sampler.desc );
    unsigned int    singleMipLevel;

    uint4 fineFootprint =
        optixTexFootprint2DLod( params.sampler.texture, desc, inputs.x, inputs.y, inputs.level, FINE_MIP_LEVEL, &singleMipLevel );
    params.outputs[launchIndex.x] = fineFootprint;

    uint4 coarseFootprint =
        optixTexFootprint2DLod( params.sampler.texture, desc, inputs.x, inputs.y, inputs.level, COARSE_MIP_LEVEL, &singleMipLevel );
    params.outputs[launchIndex.x + launchDim.x] = coarseFootprint;

    requestTexFootprint2DLod( params.sampler, params.referenceBits, params.residenceBits, inputs.x, inputs.y, inputs.level );
#endif
}

extern "C" __global__ void __raygen__testFootprint()
{
#ifdef SPARSE_TEX_SUPPORT
    uint3           launchDim   = optixGetLaunchDimensions();
    uint3           launchIndex = optixGetLaunchIndex();
    FootprintInputs inputs      = params.inputs[launchIndex.x];
    unsigned int    desc        = *reinterpret_cast<unsigned int*>( &params.sampler.desc );
    unsigned int    singleMipLevel;

    uint4 fineFootprint = optixTexFootprint2DLod( params.sampler.texture, desc, inputs.x, inputs.y, 0.0f, FINE_MIP_LEVEL, &singleMipLevel );
    params.outputs[launchIndex.x] = fineFootprint;

    uint4 coarseFootprint =
        optixTexFootprint2DLod( params.sampler.texture, desc, inputs.x, inputs.y, 0.0f, COARSE_MIP_LEVEL, &singleMipLevel );
    params.outputs[launchIndex.x + launchDim.x] = coarseFootprint;

    requestTexFootprint2DLod( params.sampler, params.referenceBits, params.residenceBits, inputs.x, inputs.y, inputs.level );
#endif
}
