//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <DemandLoading/Texture2D.h>

#include "TestTextureFootprint.h"

#include <optix.h>

using namespace demandLoading;

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__testFootprintGrad()
{
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
                               inputs.dPdx_y, inputs.dPdy_x, inputs.dPdy_y );
}

extern "C" __global__ void __raygen__testFootprintLod()
{
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
}

extern "C" __global__ void __raygen__testFootprint()
{
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
}
