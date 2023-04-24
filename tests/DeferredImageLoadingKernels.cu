//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
