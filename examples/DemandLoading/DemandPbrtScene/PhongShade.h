//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "Params.h"

#include <vector_functions.h>

namespace demandPbrtScene {

__forceinline__ __device__ float3 phongShadeDirectionalLight( const float3&        rayDir,
                                                              const PhongMaterial& mat,
                                                              const float3&        surfaceNormal,
                                                              const float3&        direction,
                                                              const float3&        color )
{
    // for directional lights the effect is simply the surface normal dot light direction
    float nDl = otk::dot( surfaceNormal, direction );

    if( nDl > 0.0f )
    {
        // perform the computation
        float3 phongRes = mat.Kd * nDl;

        float3 H   = otk::normalize( direction - rayDir );
        float  nDh = otk::dot( surfaceNormal, H );
        if( nDh > 0 )
        {
            float power = pow( nDh, mat.phongExp );
            phongRes += mat.Ks * power;
        }
        return phongRes * color;
    }
    return {};
}

__forceinline__ __device__ float3 phongShade( const PhongMaterial& mat, float3 const& surfaceNormal, float3 const& rayDir )
{
    const float3& ambientColor = PARAMS_VAR_NAME.ambientColor + float3{3.0f,3.0f,3.0f};
    float3 result = mat.Kd * ambientColor;
    {
        // directional light contribution
        const uint_t            numLights = PARAMS_VAR_NAME.numDirectionalLights;
        const DirectionalLight* lights    = PARAMS_VAR_NAME.directionalLights;
        for( uint_t i = 0; i < numLights; i++ )
        {
            result += phongShadeDirectionalLight( rayDir, mat, surfaceNormal, lights[i].direction, lights[i].color );
        }
    }
    {
        // infinite light contribution
        const uint_t         numLights = PARAMS_VAR_NAME.numInfiniteLights;
        const InfiniteLight* lights    = PARAMS_VAR_NAME.infiniteLights;
        for( uint_t i = 0; i < numLights; ++i )
        {
            result += mat.Kd * lights[i].color;
        }
    }

    return result;
}

}  // namespace demandPbrtScene
