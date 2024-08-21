// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Params.h"

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
    float3 result = PARAMS_VAR_NAME.ambientColor * mat.Ka;
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
