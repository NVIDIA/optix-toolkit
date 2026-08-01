// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/FourierBsdfEval.h"

#include <optix.h>

namespace demandPbrtScene {

extern "C" __device__ float3 __direct_callable__fourierBsdfEvaluate( const FourierMaterialResource* resource,
                                                                     const float3                   outgoing,
                                                                     const float3                   incoming )
{
    if( resource == nullptr )
    {
        return make_float3( 0.0f, 0.0f, 0.0f );
    }
    return evaluateFourierBsdf( *resource, outgoing, incoming, FourierBsdfTransportMode::IMPORTANCE ).value;
}

extern "C" __device__ float __direct_callable__fourierBsdfPdf( const FourierMaterialResource* resource,
                                                               const float3                   outgoing,
                                                               const float3                   incoming )
{
    if( resource == nullptr )
    {
        return 0.0f;
    }
    return evaluateFourierBsdf( *resource, outgoing, incoming, FourierBsdfTransportMode::IMPORTANCE ).pdf;
}

}  // namespace demandPbrtScene
