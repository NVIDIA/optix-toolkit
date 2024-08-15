// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "BakeTexture.h"

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

__global__ void textureToState( const TextureToStateParams p )
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t x = idx % p.width;
    uint32_t y = idx / p.width;

    float tu = x + 0.5f;
    float tv = y + 0.5f;

    if( p.isNormalizedCoords )
    {
        tu *= ( 1.f / p.width );
        tv *= ( 1.f / p.height );
    }

    if( y < p.height )
    {
        float value;
        if( p.isRgba )
            value = tex2D<float4>( p.tex, tu, tv ).w;
        else
            value = tex2D<float>( p.tex, tu, tv );

        unsigned int state = ( unsigned int )cuOmmBaking::OpacityState::STATE_UNKNOWN;
        if( value <= p.transparencyCutoff )
            state = ( unsigned int )cuOmmBaking::OpacityState::STATE_TRANSPARENT;
        else if( value >= p.opacityCutoff )
            state = ( unsigned int )cuOmmBaking::OpacityState::STATE_OPAQUE;

        const uint32_t bit = 2 * x + p.pitchInBits * y;
        const uint32_t word = ( bit / 32 );
        const uint32_t mask = ( state << ( bit & 31 ) );

        atomicOr( p.buffer + word, mask );
    }
}

cudaError_t launchTextureToState( const TextureToStateParams params, cudaStream_t stream )
{
   dim3     threadsPerBlock( 128, 1 );
    uint32_t numTexels = params.width * params.height;
    uint32_t numBlocks = ( uint32_t )( ( numTexels + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( numTexels )
        textureToState << <numBlocks, threadsPerBlock, 0, stream >> > ( params );
    return cudaGetLastError();
}