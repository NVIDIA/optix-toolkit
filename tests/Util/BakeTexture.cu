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