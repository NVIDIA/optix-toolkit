// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cub/cub.cuh>

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

struct Or
{
    /// logical or operator, returns <tt>a | b</tt>
    template <typename T>
    __host__ __device__ __forceinline__ T operator()( const T& a, const T& b ) const
    {
        return a | b;
    }
};

// Bake the luminance texture to a state table, to be consumed by cuOmmBaking
__global__ void __launch_bounds__( 128 ) bakeLuminanceOpacity(
    uint32_t width,
    uint32_t height,
    uint32_t pitchInBytes,
    float transparencyCutoff,
    float opacityCutoff,
    cudaTextureObject_t texture,
    uint64_t* output )
{
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;

    const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if( y < height )
    {
        const float dx = 1 / ( float )width;
        const float dy = 1 / ( float )height;

        const float u = x * dx;
        const float v = y * dy;

        // load the luminance from the texture
        const float4 color = tex2D<float4>( texture, u, v );
        const float luminance = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;

        cuOmmBaking::OpacityState state = cuOmmBaking::OpacityState::STATE_UNKNOWN;
        if( luminance <= transparencyCutoff )
            state = cuOmmBaking::OpacityState::STATE_TRANSPARENT;
        else if( luminance >= opacityCutoff )
            state = cuOmmBaking::OpacityState::STATE_OPAQUE;

        // pack a warp worth of states into 64 bit using warp reduction and have a single thread write it out.
        uint64_t mask = ( ( uint64_t )state ) << ( ( x % 32 ) * 2 );

        typedef cub::WarpReduce<uint64_t> WarpReduce;

        // Allocate WarpReduce shared memory for 4 warps.
        __shared__ typename WarpReduce::TempStorage temp_storage[4];

        uint64_t aggregate = WarpReduce( temp_storage[warp] ).Reduce<Or>( mask, Or() );

        // The first lane writes out the packed 64 bit warp state vector.
        if( lane == 0 )
        {
            uint32_t byte = x / 4;
            *( uint64_t* )( ( uint64_t )output + byte + y * pitchInBytes ) = aggregate;
        }
    }
}

__host__ cudaError_t launchBakeLuminanceOpacity( 
    uint32_t width,
    uint32_t height,
    uint32_t pitchInBytes,
    float transparencyCutoff,
    float opacityCutoff,
    cudaTextureObject_t texture,
    uint64_t* output )
{
    // the output is padded so rows are a multiple of 32 texels.
    if( ( pitchInBytes % 8 ) != 0 )
        return cudaErrorInvalidPitchValue;

    if( pitchInBytes * 8 < width * 2 )
        return cudaErrorInvalidValue;

    if( opacityCutoff < transparencyCutoff )
        return cudaErrorInvalidValue;

    dim3 threadsPerBlock( 32, 4 );
    dim3 blocksPeGrid(
        ( uint32_t )( ( width + threadsPerBlock.x - 1 ) / threadsPerBlock.x ),
        ( uint32_t )( ( height + threadsPerBlock.y - 1 ) / threadsPerBlock.y ),
        1 );
    if( width && height )
        bakeLuminanceOpacity << <blocksPeGrid, threadsPerBlock, 0 >> > ( width, height, pitchInBytes, transparencyCutoff, opacityCutoff, texture, output );

    return cudaSuccess;
}
