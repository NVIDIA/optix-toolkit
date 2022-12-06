#include <cub/cub.cuh>

#include "Procedural.h"

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

// Bake the procedural texture to a state table, to be consumed by cuOmmBaking
__global__ void __launch_bounds__( 128 ) bakeProcedural(
    uint32_t width,
    uint32_t height,
    uint32_t pitchInBytes,
    float transpacentyCutoff,
    float opacityCutoff,
    uint64_t* output )
{
    uint32_t lane = threadIdx.x;
    uint32_t warp = threadIdx.y;

    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if( y < height )
    {
        float dx = 1.f / ( float )width;
        float dy = 1.f / ( float )height;

        interval_t ix( x * dx, ( x + 1.f ) * dx );
        interval_t iy( y * dy, ( y + 1.f ) * dy );

        vec2<interval_t> area( ix, iy );
        interval_t alpha = eval_procedural( area );

        cuOmmBaking::OpacityState state = cuOmmBaking::OpacityState::STATE_UNKNOWN;
        if( alpha.hi <= transpacentyCutoff )
            state = cuOmmBaking::OpacityState::STATE_TRANSPARENT;
        else if( alpha.lo >= opacityCutoff )
            state = cuOmmBaking::OpacityState::STATE_OPAQUE;

        // pack a warp worth of states into 64 bit using warp reduction and have a single thread write it out.
        uint64_t mask = ( ( uint64_t )state ) << ( ( x % 32 ) * 2 );

        typedef cub::WarpReduce<uint64_t> WarpReduce;

        // Allocate WarpReduce shared memory for 4 warps
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

__host__ void launchEvaluateOmmOpacity( 
    uint32_t width,
    uint32_t height,
    uint32_t pitchInBytes,
    float transpacentyCutoff,
    float opacityCutoff,
    uint64_t* output )
{
    dim3 threadsPerBlock( 32, 4 );
    dim3 blocksPeGrid(
        ( uint32_t )( ( width + threadsPerBlock.x - 1 ) / threadsPerBlock.x ),
        ( uint32_t )( ( height + threadsPerBlock.y - 1 ) / threadsPerBlock.y ),
        1 );
    if( width && height )
        bakeProcedural << <blocksPeGrid, threadsPerBlock, 0 >> > ( width, height, pitchInBytes, transpacentyCutoff, opacityCutoff, output );
}
