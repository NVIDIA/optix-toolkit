// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace sat
{
    #define SCAN_BLOCK_DIM            256
    #define TRANSPOSE_WARPS_PER_BLOCK 8

    #define TILE_SIZE      32
    #define TILE_SIZE_MASK (TILE_SIZE - 1)

    template <typename InputFunctorT, typename OutputFunctorT>
    struct ScanLinesParams
    {
        // number of lines
        uint32_t lines;
        // length of each line
        uint32_t length;
        // data input functor
        InputFunctorT  input;
        // data output functor
        OutputFunctorT output;
    };

    template <typename Ty>
    struct TransposeTilesParams
    {
        // input data
        Ty* data;
        // size in pixels. must each be multiple of TILE_SIZE.
        uint32_t width;
        uint32_t height;
    };

    template <typename InputFunctorT, typename OutputFunctorT>
    cudaError_t launchScanLines( ScanLinesParams<InputFunctorT, OutputFunctorT> params, cudaStream_t stream );

    template <typename Ty>
    cudaError_t launchTransposeTiles( TransposeTilesParams<Ty> params, cudaStream_t stream );


    template <typename InTy, typename OutTy>
    inline __device__ OutTy cvrt( InTy v )
    {
        return ( OutTy )v;
    }

    template <typename Ty>
    inline __device__ Ty cvrt( Ty v )
    {
        return v;
    }

    template <>
    inline __device__ uint2 cvrt<ushort2, uint2>( ushort2 v )
    {
        return uint2{ v.x, v.y };
    }

    template <typename OutputIteratorT>
    struct ConditionalOutputItr
    {
        // The output value type
        using OutputT = typename std::iterator_traits<OutputIteratorT>::value_type;

        __device__ ConditionalOutputItr()
            : write( false ) {};

        __device__ ConditionalOutputItr( OutputIteratorT _outputItr )
            : write( true ), outputItr(_outputItr ) {};

        __device__ void operator=( const OutputT& value )
        {
            if( write )
                *outputItr = value;
        }

        bool            write;
        OutputIteratorT outputItr;
    };

    template <typename IteratorT>
    struct Pitched2DFunctor
    {
        struct config_t
        {
            uint32_t  pitch    = {};
            IteratorT iterator = {};

            config_t() = default;
            config_t( uint32_t _pitch, IteratorT _iterator )
                : pitch( _pitch )
                , iterator( _iterator )
            {
            }
        } config;

        __host__ __device__ Pitched2DFunctor() {};
        __host__ __device__ Pitched2DFunctor( config_t _config )
            : config( _config ) {};

        __device__ IteratorT operator()( uint32_t idx, uint32_t line )
        {
            return config.iterator + ( idx + line * config.pitch );
        }
    };

    template <typename InputIteratorT>
    struct TileTransposedInputFunctor
    {
        struct config_t
        {
            uint32_t       width = {};
            InputIteratorT table = {};
        } config;

        using value_type = typename std::iterator_traits<InputIteratorT>::value_type;

        __host__ __device__ TileTransposedInputFunctor() {};
        __host__ __device__ TileTransposedInputFunctor( config_t _config )
            : config( _config ) {};

        __device__ InputIteratorT operator()( uint32_t idx, uint32_t line )
        {
            const uint32_t x = line;
            const uint32_t y = idx;

            const unsigned int tx = x & ( ~TILE_SIZE_MASK );
            const unsigned int ty = y & ( ~TILE_SIZE_MASK );

            const unsigned int px = ( x & TILE_SIZE_MASK );
            const unsigned int py = ( y & TILE_SIZE_MASK );

            // transpose within 32x32 tiles for coherent data access
            const unsigned int sx = tx | py;
            const unsigned int sy = ty | px;

            uint32_t src_idx = sx + sy * config.width;

            return config.table + src_idx;
        }
    };

    template <typename InputFunctorT, typename OutputFunctorT, uint32_t BLOCK_DIM = 128>
    __global__ void scanLines( ScanLinesParams<InputFunctorT, OutputFunctorT> p )
    {
        int lane = threadIdx.x;
        int line = blockIdx.x;

        if( line >= p.lines )
            return;

        // The input value type
        using InputIteratorT = typename std::result_of<InputFunctorT( uint32_t, uint32_t )>::type;
        
        // The output value type
        using InputT = typename std::iterator_traits<InputIteratorT>::value_type;

        // The output iterator type
        using OutputIteratorT = typename std::result_of<OutputFunctorT( uint32_t, uint32_t )>::type;

        // The output value type
        using OutputT = typename std::iterator_traits<OutputIteratorT>::value_type;

        // Specialize BlockScan for a 1D block of 128 threads on type int
        typedef cub::BlockScan<OutputT, BLOCK_DIM> BlockScan;

        union smem_t
        {
            // Allocate shared memory for BlockScan
            typename BlockScan::TempStorage temp_storage;
            OutputT                         block_aggregate;
        };

        __shared__ smem_t smem;

        OutputT sum = {};
        for( uint32_t base = 0; base < p.length; base += BLOCK_DIM )
        {
            uint32_t idx = base + lane;

            // read value
            OutputT input = {};
            if( idx < p.length )
            {
                InputT tmp = *p.input( idx, line );
                input = cvrt<InputT, OutputT>( tmp );
            }

            OutputT output = {};

            // Collectively compute the block-wide exclusive prefix sum
            BlockScan( smem.temp_storage ).InclusiveSum( input, output );

            // write value
            if( idx < p.length && line < p.lines )
            {
                *p.output( idx, line ) = sum + output;
            }

            // last lane writes the block aggregate
            if( lane == ( BLOCK_DIM - 1 ) )
                smem.block_aggregate = output;

            __syncthreads();

            // all threads read the block aggregate
            sum += smem.block_aggregate;
        }
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t launchScanLines( ScanLinesParams<InputIteratorT, OutputIteratorT> params, cudaStream_t stream )
    {
        dim3     threadsPerBlock( SCAN_BLOCK_DIM, 1 );
        uint32_t numPixels = params.length * params.lines;
        uint32_t numBlocks = params.lines;
        if( numPixels )
            scanLines<InputIteratorT, OutputIteratorT, /*BLOCK_DIM=*/SCAN_BLOCK_DIM> << <numBlocks, threadsPerBlock, 0, stream >> > ( params );
        return cudaGetLastError();
    }

    template <typename Ty, unsigned int WARPS_PER_BLOCK>
    __global__ void transposeTiles( const TransposeTilesParams<Ty> p )
    {
        // one tile per block
        __shared__ Ty s[TILE_SIZE * TILE_SIZE];

        const unsigned int LINES_PER_WARP = TILE_SIZE / WARPS_PER_BLOCK;

        uint32_t widx = threadIdx.y;
        uint32_t lidx = threadIdx.x;

        uint32_t b_ofs = ( blockIdx.y * p.width + blockIdx.x ) * TILE_SIZE;

        // load tile lines in smem
#pragma unroll
        for( uint32_t i = 0; i < LINES_PER_WARP; i++ )
        {
            uint32_t idx = ( widx + i * WARPS_PER_BLOCK );
            uint32_t g_idx = b_ofs + idx * p.width + lidx;
            // rotate to prevent bank conflicts in the transpose
            uint32_t s_idx = idx * TILE_SIZE + ( ( lidx + idx ) & TILE_SIZE_MASK );

            s[s_idx] = p.data[g_idx];
        }

        __syncthreads();

        // store transposed tile lines back to gmem
#pragma unroll
        for( uint32_t i = 0; i < LINES_PER_WARP; i++ )
        {
            uint32_t idx = ( widx + i * WARPS_PER_BLOCK );
            uint32_t g_idx = b_ofs + idx * p.width + lidx;
            // rotate to prevent bank conflicts in the transpose
            uint32_t s_idx = ( ( idx + lidx ) & TILE_SIZE_MASK ) + lidx * TILE_SIZE;

            p.data[g_idx] = s[s_idx];
        }
    }

    template <typename Ty>
    cudaError_t launchTransposeTiles( TransposeTilesParams<Ty> params, cudaStream_t stream )
    {
        assert( params.width % TILE_SIZE == 0 );
        assert( params.height % TILE_SIZE == 0 );
        dim3     threadsPerBlock( TILE_SIZE, TRANSPOSE_WARPS_PER_BLOCK );
        unsigned blockSizeX = params.width / TILE_SIZE;
        unsigned blockSizeY = params.height / TILE_SIZE;
        dim3     blocksPerGrid( blockSizeX, blockSizeY );
        if( blockSizeX && blockSizeY )
            transposeTiles<Ty, /*WARPS_PER_BLOCK=*/TRANSPOSE_WARPS_PER_BLOCK> << <blocksPerGrid, threadsPerBlock, 0, stream >> > ( params );
        return cudaGetLastError();
    }

    inline __host__ __device__ unsigned roundToPowerOfTwo( unsigned val, unsigned power )
    {
        return ( val + power - 1 ) & ( ~( power - 1 ) );
    }

    template <typename InputFunctorT, typename OutputFunctorT>
    cudaError_t TransposedSummedAreaTable(
        void*           d_temp_storage,
        size_t&         temp_storage_bytes,
        unsigned        width,
        unsigned        height,
        InputFunctorT   input,
        OutputFunctorT  output,
        cudaStream_t    stream )
    {

        // The input value type
        using InputIteratorT = typename std::result_of<InputFunctorT( uint32_t, uint32_t )>::type;

        // The output value type
        using InputT = typename std::iterator_traits<InputIteratorT>::value_type;

        // The output iterator type
        using OutputIteratorT = typename std::result_of<OutputFunctorT( uint32_t, uint32_t )>::type;

        // The output value type
        using OutputT = typename std::iterator_traits<OutputIteratorT>::value_type;

        unsigned paddedWidth = roundToPowerOfTwo( width, TILE_SIZE );
        unsigned paddedHeight = roundToPowerOfTwo( height, TILE_SIZE );

        size_t required_temp_storage_bytes = sizeof( InputT ) * paddedWidth * paddedHeight;

        if( temp_storage_bytes == 0 )
        {
            temp_storage_bytes = required_temp_storage_bytes;
            return cudaSuccess;
        }

        if( temp_storage_bytes < required_temp_storage_bytes )
            return cudaErrorInvalidValue;

        if( paddedWidth == 0 || paddedHeight == 0 )
            return cudaSuccess;

        // padd so the table can be split in 32x32 tiles
        {
            typename Pitched2DFunctor<InputT*>::config_t outConfig{};
            outConfig.pitch    = paddedWidth;
            outConfig.iterator = (InputT*)d_temp_storage;
            Pitched2DFunctor<InputT*> horizontalOutput( outConfig );

            ScanLinesParams<InputFunctorT, Pitched2DFunctor<InputT*>> p = {};
            p.length = width;
            p.lines = height;
            p.input = input;
            p.output = horizontalOutput;

            cudaError_t error = launchScanLines( p, stream );
            if( error != cudaSuccess )
                return error;
        }

        // intra-transpose 32x32 tiles. tiles themselves are left in place.
        {
            TransposeTilesParams<InputT> p = {};
            p.data = ( InputT* )d_temp_storage;
            p.width = paddedWidth;
            p.height = paddedHeight;
            cudaError_t error = launchTransposeTiles( p, stream );
            if( error != cudaSuccess )
                return error;
        }

        // scan vertically through block transposed padded data
        {
            typename TileTransposedInputFunctor<InputT*>::config_t inConfig = {};
            inConfig.table = ( InputT* )d_temp_storage;
            inConfig.width = paddedWidth;
            TileTransposedInputFunctor<InputT*> verticalInput( inConfig );

            ScanLinesParams<TileTransposedInputFunctor<InputT*>, OutputFunctorT> p = {};
            p.length = height;
            p.lines  = width;
            p.input  = verticalInput;
            p.output = output;
            cudaError_t error = launchScanLines( p, stream );
            if( error != cudaSuccess )
                return error;
        }

        return cudaSuccess;
    }

    template <typename InputFunctorT, typename OutputT>
    cudaError_t TransposedSummedAreaTable(
        void*           d_temp_storage,
        size_t&         temp_storage_bytes,
        unsigned        width,
        unsigned        height,
        InputFunctorT   input,
        unsigned        pitch,
        OutputT*        output,
        cudaStream_t    stream )
    {
        return TransposedSummedAreaTable(
            d_temp_storage,
            temp_storage_bytes,
            width,
            height,
            input,
            Pitched2DFunctor<OutputT*>( { pitch, output } ),
            stream );
    }

    /// <summary>
    /// Computes a transposed summed area table. 
    /// 
    /// Performance notes:
    /// The table is first summed horizontally. The horizontally accumulated value is of the input value type. 
    /// This horizontally summed table is then summed vertically. The vertically accumulated value is of the output value type. 
    /// To prevent overfows, the precision of the input value type must be sufficient for horizontal summation.
    /// The required temporary memory is proportional to the size of the input value type. Choosing the smallest 
    /// input value type sufficient to represent the horizontally summed values generally results in the best perfomrnace 
    /// and least required temporary memory.
    /// The output functor receives the output values in transposed order. Outputting threads are mapped to the output table in column-major order.
    /// Therefore writing out the transposed table will result in coalesced memory writes.
    /// </summary>
    /// <typeparam name="InputFunctorT">Input functor of type InputIteratorT(unsigned x, unsigned y). Reading threads are mapped to the input table in row-major order.</typeparam>
    /// <typeparam name="OutputFunctorT">Ouput functor of type OutputIteratorT(unsigned y, unsigned x). Writing threads are mapped to the output table in column-major order.</typeparam>
    /// <param name="d_temp_storage">Temporary memory buffer pointer.</param>
    /// <param name="temp_storage_bytes">Temporary memory buffer size. If left zero, the function will compute and set the required memory size without executing any work.</param>
    /// <param name="input">Input functor.</param>
    /// <param name="output">Output functor.</param>
    /// <param name="width">Width of the table in elements.</param>
    /// <param name="height">Height of the table in elements.</param>
    /// <param name="stream">Cuda stream all the work is launched on.</param>
    /// <returns></returns>
    template <typename InputFunctorT, typename OutputFunctorT>
    cudaError_t TransposedSummedAreaTable(
        void*           d_temp_storage,
        size_t&         temp_storage_bytes,
        unsigned        width,
        unsigned        height,
        InputFunctorT   input,
        OutputFunctorT  output,
        cudaStream_t    stream );

    template <typename InputFunctorT, typename OutputT>
    cudaError_t TransposedSummedAreaTable(
        void*           d_temp_storage,
        size_t&         temp_storage_bytes,
        unsigned        width,
        unsigned        height,
        InputFunctorT   input,
        unsigned        pitch,
        OutputT*        output,
        cudaStream_t    stream );

}; // namespace sat
