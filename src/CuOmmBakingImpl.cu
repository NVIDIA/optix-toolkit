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

#include "CuOmmBakingImpl.h"
#include "Evaluate.h"
#include "Triangle.h"
#include "SummedAreaTable.h"

#include <assert.h>
#include <cub/cub.cuh>
#include <cuda.h>

namespace {

    // à la std::upper_bound, specialized for ForwardIt=uint32_t
    template <class T, typename Compare>
    __device__ uint32_t upper_bound( uint32_t first, uint32_t last, T value, Compare comp )
    {
        while( first < last )
        {
            uint32_t m = ( first + last ) / 2;
            if( !comp( value, m ) )
                first = m + 1;
            else
                last = m;
        };

        return first;
    }
};

// workaround for bug in optix_micromap.h
__device__ __host__ float __uint_as_float( unsigned int i )
{
    return *reinterpret_cast< float* >( &i );
}

__device__ OpacityStateSet sampleTextureState( const TextureInput* textures, Triangle triangle, unsigned resolution )
{
    const TextureData& texture = textures[triangle.texture].data;

    const float2 scale = { ( float )texture.width, ( float )texture.height };

    float2 uv0 = triangle.uv0 * scale;
    float2 uv1 = triangle.uv1 * scale;
    float2 uv2 = triangle.uv2 * scale;

    return sampleMemoryTexture( texture, uv0, uv1, uv2, texture.filterKernelRadiusInTexels, resolution );
}

#include <optix_micromap.h>

__global__ void setupBakeInput( SetupBakeInputParams params )
{
    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if( index < params.numTriangles )
    {
        TriangleID id = {};
        id.triangleIndex = index;
        id.inputIndex = params.inputIdx;

        Triangle triangle = loadTriangle( params.input, index );

        OpacityStateSet state = {};
        // filter out invalid triangles
        if( !isnan( triangle.Area() ) )
            state = sampleTextureState( params.textures, triangle, 16 );

        id.uniform = 1;
        if( params.format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE )
        {
            // conservatively mark everything as opaque except fully transparent triangles.
            if( state.isTransparent() )
            {
                id.state = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
            }
            else if( state.hasTransparent() )
            {
                // has a mixture of transparent and non-transparent states.
                id.uniform = 0;
            }
            else
            {
                // mixtures of opaque and unknown are marked as opaque.
                id.state = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            }
        }
        else // OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE
        {
            if( !state.isUniform() )
            {
                // has a mixture of states
                id.uniform = 0;
            }
            else
            {
                if( state.isTransparent() )
                {
                    id.state = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
                }
                else if( state.isOpaque() )
                {
                    id.state = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
                }
                else
                {
                    id.state = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
                }
            }
        }

        uint32_t key = 0;
        if( id.uniform == 0 )
        {
            id.uniform = 0;
            key = hash( canonicalizeTriangle( triangle, params.textures ) );
        }

        params.outTriangleIDs[index] = id;
        params.outHashKeys[index] = key;
    }
}

__host__ cudaError_t launchSetupBakeInput( SetupBakeInputParams params, cudaStream_t stream )
{
    unsigned numThreads = std::max<unsigned>( sizeof( BakeInput ), params.numTriangles );
    dim3     threadsPerBlock( 128, 1 );
    uint32_t numBlocks = ( uint32_t )( ( numThreads + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( numThreads )
        setupBakeInput<< <numBlocks, threadsPerBlock, 0, stream >> > ( params );
    return cudaGetLastError();
}

__global__ void markFirstOmmOccurance( MarkFirstOmmOccuranceParams params )
{
    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if( index < params.numTriangles )
    {
        bool isNewDuplicateGroup;

        TriangleID id = params.inTriangleIDs[index];

        // skip fully opaque/transparent triangles
        if( id.uniform )
        {
            isNewDuplicateGroup = false;
        }
        else
        {
            // early out hash check. if the hashes don't match there's no need to perform the costly collision check.
            if( index > 0 && params.inHashKeys[index - 1] == params.inHashKeys[index] )
            {
                TriangleID prevId = params.inTriangleIDs[index-1];

                Triangle nextTriangle = loadTriangle( params.inBakeInputs[id.inputIndex].desc, id.triangleIndex );
                Triangle prevTriangle = loadTriangle( params.inBakeInputs[prevId.inputIndex].desc, prevId.triangleIndex );

                if( prevTriangle.texture != nextTriangle.texture )
                {
                    isNewDuplicateGroup = true;
                }
                else
                {
                    // compare the canonicalized triangles to match near identical triangles and match under wrapping.
                    nextTriangle = canonicalizeTriangle( nextTriangle, params.inBakeInputs[id.inputIndex].inTextures );
                    prevTriangle = canonicalizeTriangle( prevTriangle, params.inBakeInputs[id.inputIndex].inTextures );

                    isNewDuplicateGroup = ( nextTriangle != prevTriangle );
                }                
            }
            else
            {
                isNewDuplicateGroup = true;
            }
        }

        params.outMarkers[index] = isNewDuplicateGroup ? 1 : 0;
    }
}

__host__ cudaError_t launchMarkFirstOmmOccurance( MarkFirstOmmOccuranceParams params, cudaStream_t stream )
{
    unsigned int numThreads = params.numTriangles;
    dim3     threadsPerBlock( 128, 1 );
    uint32_t numBlocks = ( uint32_t )( ( numThreads + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( numThreads )
        markFirstOmmOccurance << <numBlocks, threadsPerBlock, 0, stream >> > ( params );
    return cudaGetLastError();
}

__global__ void generateAssignment( GenerateAssignmentParams params )
{
    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if( index < params.numTriangles )
    {
        TriangleID id = params.inTriangleIDs[index];
        uint32_t   assignment;
        if( !id.uniform ) {
            assignment = params.inAssignment[index] - 1;

            // crude method to prevent omm array overflow by marking any excess omms as unkown. 
            if( assignment >= params.maxOmms )
            {
                assignment = ( uint32_t )OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_OPAQUE;
            }
            // write out a representative triangle id for each duplicate group
            else if( index == 0 || assignment != (params.inAssignment[index - 1] - 1) )
            {
                params.outOmmTriangleId[assignment] = id;

                const BakeInput& input = params.inBakeInputs[id.inputIndex];

                Triangle triangle = loadTriangle( input.desc, id.triangleIndex );

                // compute area in texels
                const TextureInput& textureInput = input.inTextures[triangle.texture];
                float2 scale = {
                    ( float )textureInput.data.width,
                    ( float )textureInput.data.height };
                triangle.uv0 *= scale;
                triangle.uv1 *= scale;
                triangle.uv2 *= scale;

                params.outOmmArea[assignment] = triangle.Area();
            }
        } else if( id.state == 0 ) {
            assignment = ( uint32_t )OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_TRANSPARENT;
        } else if( id.state == 1 ) {
            assignment = ( uint32_t )OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE;
        } else if( id.state == 2 ) {
            assignment = ( uint32_t )OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_TRANSPARENT;
        } else if( id.state == 3 ) {
            assignment = ( uint32_t )OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_OPAQUE;
        }

        void* outAssignments = params.inBakeInputs[id.inputIndex].outAssignments;
        if( params.indexFormat == cuOmmBaking::IndexFormat::I16_UINT )
            (( uint16_t* )outAssignments)[id.triangleIndex] = assignment;
        else
            (( uint32_t* )outAssignments)[id.triangleIndex] = assignment;
    }
    else if( index == params.numTriangles )
    {
        // copy total number of omms, clamped to the maximum number of omms
        *params.outNumOmms = index ? min( params.maxOmms, params.inAssignment[params.numTriangles - 1] ) : 0u;
    }
}

__host__ cudaError_t launchGenerateAssignment( GenerateAssignmentParams params, cudaStream_t stream )
{
    unsigned int numThreads = params.numTriangles + 1; // last thread copies the number of omms
    dim3     threadsPerBlock( 128, 1 );
    uint32_t numBlocks = ( uint32_t )( ( numThreads + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( numThreads )
        generateAssignment << <numBlocks, threadsPerBlock, 0, stream >> > ( params );
    return cudaGetLastError();
}

__global__ void __launch_bounds__(128) generateLayout( GenerateLayoutParams params )
{
    // Ideally, we'd launch one correctly sized kernel per omm.
    // However, due to deduplication we don't know the number of omms on the host.
    // Therefore we use persistant threads instead.

    // sizes are accumulated locally before a block-wide reduction and atomic accumulation into gmem
    uint32_t sizeInBytes = 0;

    {
        // atomically accumulate the subdivision level histogram in smem before atomically accumulate it into gmem
        __shared__ uint32_t histogram[OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL + 1];

        // initialize the histogram
        if( threadIdx.x < OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL + 1 )
            histogram[threadIdx.x] = 0;
        __syncthreads();

        uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t indexStride = blockDim.x * gridDim.x;

        const uint32_t numOmms = *params.inNumOmms;
        const float    sumArea = params.inSumArea ? *params.inSumArea : 0;

        const uint32_t logStatesPerByte = ( params.format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE ) ? 3 : 2;

        while( index < numOmms )
        {
            const float area = params.inOmmArea[index];

            // the normalized weight determines the available share of omm data in bytes for this omm.
            // the subdivision level is maximized within this available size bytes.
            // the size share needs to be conservative to prevent over allocation and buffer overflow due to numerical rounding.
            float normalizedWeight = ( sumArea > 0 )
                ? __fdiv_rz( area, sumArea )
                : __frcp_rd( ( float )numOmms );
            uint32_t maxSizeInBytes = ( uint32_t )floorf( __fmaf_rz( normalizedWeight, ( float )( params.maxOmmArraySizeInBytes - numOmms ), 1.f ) );
            uint32_t maxLogSizeInBytes = 31 - __clz( maxSizeInBytes );
            uint32_t maxSubdivisionLevel = ( maxLogSizeInBytes + logStatesPerByte ) / 2;

            uint32_t subdivisionLevel = maxSubdivisionLevel;

            // clamp the subdivision level based on the target micro-triangle density
            if( params.microTrianglesPerTexel )
            {
                float numMicroTriangles = area * params.microTrianglesPerTexel;

                uint32_t targetSubdivisionLevel = max( 1u, (uint32_t)roundf( 0.5f * log2f( numMicroTriangles ) ) );
                
                if( subdivisionLevel > targetSubdivisionLevel )
                    subdivisionLevel = targetSubdivisionLevel;
            }

            if( subdivisionLevel > OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL )
                subdivisionLevel = OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL;

            sizeInBytes += 1u << ( max( 2u * subdivisionLevel, logStatesPerByte ) - logStatesPerByte );

            assert( subdivisionLevel > 0 );
            assert( subdivisionLevel <= OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL );

            OptixOpacityMicromapDesc desc = {};
            desc.byteOffset = 0;
            desc.subdivisionLevel = subdivisionLevel;
            desc.format = params.format;

            params.ioDescs[index] = desc;

            // atomically accumulate histogram in shared memory
            atomicAdd( histogram + subdivisionLevel, 1u );

            index += indexStride;
        };

        // block wide accumulation of omm histogram
        __syncthreads();
        if( threadIdx.x < OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL + 1 && histogram[threadIdx.x] != 0 )
            atomicAdd( &params.ioHistogram[threadIdx.x].count, histogram[threadIdx.x] );
    }

    // block wide accumulation of omm size in bytes
    {
        typedef cub::BlockReduce<uint32_t, 128> BlockReduce;

        __shared__ typename BlockReduce::TempStorage temp_storage;

        uint32_t aggregate = BlockReduce( temp_storage ).Sum( sizeInBytes );

        if( threadIdx.x == 0 && aggregate )
            atomicAdd( params.ioSizeInBytes, aggregate );
    }
}

__host__ cudaError_t launchGenerateLayout( GenerateLayoutParams params, unsigned int numThreads, cudaStream_t stream )
{
    dim3     threadsPerBlock( 128, 1 );
    uint32_t numBlocks = ( uint32_t )( ( numThreads + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( numThreads )
        generateLayout << <numBlocks, threadsPerBlock, 0, stream >> > ( params );
    return cudaGetLastError();
}

struct ByteOffsetOutputIterator
{
    __device__ __host__ ByteOffsetOutputIterator() {};
    __device__ __host__ ByteOffsetOutputIterator( OptixOpacityMicromapDesc* _desc )
        : desc( _desc )
    { }

    using iterator_category = std::random_access_iterator_tag;
    using value_type = uint32_t;
    using difference_type = uint32_t;

    using pointer = uint32_t;
    using reference = uint32_t;

    OptixOpacityMicromapDesc* desc = 0;

    __device__ value_type operator=( uint32_t sum ) const
    {
        return desc->byteOffset = sum;
    }

    __device__ ByteOffsetOutputIterator operator+( size_t offset ) const { return ByteOffsetOutputIterator( desc + offset ); }

    __device__ ByteOffsetOutputIterator operator[]( size_t idx ) { return ByteOffsetOutputIterator( desc + idx ); }
};

struct OmmSizeInBytesInputIterator
{
    __device__ __host__ OmmSizeInBytesInputIterator() {};
    __device__ __host__ OmmSizeInBytesInputIterator( const OptixOpacityMicromapDesc* _desc, uint32_t _logStatesPerByte )
        : desc( _desc ), logStatesPerByte( _logStatesPerByte )
    {  }

    using iterator_category = std::random_access_iterator_tag;
    using value_type = uint32_t;
    using difference_type = uint32_t;

    using pointer = uint32_t;
    using reference = uint32_t;

    const OptixOpacityMicromapDesc* desc = 0;
    const uint32_t logStatesPerByte = 0;

    __device__ value_type operator[]( uint32_t offset ) const
    {
        uint32_t     ommIdx = offset;
        unsigned int sizeInBytes = ( 1u << ( max( 2u * desc[ommIdx].subdivisionLevel, logStatesPerByte ) - logStatesPerByte ) );
        return sizeInBytes ? sizeInBytes : 1;
    }

    __device__ value_type operator*() const { return operator[]( 0 ); }

    __device__ OmmSizeInBytesInputIterator operator+( uint32_t offset ) const { return OmmSizeInBytesInputIterator( desc + offset, logStatesPerByte ); }
};

__host__ cudaError_t launchGenerateStartOffsets( 
    void* temp,
    size_t& tempSizeInBytes,
    const OptixOpacityMicromapDesc* inDesc,
    OptixOpacityMicromapDesc* outDesc,
    unsigned int numItems,
    OptixOpacityMicromapFormat format,
    cudaStream_t stream )
{
    const uint32_t logStatesPerByte = ( format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE ) ? 3 : 2;

    OmmSizeInBytesInputIterator in( inDesc, logStatesPerByte );
    ByteOffsetOutputIterator out( outDesc );
    return cub::DeviceScan::ExclusiveSum<OmmSizeInBytesInputIterator, ByteOffsetOutputIterator>( temp, tempSizeInBytes, in, out, numItems, stream );
}

__global__ void __launch_bounds__( 128 ) GenerateInputHistogram( GenerateInputHistogramParams params )
{
    // atomically accumulate the subdivision level histogram in smem before atomically accumulate it into gmem
    __shared__ uint32_t histogram[OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL + 1];

    // initialize the histogram
    if( threadIdx.x < OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL + 1 )
        histogram[threadIdx.x] = 0;
    __syncthreads();

    uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if( index < params.numTriangles )
    {
        uint32_t assignment;
        if( params.indexFormat == cuOmmBaking::IndexFormat::I16_UINT )
        {
            uint16_t assignment16 = ( ( const uint16_t* )params.inAssignment )[index];

            // preserve predefined assignments
            if( assignment16 >= ( uint16_t )( -4 ) )
                assignment = (uint32_t)(int32_t)(int16_t)assignment16;
            else
                assignment = assignment16;
        }
        else
        {
            assignment = ( ( const uint32_t* )params.inAssignment )[index];
        }

        // skip predefined assignments
        if( assignment < ( uint32_t )( -4 ) )
        {
            uint32_t subdivisionLevel = params.inDescs[assignment].subdivisionLevel;
            atomicAdd( histogram + subdivisionLevel, 1u );
        }
    }

    // block wide accumulation of omm histogram
    __syncthreads();
    if( threadIdx.x < OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL + 1 && histogram[threadIdx.x] != 0 )
        atomicAdd( &params.ioHistogram[threadIdx.x].count, histogram[threadIdx.x] );
}

__host__ cudaError_t launchGenerateInputHistogram( GenerateInputHistogramParams params, cudaStream_t stream )
{
    unsigned int numThreads = params.numTriangles;
    dim3     threadsPerBlock( 128, 1 );
    uint32_t numBlocks = ( uint32_t )( ( numThreads + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( numThreads )
        GenerateInputHistogram << <numBlocks, threadsPerBlock, 0, stream >> > ( params );
    return cudaGetLastError();
}

__device__ OpacityStateSet evaluateMicroTriangleOpacity( const BakeInput* input, uint32_t subdivisionLevel, TriangleID id, uint32_t microTriangleIndex )
{
    float2 uv0, uv1, uv2;
    optixMicromapIndexToBaseBarycentrics(
        microTriangleIndex,
        subdivisionLevel,
        uv0, uv1, uv2 );

    Triangle triangle = loadTriangle( input[id.inputIndex].desc, id.triangleIndex );

    float2 du = triangle.uv1 - triangle.uv0;
    float2 dv = triangle.uv2 - triangle.uv0;

    // convert micro-triangle uvs to texture uvs
    triangle.uv1 = triangle.uv0 + uv1.x * du + uv1.y * dv;
    triangle.uv2 = triangle.uv0 + uv2.x * du + uv2.y * dv;
    triangle.uv0 += uv0.x * du + uv0.y * dv;

    return sampleTextureState( input[id.inputIndex].inTextures, triangle, 1 );
}

struct Or
{
    /// logical or operator, returns <tt>a | b</tt>
    template <typename T>
    __host__ __device__ __forceinline__ T operator()( const T& a, const T& b ) const
    {
        return a | b;
    }
};

__global__ void __launch_bounds__( 128 ) evaluateOmmOpacity( EvaluateOmmOpacityParams params )
{
    // Ideally, we'd launch one correctly sized kernel per subdivision level.
    // However, as we don't know the subdivision level counts on the host, we
    // iterate over all subdivision levels and use persistent threads to process
    // all mirco triangles in each subdivision level.

    uint64_t index        = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t indexStride  = blockDim.x * gridDim.x;
    
    uint32_t sizeInBytes = *params.inSizeInBytes;
    uint32_t numOmms     = *params.inNumOmms;
    
    assert( sizeInBytes <= params.dataSizeInBytes );

    const uint32_t logStatesPerByte = ( params.format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE ) ? 3 : 2;

    const uint64_t numMicroTriangles = ( uint64_t )sizeInBytes << logStatesPerByte;
    
    // iterate over all microtriangles in all omms at the current subdivision level.
    for( ; __any_sync(~0u, index < numMicroTriangles); index += indexStride )
    {
        OpacityStateSet state = {};

        if( index < numMicroTriangles )
        {
            uint32_t byteIndex = index >> logStatesPerByte;

            // Binary search in the prefix summed assignments to map the triangle index to a bake input.
            uint32_t ommIndex = upper_bound( 0, numOmms, byteIndex, [&]( uint32_t value, uint32_t element ) { return value < params.inDescs[element].byteOffset; } ) - 1;

            assert( ommIndex < numOmms );

            OptixOpacityMicromapDesc desc = params.inDescs[ommIndex];

            uint32_t microTriangleIndex = index - ( ( uint64_t )desc.byteOffset << logStatesPerByte );

            TriangleID id = params.inTriangleIdPerOmm[ommIndex];

            assert( !id.uniform );

            uint32_t subdivisionLevel = desc.subdivisionLevel;

            // 2-state subdiv level 0 and 1, and 4-state subdiv level 0 cover less than one byte.
            if( microTriangleIndex < ( 1u << ( 2 * subdivisionLevel ) ) )
            {
                state = evaluateMicroTriangleOpacity( params.inBakeInputs, subdivisionLevel, id, microTriangleIndex );
            }
        }

        uint32_t lane = threadIdx.x % 32;
        uint32_t warp = threadIdx.x / 32;

        if( params.format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE )
        {
            // all but fully transparent micro triangles are marked as opaque
            uint32_t opacityState;
            if( state.isTransparent() )
                opacityState = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
            else
                opacityState = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;

            uint32_t mask = ( uint32_t )opacityState << ( lane );

            typedef cub::WarpReduce<uint32_t> WarpReduce;

            // Allocate WarpReduce shared memory for 4 warps
            __shared__ typename WarpReduce::TempStorage temp_storage[4];

            // warp reduction and single write by head thread
            uint32_t aggregate = WarpReduce( temp_storage[warp] ).Reduce<Or>( mask, Or() );

            if( lane == 0 && aggregate )
            {
                uint32_t* address = ( uint32_t* )params.ioData + ( index / 32 );
                atomicOr( address, aggregate );
            }
        }
        else // OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE
        {
            uint32_t opacityState;
            if( state.isTransparent() )
                opacityState = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
            else if( state.isOpaque() )
                opacityState = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            else
                opacityState = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;

            uint64_t mask = ( uint64_t )opacityState << ( 2 * lane );

            typedef cub::WarpReduce<uint64_t> WarpReduce;

            // Allocate WarpReduce shared memory for 4 warps
            __shared__ typename WarpReduce::TempStorage temp_storage[4];

            // warp reduction and single write by head thread
            uint64_t aggregate = WarpReduce( temp_storage[warp] ).Reduce<Or>( mask, Or() );

            if( lane == 0 && aggregate )
            {
                unsigned long long* address = ( unsigned long long* )params.ioData + ( index / 32 );
                atomicOr( address, aggregate );
            }
        }
    };
}

__host__ cudaError_t launchEvaluateOmmOpacity( EvaluateOmmOpacityParams params, unsigned int numThreads, cudaStream_t stream )
{
    dim3     threadsPerBlock( 128, 1 );
    uint32_t numBlocks = ( uint32_t )( ( numThreads + threadsPerBlock.x - 1 ) / threadsPerBlock.x );
    if( numThreads )
        evaluateOmmOpacity<<<numBlocks, threadsPerBlock, 0, stream >>> ( params );
    return cudaGetLastError();
}

/**
 * \brief Rounding up sum functor
 */
struct SumRoundUp
{
    /// Boolean sum operator, returns <tt>a + b</tt>
    template <typename T>
    __device__ __forceinline__ T operator()( const T& a, const T& b ) const;
};

template <>
__device__ __forceinline__ float SumRoundUp::operator()( const float& a, const float& b ) const
{
    return __fadd_ru( a, b );
}

template<typename InputIteratorT, typename OutputIteratorT, typename T>
cudaError_t ReduceRoundUp<InputIteratorT, OutputIteratorT, T>::operator()(
    void*           d_temp_storage,
    size_t&         temp_storage_bytes,
    InputIteratorT  d_in,
    OutputIteratorT d_out,
    int             num_items,
    cudaStream_t    stream ) const
{
    return cub::DeviceReduce::Reduce<InputIteratorT, OutputIteratorT, SumRoundUp, T>( d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, SumRoundUp(), T{}, stream );
}

// explicit template instantiation
template class ReduceRoundUp<float*, float*, float>;

template <typename InputIteratorT, typename OutputIteratorT>
cudaError_t InclusiveSum<InputIteratorT, OutputIteratorT>::operator()( 
    void*           d_temp_storage,      ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&         temp_storage_bytes,  ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    InputIteratorT  d_in,                ///< [in] Pointer to the input sequence of data items
    OutputIteratorT d_out,               ///< [out] Pointer to the output sequence of data items
    int             num_items,           ///< [in] Total number of input items (i.e., the length of \p d_in)
    cudaStream_t    stream ) const
{
    return cub::DeviceScan::InclusiveSum<InputIteratorT, OutputIteratorT>( d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream );
}

// explicit template instantiation
template class InclusiveSum<uint32_t*, uint32_t*>;

template <typename KeyT, typename ValueT>
cudaError_t SortPairs<KeyT, ValueT>::operator()( 
    void*         d_temp_storage,      
    size_t&       temp_storage_bytes,
    const KeyT*   d_keys_in,     
    KeyT*         d_keys_out,          
    const ValueT* d_values_in, 
    ValueT*       d_values_out,      
    int           num_items,   
    int           begin_bit,   
    int           end_bit,     
    cudaStream_t  stream ) const
{
    return cub::DeviceRadixSort::SortPairs<KeyT, ValueT>( 
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream );
}

// explicit template instantiation
template class SortPairs<uint32_t, TriangleID>;
