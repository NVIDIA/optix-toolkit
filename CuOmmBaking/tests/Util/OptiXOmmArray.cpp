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

#include "OptiXOmmArray.h"
#include "Util/BufferLayout.h"

#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>

#define OPTIX_CHECK( x )                                                                                                                                                                                                                                                               \
    {                                                                                                                                                                                                                                                                                  \
        OptixResult err = ( OptixResult )( x );                                                                                                                                                                                                                                        \
        if( err != OPTIX_SUCCESS )                                                                                                                                                                                                                                                     \
            return cuOmmBaking::Result::ERROR_CUDA;                                                                                                                                                                                                                                                 \
    };

#define CUDA_CHECK( x )                                                                                                                                                                                                                                                                \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t err = (cudaError_t)( x );                                                                                                                                                                                                                                          \
        if( err != cudaSuccess )                                                                                                                                                                                                                                                       \
            return cuOmmBaking::Result::ERROR_CUDA;                                                                                                                                                                                                                                                 \
    };

#define OMM_CHECK( x )                                                                                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                                                                  \
        cuOmmBaking::Result err = ( x );                                                                                                                                                                                                                                                            \
        if( err != cuOmmBaking::Result::SUCCESS )                                                                                                                                                                                                                                                   \
            return err;                                                                                                                                                                                                                                                                \
    };

namespace {

    std::ostream& operator<<( std::ostream& os, const unsigned char c )
    {
        os << std::hex << ( uint32_t )c;
        return os;
    }

    std::ostream& operator<<( std::ostream& os, const OptixOpacityMicromapDesc &desc )
    {
        os << "byteOffset=" << desc.byteOffset << ", format=" << desc.format << ", subdivisionLevel=" << desc.subdivisionLevel;
        return os;
    }

};

cuOmmBaking::Result OptixOmmArray::create( OptixDeviceContext context, const cuOmmBaking::BakeOptions& options, const cuOmmBaking::BakeInputDesc* inputs, unsigned int numInputs )
{
    OMM_CHECK( destroy() );

    if( !inputs )
        return cuOmmBaking::Result::ERROR_INVALID_VALUE;

    if( numInputs == 0 )
        return cuOmmBaking::Result::ERROR_INVALID_VALUE;

    std::vector<cuOmmBaking::BakeInputBuffers> inputBuffers;
    inputBuffers.resize( numInputs );

    cuOmmBaking::BakeBuffers buffers = {};
    OMM_CHECK( cuOmmBaking::GetPreBakeInfo( &options, numInputs, inputs, inputBuffers.data(), &buffers ));
    
    CuBuffer<>                          d_optixOmmArray;
    CuBuffer<>                          d_indices;
        
    cuOmmBaking::IndexFormat indexFormat = buffers.indexFormat;

    std::vector<size_t>   inputIndexBufferOffsetInBytes;
    std::vector<size_t>   inputUsageDescOffset;
    size_t outOmmIndexBufferSizeInBytes = 0;
    size_t outOmmUsageDescSize = 0;
    for( unsigned i = 0; i < numInputs; ++i )
    {
        inputIndexBufferOffsetInBytes.push_back( outOmmIndexBufferSizeInBytes );
        outOmmIndexBufferSizeInBytes += inputBuffers[i].indexBufferSizeInBytes;

        inputUsageDescOffset.push_back( outOmmUsageDescSize );
        outOmmUsageDescSize += inputBuffers[i].numMicromapUsageCounts;
    }
    inputUsageDescOffset.push_back( outOmmUsageDescSize );

    CUDA_CHECK( d_indices.alloc( outOmmIndexBufferSizeInBytes ) );
    
    HistogramEntries_t           histogram;
    cuOmmBaking::PostBakeInfo  postInfo = {};

    uint32_t usageCount = 0;
    std::vector<UsageCounts_t>               usage( numInputs );

    {
        CuBuffer<>                                 d_temp;
        CuBuffer<OptixOpacityMicromapDesc>         d_perMicromapDescs;
        CuBuffer<>                                 d_output;       
        
        // combine all buffers that need to be downloaded after the omm bake, 
        // so we only need a single device to host memory transfer.
        CuBuffer<>                                       d_downloadAggregate;

        BufferLayout<>                                   downloadAggregateBuf;
        BufferLayout<OptixOpacityMicromapUsageCount>     usageBuf;
        BufferLayout<OptixOpacityMicromapHistogramEntry> histogramBuf;
        BufferLayout<cuOmmBaking::PostBakeInfo>          infoBuf;
        
        usageBuf.setNumElems( outOmmUsageDescSize ).setAlignmentInBytes( cuOmmBaking::BufferAlignmentInBytes::MICROMAP_USAGE_COUNTS );
        histogramBuf.setNumElems( buffers.numMicromapHistogramEntries ).setAlignmentInBytes( cuOmmBaking::BufferAlignmentInBytes::MICROMAP_HISTOGRAM_ENTRIES );
        infoBuf.setNumElems( 1 ).setAlignmentInBytes( cuOmmBaking::BufferAlignmentInBytes::POST_BAKE_INFO );
        downloadAggregateBuf
            .aggregate( usageBuf )
            .aggregate( histogramBuf )
            .aggregate( infoBuf );

        // materialize to obtain the buffer size
        downloadAggregateBuf.materialize();

        // allocate the post bake buffer
        d_downloadAggregate.alloc( downloadAggregateBuf.getNumBytes() );

        // rematerialize with the actual device pointer
        downloadAggregateBuf.materialize( (unsigned char*)d_downloadAggregate.get() );

        CUDA_CHECK( d_temp.alloc( buffers.tempBufferSizeInBytes ) );
        CUDA_CHECK( d_output.alloc( buffers.outputBufferSizeInBytes ) );
        CUDA_CHECK( d_perMicromapDescs.alloc( buffers.numMicromapDescs ) );

        outOmmIndexBufferSizeInBytes = 0;
        for( uint32_t i = 0; i < numInputs; ++i )
        {
            inputBuffers[i].indexBuffer = d_indices.get() + inputIndexBufferOffsetInBytes[i];
            inputBuffers[i].micromapUsageCountsBuffer = (CUdeviceptr)(usageBuf.access() + inputUsageDescOffset[i]);
        }

        buffers.outputBuffer = d_output.get();
        buffers.perMicromapDescBuffer = d_perMicromapDescs.get();
        buffers.micromapHistogramEntriesBuffer = ( CUdeviceptr )histogramBuf.access();
        if( ( options.flags & cuOmmBaking::BakeFlags::ENABLE_POST_BAKE_INFO ) == cuOmmBaking::BakeFlags::ENABLE_POST_BAKE_INFO )
            buffers.postBakeInfoBuffer = ( CUdeviceptr )infoBuf.access();
        buffers.tempBuffer = d_temp.get();

        OMM_CHECK( cuOmmBaking::BakeOpacityMicromaps( &options, numInputs, inputs, inputBuffers.data(), &buffers, 0 ) );

        // download the aggregate buffer into a properly aligned host buffer. TODO: eventually use aligned_alloc (c++17)
        std::vector<unsigned char> h_postBuildData( downloadAggregateBuf.getNumBytes() + downloadAggregateBuf.getAlignmentInBytes() );
        void* ptr = h_postBuildData.data(); size_t space = h_postBuildData.size();
        ptr = std::align( downloadAggregateBuf.getAlignmentInBytes(), downloadAggregateBuf.getNumBytes(), ptr, space );

        // rematerialize the buffer layout on the host buffer
        downloadAggregateBuf.materialize( ( unsigned char* )ptr );

        CUDA_CHECK( d_downloadAggregate.download( (unsigned char*)downloadAggregateBuf.access() ) );
        for( uint32_t i = 0; i < numInputs; ++i )
        {
            usage[i] = UsageCounts_t( usageBuf.access() + inputUsageDescOffset[i], usageBuf.access() + inputUsageDescOffset[i + 1] );

            for( const auto& entry : usage[i] )
                usageCount += entry.count;
        }

        histogram = HistogramEntries_t( histogramBuf.access(), histogramBuf.access() + histogramBuf.getNumElems() );

        if( ( options.flags & cuOmmBaking::BakeFlags::ENABLE_POST_BAKE_INFO ) == cuOmmBaking::BakeFlags::ENABLE_POST_BAKE_INFO )
        {
            postInfo = *infoBuf.access();
        }
        else
        {
            postInfo = {};
            for( auto& entry : histogram )
                postInfo.numMicromapDescs += entry.count;
        }

        // std::cout << "indices: " << d_indices << std::endl;
        // std::cout << "desc: " << d_perMicromapDescs << std::endl;
        // std::cout << "data: " << d_output << std::endl;

        // build optix array
        if( postInfo.numMicromapDescs )
        {
            OptixMicromapBufferSizes            ommArraySizes = {};
            OptixOpacityMicromapArrayBuildInput ommArrayInput = {};

            ommArrayInput.micromapHistogramEntries     = histogram.data();
            ommArrayInput.numMicromapHistogramEntries  = ( uint32_t )histogram.size();
            ommArrayInput.perMicromapDescStrideInBytes = sizeof( OptixOpacityMicromapDesc );
            OPTIX_CHECK( optixOpacityMicromapArrayComputeMemoryUsage( context, &ommArrayInput, &ommArraySizes ) );

            CUDA_CHECK( d_temp.allocIfRequired( ommArraySizes.tempSizeInBytes ) );
            CUDA_CHECK( d_optixOmmArray.alloc( ommArraySizes.outputSizeInBytes ) );

            OptixMicromapBuffers ommArrayBuffers = {};
            ommArrayBuffers.output = d_optixOmmArray.get();
            ommArrayBuffers.outputSizeInBytes = d_optixOmmArray.byteSize();
            ommArrayBuffers.temp = d_temp.get();
            ommArrayBuffers.tempSizeInBytes = d_temp.byteSize();
            ommArrayInput.perMicromapDescBuffer = d_perMicromapDescs.get();
            ommArrayInput.inputBuffer = d_output.get();
            OPTIX_CHECK( optixOpacityMicromapArrayBuild( context, 0, &ommArrayInput, &ommArrayBuffers ) );
        }
    }

    std::vector<OptixBuildInputOpacityMicromap> optixOmmBuildInputs;
    for( uint32_t i = 0; i < numInputs; ++i )
    {
        OptixBuildInputOpacityMicromap omm = {};
        omm.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE;

        CUdeviceptr indexBuffer = ( CUdeviceptr )( d_indices.get() + inputIndexBufferOffsetInBytes[i] );
        uint32_t indexSizeInBytes = 0;
        switch( indexFormat )
        {
        case cuOmmBaking::IndexFormat::NONE:
            indexSizeInBytes = 0;
            break;
        case cuOmmBaking::IndexFormat::I8_UINT:
            indexSizeInBytes = 1;
            break;
        case cuOmmBaking::IndexFormat::I16_UINT:
            indexSizeInBytes = 2;
            break;
        case cuOmmBaking::IndexFormat::I32_UINT:
            indexSizeInBytes = 4;
            break;
        default:
            indexSizeInBytes = 0;
        }
        CUdeviceptr opacityMicromapArray = d_optixOmmArray.get();

        if( indexBuffer || opacityMicromapArray )
        {
            omm.opacityMicromapArray = opacityMicromapArray;

            if( indexBuffer )
            {
                omm.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
                omm.indexBuffer = indexBuffer;
                omm.indexSizeInBytes = indexSizeInBytes;
                omm.indexStrideInBytes = 0;
            }
            else
            {
                omm.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
            }
        }

        omm.numMicromapUsageCounts = usage[i].size();
        omm.micromapUsageCounts = usage[i].data();

        optixOmmBuildInputs.push_back( omm );
    }
    
    std::swap( optixOmmBuildInputs, m_optixOmmBuildInputs );
    std::swap( usageCount, m_totalUsageCount );
    std::swap( d_indices, m_optixOmmIndices );
    std::swap( d_optixOmmArray, m_optixOmmArray );
    std::swap( histogram, m_histogram );
    std::swap( usage, m_usageCounts );
    std::swap( postInfo, m_info );
        
    return cuOmmBaking::Result::SUCCESS;
}

cuOmmBaking::Result OptixOmmArray::destroy()
{
    m_info       = {};
    m_totalUsageCount = 0;

    m_histogram.clear();
    m_usageCounts.clear();
    m_optixOmmBuildInputs.clear();

    m_optixOmmArray.free();
    m_optixOmmIndices.free();

    return cuOmmBaking::Result::SUCCESS;
}
