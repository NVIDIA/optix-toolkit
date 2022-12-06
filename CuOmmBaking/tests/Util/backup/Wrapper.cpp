#include "wrapper.h"

#include <optix_stubs.h>

#define OPTIX_CHECK( x )                                                                                                                                                                                                                                                               \
    {                                                                                                                                                                                                                                                                                  \
        OptixResult err = ( OptixResult )( x );                                                                                                                                                                                                                                        \
        if( err != OPTIX_SUCCESS )                                                                                                                                                                                                                                                     \
            return ommBaking::Result::ERROR_CUDA;                                                                                                                                                                                                                                                 \
    };

#define CUDA_CHECK( x )                                                                                                                                                                                                                                                                \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t err = (cudaError_t)( x );                                                                                                                                                                                                                                          \
        if( err != cudaSuccess )                                                                                                                                                                                                                                                       \
            return ommBaking::Result::ERROR_CUDA;                                                                                                                                                                                                                                                 \
    };

#define OMM_CHECK( x )                                                                                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                                                                  \
        ommBaking::Result err = ( x );                                                                                                                                                                                                                                                            \
        if( err != ommBaking::Result::SUCCESS )                                                                                                                                                                                                                                                   \
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

ommBaking::Result OptixOmmArray::create( OptixDeviceContext context, const ommBaking::Options& options, const ommBaking::BakeInputDesc* inputs, unsigned int numInputs )
{
    OMM_CHECK( destroy() );

    if( !inputs )
        return ommBaking::Result::ERROR_INVALID_VALUE;

    if( numInputs == 0 )
        return ommBaking::Result::ERROR_INVALID_VALUE;

    std::vector<ommBaking::BakeInputPreBuildInfo> inputInfo;
    inputInfo.resize( numInputs );

    ommBaking::BakePreBuildInfo info = {};
    OMM_CHECK( ommBaking::GetPreBakeInfo( &options, numInputs, inputs, inputInfo.data(), &info));
    
    CuBuffer<>                          d_optixOmmArray;
    CuBuffer<>                          d_optixOmmIndices;
    CuBuffer<>                          d_tmp;
    
    ommBaking::IndexFormat indexFormat = info.indexFormat;

    std::vector<size_t>   inputIndexBufferOffsetInBytes;
    std::vector<size_t>   inputUsageDescOffset;
    size_t outOmmIndexBufferSizeInBytes = 0;
    size_t outOmmUsageDescSize = 0;
    for( unsigned i = 0; i < numInputs; ++i )
    {
        inputIndexBufferOffsetInBytes.push_back( outOmmIndexBufferSizeInBytes );
        outOmmIndexBufferSizeInBytes += inputInfo[i].outOmmIndexBufferSizeInBytes;

        inputUsageDescOffset.push_back( outOmmUsageDescSize );
        outOmmUsageDescSize += inputInfo[i].outOmmUsageCount;
    }
    inputUsageDescOffset.push_back( outOmmUsageDescSize );

    CUDA_CHECK( d_optixOmmIndices.alloc( outOmmIndexBufferSizeInBytes ) );
    CUDA_CHECK( d_tmp.alloc( info.tmpSizeInBytes ) );
    
    ArrayHistogram_t              histogram;
    ommBaking::BakePostBuildInfo  postInfo = {};

    uint32_t usageCount = 0;
    std::vector<IndexHistogram_t>               usage( numInputs );

    {
        CuBuffer<OptixOpacityMicromapDesc>           d_ommDescs;
        CuBuffer<>                                   d_ommData;
        CuBuffer<OptixOpacityMicromapUsageCount>     d_usage;
        CuBuffer<OptixOpacityMicromapHistogramEntry> d_histogram;
        CuBuffer<ommBaking::BakePostBuildInfo>       d_info;
        
        CUDA_CHECK( d_info.alloc( 1 ) );
        CUDA_CHECK( d_ommData.alloc( info.outOmmArrayDataSizeInBytes ) );
        CUDA_CHECK( d_histogram.alloc( info.outOmmHistogramCount ) );
        CUDA_CHECK( d_usage.alloc( outOmmUsageDescSize ) );

        CUDA_CHECK( d_ommDescs.alloc( info.outOmmDescCount ) );

        std::vector<ommBaking::BakeInputBuffers> inputBuffers;
        inputBuffers.resize( numInputs );

        outOmmIndexBufferSizeInBytes = 0;
        for( uint32_t i = 0; i < numInputs; ++i )
        {
            inputBuffers[i].outOmmIndexSizeInBytes = inputInfo[i].outOmmIndexBufferSizeInBytes;
            inputBuffers[i].outOmmIndex            = d_optixOmmIndices.get() + inputIndexBufferOffsetInBytes[i];

            inputBuffers[i].outOmmUsageCount = inputInfo[i].outOmmUsageCount;
            inputBuffers[i].outOmmUsage      = d_usage.get( inputUsageDescOffset[i] );
        }

        ommBaking::BakeBuffers buffers = {};
        buffers.outOmmArrayData = d_ommData.get();
        buffers.outOmmArrayDataSizeInBytes = d_ommData.byteSize();

        buffers.outOmmDesc = d_ommDescs.get();
        buffers.outOmmDescCount = d_ommDescs.count();

        buffers.outOmmHistogram = d_histogram.get();
        buffers.outOmmHistogramCount = d_histogram.count();

        buffers.outPostBuildInfo = d_info.get();
        buffers.outPostBuildInfoSizeInBytes = d_info.byteSize();

        buffers.tmp = d_tmp.get();
        buffers.tmpSizeInBytes = d_tmp.byteSize();

        OMM_CHECK( ommBaking::BakeOpacityMicromaps( &options, numInputs, inputs, inputBuffers.data(), &buffers, 0 ) );

        CUDA_CHECK( d_histogram.download( histogram ) );

        std::vector<OptixOpacityMicromapUsageCount> h_usage;
        CUDA_CHECK( d_usage.download( h_usage ) );
        for( uint32_t i = 0; i < numInputs; ++i )
        {
            usage[i] = IndexHistogram_t( h_usage.begin() + inputUsageDescOffset[i], h_usage.begin() + inputUsageDescOffset[i + 1] );

            for( const auto& entry : usage[i] )
                usageCount += entry.count;
        }

        CUDA_CHECK( d_info.download( &postInfo ) );

        // std::cout << "indices: " << ommIndex << std::endl;
        // std::cout << "desc: " << d_ommDescs << std::endl;
        // std::cout << "data: " << d_ommData << std::endl;

        // build optix array
        if( postInfo.numOmmsInArray )
        {
            OptixMicromapBufferSizes            ommArraySizes = {};
            OptixOpacityMicromapArrayBuildInput ommArrayInput = {};

            ommArrayInput.micromapHistogramEntries     = histogram.data();
            ommArrayInput.numMicromapHistogramEntries  = ( uint32_t )histogram.size();
            ommArrayInput.perMicromapDescStrideInBytes = sizeof( OptixOpacityMicromapDesc );
            OPTIX_CHECK( optixOpacityMicromapArrayComputeMemoryUsage( context, &ommArrayInput, &ommArraySizes ) );

            d_tmp.allocIfRequired( ommArraySizes.tempSizeInBytes );

            CUDA_CHECK( d_optixOmmArray.alloc( ommArraySizes.outputSizeInBytes ) );

            OptixMicromapBuffers ommArrayBuffers = {};
            ommArrayBuffers.output = d_optixOmmArray.get();
            ommArrayBuffers.outputSizeInBytes = d_optixOmmArray.byteSize();
            ommArrayBuffers.temp = d_tmp.get();
            ommArrayBuffers.tempSizeInBytes = d_tmp.byteSize();
            ommArrayInput.perMicromapDescBuffer = d_ommDescs.get();
            ommArrayInput.inputBuffer = d_ommData.get();
            OPTIX_CHECK( optixOpacityMicromapArrayBuild( context, 0, &ommArrayInput, &ommArrayBuffers ) );
        }
    }

    std::vector<OptixBuildInputOpacityMicromap> optixOmmBuildInputs;
    for( uint32_t i = 0; i < numInputs; ++i )
    {
        OptixBuildInputOpacityMicromap omm = {};
        omm.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE;

        CUdeviceptr indexBuffer = ( CUdeviceptr )( d_optixOmmIndices.get() + inputIndexBufferOffsetInBytes[i] );
        uint32_t indexSizeInBytes = 0;
        switch( indexFormat )
        {
        case ommBaking::IndexFormat::NONE:
            indexSizeInBytes = 0;
            break;
        case ommBaking::IndexFormat::I8_UINT:
            indexSizeInBytes = 1;
            break;
        case ommBaking::IndexFormat::I16_UINT:
            indexSizeInBytes = 2;
            break;
        case ommBaking::IndexFormat::I32_UINT:
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
    std::swap( usageCount, m_usageCount );
    std::swap( d_optixOmmIndices, m_optixOmmIndices );
    std::swap( d_optixOmmArray, m_optixOmmArray );
    std::swap( histogram, m_histogram );
    std::swap( usage, m_usage );
    std::swap( postInfo, m_info );
        
    return ommBaking::Result::SUCCESS;
}

ommBaking::Result OptixOmmArray::destroy()
{
    m_info       = {};
    m_usageCount = 0;

    m_histogram.clear();
    m_usage.clear();
    m_optixOmmBuildInputs.clear();

    m_optixOmmArray.free();
    m_optixOmmIndices.free();

    return ommBaking::Result::SUCCESS;
}
