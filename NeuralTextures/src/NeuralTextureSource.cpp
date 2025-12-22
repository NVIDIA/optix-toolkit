// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cuda_runtime.h>
#include <vector>

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/NeuralTextures/NeuralTextureSource.h>

namespace neuralTextures {

NeuralTextureSource::NeuralTextureSource( const std::string& filename )
    : m_filename( filename )
    , m_isOpen( false )
{
}

void NeuralTextureSource::open( imageSource::TextureInfo* info )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    if( !m_isOpen )
    {
        std::string errString = "Could not open NTC image file " + m_filename;
        bool success = m_imageReader.loadFile( m_filename.c_str() );
        OTK_ERROR_CHECK_MSG( !success, errString.c_str() );
        InferenceDataOptix infData = m_imageReader.getInferenceData();

        m_latentsInfo.width        = infData.latentWidth;
        m_latentsInfo.height       = infData.latentHeight;
        m_latentsInfo.format       = CU_AD_FORMAT_UNSIGNED_INT16;
        m_latentsInfo.numChannels  = ( infData.latentFeatures != 12 ) ? infData.latentFeatures / 4 : 4;
        m_latentsInfo.numMipLevels = infData.numLatentMips;
        m_latentsInfo.isValid      = true;
        m_latentsInfo.isTiled      = true;
    }

    m_isOpen = true;
    if( info != nullptr )
    {
        *info = m_latentsInfo;
    }
}

bool NeuralTextureSource::readTile( char* dest, unsigned int latentMipLevel, const imageSource::Tile& tile, CUstream stream )
{
    (void) stream;
    return m_imageReader.readLatentRectUshort( (ushort*)dest, latentMipLevel, tile.x * tile.width, tile.y * tile.height, tile.width, tile.height );
}

bool NeuralTextureSource::readMipLevel( char* dest, unsigned int latentMipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    (void) stream;
    (void) expectedWidth;
    (void) expectedHeight;
    
    int mipWidth = m_latentsInfo.width >> latentMipLevel;
    int mipHeight = m_latentsInfo.height >> latentMipLevel;
    OTK_ASSERT( expectedWidth == mipWidth && expectedHeight == mipHeight );
    return m_imageReader.readLatentRectUshort( (ushort*)dest, latentMipLevel, 0, 0, mipWidth, mipHeight );
}

CUdeviceptr NeuralTextureSource::makeOptixInferenceData( OptixDeviceContext optixContext )
{
    // Allocate device side buffer to hold network weights and inference data
    open( nullptr );
    CUdeviceptr d_infData = 0;
    OTK_ERROR_CHECK( cuMemAlloc( &d_infData, NTC_NETWORK_MAX_SIZE ) );

    // Make inferencing optimal network weights on the device
    CUdeviceptr d_mlpWeights = d_infData + NTC_TEXTURE_SET_CHUNK_SIZE;
    uint32_t bufferSize = NTC_NETWORK_MAX_SIZE - NTC_TEXTURE_SET_CHUNK_SIZE;
    m_imageReader.prepareDeviceNetwork( optixContext, d_mlpWeights, bufferSize );

    // Copy inference metadata to device
    InferenceDataOptix infData = m_imageReader.getInferenceData();
    infData.d_mlpWeights = d_mlpWeights;
    OTK_ERROR_CHECK( cuMemcpyHtoD( d_infData, &infData, sizeof( InferenceDataOptix ) ) );

    return d_infData;
}

}  // namespace neuralTextures

