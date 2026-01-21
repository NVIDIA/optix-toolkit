/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <math.h>
#include <fstream>

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <OptiXToolkit/NeuralTextures/NtcImageReader.h>

#define OPTIX_CHK( call ) if( call != OPTIX_SUCCESS ) return false
#define CUDA_CHK( call ) if( call != CUDA_SUCCESS ) return false

bool NtcImageReader::loadFile( const char* fileName )
{
    const uint32_t NTEX_MAGIC_NUMBER = 0x5845544E; // "NTEX"
    const uint32_t NTEX_SUPPORTED_VERSION = 0x100;

    std::ifstream file( fileName, std::ios::binary );
    if( !file.is_open() )
        return false;

    // Read file header
    NtcFileHeader header{};
    file.read( reinterpret_cast<char*>( &header ), sizeof( NtcFileHeader ) );
    if( header.magicNumber != NTEX_MAGIC_NUMBER || header.version != NTEX_SUPPORTED_VERSION )
        return false;

    // Read json text
    std::vector<char> jsonText( header.jsonSize + 1, '\0' );
    file.seekg( header.jsonOffset );
    file.read( jsonText.data(), header.jsonSize );

    // Parse json document and fill in texture description
    rapidjson::Document jsonDoc;
    jsonDoc.Parse( jsonText.data() );
    if ( jsonDoc.HasParseError() )
        return false;
    if( !parseTextureSetDescription( jsonDoc ) )
        return false;

    // Figure out how much of the data chunk to read
    bool loadLatents = true;
    int dataChunkReadSize = static_cast<int>(header.dataSize);
    if( !loadLatents )
    {
        int latentOffsetViewIdx = jsonDoc["latents"][0]["view"].GetInt();
        int latentOffset = jsonDoc["views"][latentOffsetViewIdx]["offset"].GetInt();
        dataChunkReadSize = latentOffset;
    }
    
    // Read data chunk, parse latents and network weights
    m_hDataChunk.resize( dataChunkReadSize, '\0' );

    file.seekg( header.dataOffset );
    file.read( m_hDataChunk.data(), dataChunkReadSize );
    if( !parseLatentsDescription( jsonDoc ) )
        return false;
    if( !parseNetworkDescription( jsonDoc ) )
        return false;

    //printTextureSetDescription();
    return true;
}


bool NtcImageReader::parseTextureSetDescription( rapidjson::Document& doc )
{
    try
    {
        NtcTextureSetConstants& tsc = m_inferenceData.constants;

        int schemaVersion = doc["schemaVersion"].GetInt();
        if( schemaVersion < 3 )
            return false;

        // General characteristics
        tsc.imageMips = doc["numColorMips"].GetInt();
        tsc.imageWidth = doc["colorMips"][0]["width"].GetInt();
        tsc.imageHeight = doc["colorMips"][0]["height"].GetInt();

        tsc.validChannelMask = 0;

        m_inferenceData.numTextures = static_cast<int>(doc["textures"].Size());
        m_inferenceData.numChannels = doc["numChannels"].GetInt();

        // Subtexture descriptions
        for( int texNum = 0; texNum < m_inferenceData.numTextures; ++texNum )
        {
            m_inferenceData.texFirstChannel[texNum] = doc["textures"][texNum]["firstChannel"].GetInt();
            m_inferenceData.texNumChannels[texNum] = doc["textures"][texNum]["numChannels"].GetInt();
        }

        // Channel color spaces
        tsc.channelColorSpaces = 0;
        for( int channelNum = 0; channelNum < m_inferenceData.numChannels; ++channelNum )
        {
            uint32_t colorSpace = getColorSpace( doc, channelNum );
            tsc.channelColorSpaces = tsc.channelColorSpaces | ( colorSpace << ( channelNum * 2 ) );
        }

        // Latents info
        m_inferenceData.latentFeatures = doc["latentShape"]["numFeatures"].GetInt();
        m_inferenceData.latentWidth = doc["latents"][0]["width"].GetInt();
        m_inferenceData.latentHeight = doc["latents"][0]["height"].GetInt();

        // Color mips
        for( int colorMipLevel = 0; colorMipLevel < tsc.imageMips; ++colorMipLevel )
        {
            NtcColorMipConstants &colorMip = tsc.colorMips[colorMipLevel];
            colorMip.neuralMip = doc["colorMips"][colorMipLevel]["latentMip"].GetInt();
            colorMip.positionLod = doc["colorMips"][colorMipLevel]["positionLod"].GetFloat();
            colorMip.positionScale = doc["colorMips"][colorMipLevel]["positionScale"].GetFloat();
        }
    }
    catch(...)
    {
        return false;
    }

    return true;
}


uint32_t NtcImageReader::getColorSpace( rapidjson::Document& doc, int channelNum )
{
    rapidjson::Value& channel = doc["channels"][channelNum];
    if( !channel.HasMember("colorSpace") )
        return CS_LINEAR;
    else if( channel["colorSpace"].GetString() == std::string("sRGB") )
        return CS_SRGB;
    else if( channel["colorSpace"].GetString() == std::string("HLG") )
        return CS_HLG;
    return CS_LINEAR;
}


bool NtcImageReader::parseLatentsDescription( rapidjson::Document& doc )
{
    try 
    {
        // The latent data is held in m_hDataChunk. Read the offsets and sizes
        m_hLatentMipOffsets.resize( doc["latents"].Size(), 0 );
        m_hLatentMipSizes.resize( doc["latents"].Size(), 0 );
        m_inferenceData.numLatentMips = static_cast<int>( doc["latents"].Size() );
        for( unsigned int level = 0; level < doc["latents"].Size(); ++level )
        {
            // FIXME: Assuming that layer views are sequential, and all layer views are the same size.
            int dataViewIdx = doc["latents"][level]["layerViews"][0].GetInt();
            int numLayerViews = doc["latents"][level]["layerViews"].Size();
            m_hLatentMipOffsets[level] = doc["views"][dataViewIdx]["offset"].GetInt();
            m_hLatentMipSizes[level] = doc["views"][dataViewIdx]["storedSize"].GetInt() * numLayerViews;
        }
    }
    catch(...)
    {
        return false;
    }

    return true;
}


bool NtcImageReader::parseNetworkDescription( rapidjson::Document& doc )
{
    try
    {
        // Find a proper network (3 or 4 layers, FloatE4M3 weights...)
        unsigned int networkIdx = 0;
        for( networkIdx = 0; networkIdx < doc["mlpVersions"].Size(); ++networkIdx )
        {
            rapidjson::Value& network = doc["mlpVersions"][networkIdx];
            if( ( network["layers"].Size() == 4u || network["layers"].Size() == 3u ) &&
                network["layers"][0]["weightType"].GetString() == std::string("FloatE4M3") )
            {
                break;
            }
        }
        if( networkIdx >= doc["mlpVersions"].Size() )
            return false;
        rapidjson::Value& network = doc["mlpVersions"][networkIdx];

        // Read the network layer descriptions, and copy data to m_hNetworkData
        // so it is in one contiguous array.

        const int maxNetworkSizeInBytes = 16384;
        m_hNetworkData.resize( maxNetworkSizeInBytes );
        m_hNetwork.resize( network["layers"].Size() );
        int offset = 0;

        // Read the network matrix weights first so network layers are all together
        for( unsigned int layerId = 0; layerId < network["layers"].Size(); ++layerId )
        {
            rapidjson::Value& networkLayer = network["layers"][layerId];
            NtcNetworkLayer& layer = m_hNetwork[layerId];

            layer.inputChannels = networkLayer["inputChannels"].GetInt();
            layer.outputChannels = networkLayer["outputChannels"].GetInt();
            
            if( networkLayer.HasMember( "weightView" ) )
            {
                int viewIdx = networkLayer["weightView"].GetInt();
                int srcOffset = doc["views"][viewIdx]["offset"].GetInt();
                layer.weightSize = doc["views"][viewIdx]["storedSize"].GetInt();
                layer.weightOffset = offset;
                memcpy( &m_hNetworkData[offset], &m_hDataChunk[srcOffset], layer.weightSize );
                offset += layer.weightSize;
            }

            if( networkLayer.HasMember( "weightType" ) )
                layer.weightType = networkLayer["weightType"].GetString();
            if( networkLayer.HasMember( "scaleType" ) )
                layer.scaleType = networkLayer["scaleType"].GetString();
            if( networkLayer.HasMember( "biasType" ) )
                layer.biasType = networkLayer["biasType"].GetString();
        }

        // Read the scales and biases
        for( unsigned int layerId = 0; layerId < network["layers"].Size(); ++layerId )
        {
            rapidjson::Value& networkLayer = network["layers"][layerId];
            NtcNetworkLayer& layer = m_hNetwork[layerId];

            if( networkLayer.HasMember( "scaleView" ) )
            {
                int viewIdx = networkLayer["scaleView"].GetInt();
                int srcOffset = doc["views"][viewIdx]["offset"].GetInt();
                layer.scaleSize = doc["views"][viewIdx]["storedSize"].GetInt();
                layer.scaleOffset = offset;
                memcpy( &m_hNetworkData[offset], &m_hDataChunk[srcOffset], layer.scaleSize );
                offset += layer.scaleSize;
            }
            if( networkLayer.HasMember( "biasView" ) )
            {
                int viewIdx = networkLayer["biasView"].GetInt();
                int srcOffset = doc["views"][viewIdx]["offset"].GetInt();
                layer.biasSize = doc["views"][viewIdx]["storedSize"].GetInt();
                layer.biasOffset = offset;
                memcpy( &m_hNetworkData[offset], &m_hDataChunk[srcOffset], layer.biasSize );
                offset += layer.biasSize;
            }
        }

        m_hNetworkData.resize( offset );
    }
    catch(...)
    {
        return false;
    }

    return true;
}


bool NtcImageReader::readLatentRectUshort( uint16_t* dest, int mipLevel, int xstart, int ystart, int width, int height )
{
    int numLatentTextures = m_inferenceData.latentFeatures / 4;
    int destPixelStride = (numLatentTextures != 3) ? numLatentTextures : 4;

    int mipWidth = m_inferenceData.latentWidth >> mipLevel;
    int mipHeight = m_inferenceData.latentHeight >> mipLevel;
    int latentOffset = m_hLatentMipOffsets[mipLevel];

    uint16_t* src = (uint16_t*) &m_hDataChunk[latentOffset];
    width = std::min( width, mipWidth - xstart );
    height = std::min( height, mipHeight - ystart );

    for( int y = 0; y < height; ++y )
    {
        for( int x = 0; x < width; ++x )
        {
            uint16_t* pixelDest = &dest[( y * width + x ) * destPixelStride];
            for( int c = 0; c < numLatentTextures; ++c )
            {
                int srcLayerOffset = mipWidth * mipHeight * c;
                int srcPixelOffset = ( y + ystart ) * mipWidth + ( x + xstart );
                uint16_t* pixelSrc = &src[srcLayerOffset + srcPixelOffset];
                pixelDest[c] = *pixelSrc;
            }
        }
    }
    return true;
}


CUtexObject NtcImageReader::makeLatentTexture()
{
    // Allocate mipmapped CUDA array
    int numLatentTextures = m_inferenceData.latentFeatures / 4;
    int pixelStride = (numLatentTextures != 3) ? numLatentTextures : 4;
    int numMips = static_cast<int>(m_hLatentMipOffsets.size());
    
    // Create mipmapped array descriptor
    CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
    arrayDesc.Width = m_inferenceData.latentWidth;
    arrayDesc.Height = m_inferenceData.latentHeight;
    arrayDesc.Depth = 0;  // 2D texture
    arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
    arrayDesc.NumChannels = pixelStride;
    arrayDesc.Flags = 0;
    
    CUmipmappedArray mipmappedArray;
    CUDA_CHK( cuMipmappedArrayCreate( &mipmappedArray, &arrayDesc, numMips ) );
    
    // Fill each mip level
    for ( int mipLevel = 0; mipLevel < numMips; mipLevel++ )
    {
        // No data for this mip level
        if( mipLevel > 0 && m_hLatentMipOffsets[mipLevel] == 0 )
            continue;

        // Get the source (latentSrc) and destination (levelArray)
        int mipWidth = m_inferenceData.latentWidth >> mipLevel;
        int mipHeight = m_inferenceData.latentHeight >> mipLevel;
        std::vector<uint16_t> latentSrc( mipWidth * mipHeight * pixelStride, 0 );
        readLatentRectUshort( latentSrc.data(), mipLevel, 0, 0, mipWidth, mipHeight );
        CUarray levelArray;
        CUDA_CHK( cuMipmappedArrayGetLevel( &levelArray, mipmappedArray, mipLevel ) );
        
        // Copy data to this mip level
        CUDA_MEMCPY2D copyDesc = {};
        copyDesc.srcMemoryType = CU_MEMORYTYPE_HOST;
        copyDesc.srcHost = latentSrc.data();
        copyDesc.srcPitch = mipWidth * pixelStride * sizeof(uint16_t);
        copyDesc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyDesc.dstArray = levelArray;
        copyDesc.WidthInBytes = mipWidth * pixelStride * sizeof(uint16_t);
        copyDesc.Height = mipHeight;
        CUDA_CHK( cuMemcpy2D( &copyDesc ) );
    }
    
    // Create texture object
    CUDA_RESOURCE_DESC resDesc = {};
    resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resDesc.res.mipmap.hMipmappedArray = mipmappedArray;
    
    CUDA_TEXTURE_DESC texDesc = {};
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
    texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_READ_AS_INTEGER;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = (float)( numMips - 1 );
    texDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    texDesc.maxAnisotropy = 16;
    
    CUtexObject texObject;
    CUDA_CHK( cuTexObjectCreate( &texObject, &resDesc, &texDesc, nullptr ) );
    return texObject;
}


bool NtcImageReader::prepareDeviceNetwork( OptixDeviceContext optixContext, CUdeviceptr d_dest, int d_destSize )
{
    if( m_hNetwork.size() != NTC_MLP_LAYERS )
        return false;

    // Make temporary device buffer and copy row major network data to it
    CUdeviceptr d_networkData = 0;
    CUDA_CHK( cuMemAlloc( &d_networkData, m_hNetworkData.size() ) );
    CUDA_CHK( cuMemcpyHtoD( d_networkData, m_hNetworkData.data(), m_hNetworkData.size() ) ); 

    // Convert to inferencing optimal in the dest array, and delete the temporary device buffer.
    bool rval = convertNetworkToOptixInferencingOptimal( optixContext, d_networkData, d_dest, d_destSize );
    CUDA_CHK( cuMemFree( d_networkData ) );
    return rval;
}


bool NtcImageReader::convertNetworkToOptixInferencingOptimal( OptixDeviceContext optixContext, CUdeviceptr d_srcNetworkData, CUdeviceptr d_dstMatrix, int d_dstSize )
{
    const int numLayers = NTC_MLP_LAYERS;
    int networkSizeInBytes = (int)m_hNetworkData.size();
    
    std::vector<OptixCoopVecMatrixDescription> srcLayerDesc( numLayers, OptixCoopVecMatrixDescription{} );
    std::vector<OptixCoopVecMatrixDescription> dstLayerDesc( numLayers, OptixCoopVecMatrixDescription{} );

    OptixCoopVecMatrixLayout srcMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
    OptixCoopVecMatrixLayout dstMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
    
    NtcTextureSetConstants& tsc = m_inferenceData.constants;
    const int optStride = 0;

    // Compute layer sizes
    for( int i = 0; i < numLayers; ++i )
    {
        const unsigned int N = m_hNetwork[i].outputChannels;
        const unsigned int K = m_hNetwork[i].inputChannels;

        size_t srcMatrixDataSize = 0;
        size_t dstMatrixDataSize = 0;

        OptixCoopVecElemType layerType = (i < numLayers - 1) ? OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3 : OPTIX_COOP_VEC_ELEM_TYPE_INT8;

        OPTIX_CHK( optixCoopVecMatrixComputeSize(
            optixContext,
            N,
            K,
            layerType,
            srcMatrixLayout,
            optStride,
            &srcMatrixDataSize
            ) );

        OPTIX_CHK( optixCoopVecMatrixComputeSize(
            optixContext,
            N,
            K,
            layerType,
            dstMatrixLayout,
            optStride,
            &dstMatrixDataSize
            ) );
        
        OptixCoopVecMatrixDescription& srcLayer = srcLayerDesc[i];
        OptixCoopVecMatrixDescription& dstLayer = dstLayerDesc[i];
        srcLayer.N = dstLayer.N = N;
        srcLayer.K = dstLayer.K = K;
        srcLayer.offsetInBytes  = m_hNetwork[i].weightOffset;
        dstLayer.offsetInBytes  = i == 0 ? 0 : dstLayerDesc[i - 1].offsetInBytes + dstLayerDesc[i - 1].sizeInBytes;
        srcLayer.elementType    = layerType;
        dstLayer.elementType    = layerType;
        srcLayer.layout         = srcMatrixLayout;
        dstLayer.layout         = dstMatrixLayout;
        srcLayer.rowColumnStrideInBytes = optStride;
        dstLayer.rowColumnStrideInBytes = optStride;
        srcLayer.sizeInBytes    = static_cast<unsigned int>( srcMatrixDataSize );
        dstLayer.sizeInBytes    = static_cast<unsigned int>( dstMatrixDataSize );

        // Put network data offsets in the texture set constants
        tsc.networkWeightOffsets[i] = dstLayer.offsetInBytes;
    }

    OptixNetworkDescription inputNetworkDescription = { srcLayerDesc.data(), static_cast<unsigned int>( srcLayerDesc.size() ) };
    OptixNetworkDescription outputNetworkDescription = { dstLayerDesc.data(), static_cast<unsigned int>( dstLayerDesc.size() ) };

    size_t dst_mats_size = dstLayerDesc.back().offsetInBytes + dstLayerDesc.back().sizeInBytes;  // trick to sum all dstLayer sizes
    size_t src_mats_size = srcLayerDesc.back().offsetInBytes + srcLayerDesc.back().sizeInBytes;  // trick to sum all srcLayer sizes
    size_t src_other_stuff_size = networkSizeInBytes - src_mats_size;
    size_t dst_total_size       = dst_mats_size + src_other_stuff_size;

    if( d_dstSize < static_cast<int>(dst_total_size) )
        return false;

    const int numNetworks = 1;
    OPTIX_CHK( optixCoopVecMatrixConvert(
        optixContext,
        CUstream{0},
        numNetworks,
        &inputNetworkDescription,
        d_srcNetworkData,
        optStride,
        &outputNetworkDescription,
        d_dstMatrix,
        optStride) );

    // Put scale and bias offsets in texture set constants
    tsc.networkScaleOffsets[0] = static_cast<int>(dst_mats_size);
    tsc.networkBiasOffsets[0] = static_cast<int>(tsc.networkScaleOffsets[0] + m_hNetwork[0].scaleSize);
    for( int i = 1; i < numLayers; ++i )
    {
        tsc.networkScaleOffsets[i] = static_cast<int>(tsc.networkBiasOffsets[i-1] + m_hNetwork[i-1].biasSize);
        tsc.networkBiasOffsets[i] = static_cast<int>(tsc.networkScaleOffsets[i] + m_hNetwork[i].scaleSize);
    }

    // copy the other stuff after the mats arrays from src to dest
    CUDA_CHK( cuMemcpyDtoD(
        d_dstMatrix + dst_mats_size,
        d_srcNetworkData + src_mats_size,
        src_other_stuff_size ) );

    return true;
}


void NtcImageReader::printTextureSetDescription()
{
    NtcTextureSetConstants& tsc = m_inferenceData.constants;
    printf( "width:%d, height:%d, mips:%d\n", tsc.imageWidth, tsc.imageHeight, tsc.imageMips );
    
    for( unsigned int i = 0; i < m_hNetwork.size(); ++i )
    {
        NtcNetworkLayer& layer = m_hNetwork[i];
        printf( "Layer %d: inputs:%d, outputs:%d, weights:%s,%d,%d, scale:%s,%d,%d, bias:%s,%d,%d\n",
            i, layer.inputChannels, layer.outputChannels,
            layer.weightType.c_str(), layer.weightOffset, layer.weightSize,
            layer.scaleType.c_str(), layer.scaleOffset, layer.scaleSize,
            layer.biasType.c_str(), layer.biasOffset, layer.biasSize );
    }

    for( int i=0; i < tsc.imageMips; ++i )
    {
        NtcColorMipConstants mip = tsc.colorMips[i];
        printf( "mip[%d]: neuralMip:%d, positionLod:%1.3f, positionScale:%1.3f\n", 
            i, mip.neuralMip, mip.positionLod, mip.positionScale );
    }

    printf("numTextures:%d, numChannels:%d\n", m_inferenceData.numTextures, m_inferenceData.numChannels );
    printf( "validChannelMask:%x, channelColorSpaces:%x\n", tsc.validChannelMask, tsc.channelColorSpaces );

    for( int i=0; i < m_inferenceData.numTextures; ++i )
    {
        printf("TextureChannels[%d] (%d, %d)\n", i, m_inferenceData.texFirstChannel[i], m_inferenceData.texNumChannels[i]);
    }

    printf("latentFeatures:%d, latentWidth:%d, latentHeight:%d\n", 
        m_inferenceData.latentFeatures, m_inferenceData.latentWidth, m_inferenceData.latentHeight );
    
    for( unsigned int i = 0; i < m_hLatentMipOffsets.size(); ++i )
    {
        printf( "latentMip[%d]: offset:%d\n", i, m_hLatentMipOffsets[i] );
    }
}
