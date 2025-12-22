/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <rapidjson/document.h>

#include "InferenceDataOptix.h"

const uint32_t NTC_NETWORK_MAX_SIZE = 32768;

class NtcImageReader
{
  public:
    /// Load an .ntc file
    bool loadFile( const char* fileName );

    /// Make the latent texture for the current cuda context
    CUtexObject makeLatentTexture();

    /// Make the device network data for optixContext in the d_dstMatrix buffer
    bool prepareDeviceNetwork( OptixDeviceContext optixContext, CUdeviceptr d_dstMatrix, int d_dstSize );

    /// Get the texture inference data for this texture
    const InferenceDataOptix& getInferenceData() { return m_inferenceData; }

    /// Read a rectangle from a mip level of the latent texture into dest on the host.
    bool readLatentRectUshort( ushort* dest, int mipLevel, int xstart, int ystart, int width, int height );

  private:

    struct NtcFileHeader
    {
        uint32_t magicNumber;
        uint32_t version;
        uint64_t jsonOffset;
        uint64_t jsonSize;
        uint64_t dataOffset;
        uint64_t dataSize;
    };
    
    struct NtcNetworkLayer
    {
        int inputChannels = -1;
        int outputChannels = -1;
        int weightOffset = -1;
        int weightSize = 0;
        int scaleOffset = -1;
        int scaleSize = 0;
        int biasOffset = -1;
        int biasSize = 0;
        std::string weightType;
        std::string scaleType;
        std::string biasType;
    };

    InferenceDataOptix m_inferenceData{};
    std::vector<char> m_hDataChunk;
    std::vector<int> m_hLatentMipOffsets;
    std::vector<int> m_hLatentMipSizes;
    std::vector<NtcNetworkLayer> m_hNetwork;
    std::vector<char> m_hNetworkData;

    bool parseTextureSetDescription( rapidjson::Document& doc );
    bool parseLatentsDescription( rapidjson::Document& doc );
    bool parseNetworkDescription( rapidjson::Document& doc );
    uint32_t getColorSpace( rapidjson::Document& doc, int channelNum );
    
    bool convertNetworkToOptixInferencingOptimal( OptixDeviceContext optixContext, CUdeviceptr d_srcNetworkData,
                                                  CUdeviceptr d_dstMatrix, int d_dstSize );

    void printTextureSetDescription();
};