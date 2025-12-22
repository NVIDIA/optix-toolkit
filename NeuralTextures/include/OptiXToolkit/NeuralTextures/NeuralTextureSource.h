// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <mutex>

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include "InferenceDataOptix.h"
#include "NtcImageReader.h"

namespace neuralTextures {

const uint32_t NTC_TEXTURE_SET_CHUNK_SIZE = 512;

/// NeuralTextureSource generates textures using neural network inference
class NeuralTextureSource : public imageSource::ImageSourceBase
{
  public:
    /// Create a neural texture source with the specified dimensions
    explicit NeuralTextureSource( const std::string& filename );

    /// The destructor is virtual.
    ~NeuralTextureSource() override = default;

    /// The open method initializes the given image info struct.
    void open( imageSource::TextureInfo* info ) override;

    /// The close operation.
    void close() override { m_isOpen = false; }

    /// Check if image is currently open.
    bool isOpen() const override { return m_isOpen; }

    /// Get the image info.  Valid only after calling open().
    const imageSource::TextureInfo& getInfo() const override { return m_latentsInfo; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    bool readTile( char* dest, unsigned int latentMipLevel, const imageSource::Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel.  Returns true for success.
    bool readMipLevel( char* dest, unsigned int latentMipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& /*dest*/ ) override { return false; }

    /// Get the extra data for the sampler
    CUdeviceptr getSamplerExtraData( OptixDeviceContext optixContext ) override { return makeOptixInferenceData( optixContext ); }

    /// Make the neural texture inference data for this optix context
    CUdeviceptr makeOptixInferenceData( OptixDeviceContext optixContext );

    /// Get the inference data
    const InferenceDataOptix& getInferenceData() { return m_imageReader.getInferenceData(); }

  private:

    std::string m_filename;
    NtcImageReader m_imageReader;
    imageSource::TextureInfo m_latentsInfo;
    bool m_isOpen;
    std::mutex m_mutex;
};

}  // namespace neuralTextures

