// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <vector_types.h>

#include <memory>
#include <vector>

namespace demandLoading {

/// DenseTexture encapsulates a standard CUDA texture and its associated CUDA array.
class DenseTexture
{
  public:
    /// Destroy the dense texture, reclaiming its resources.
    ~DenseTexture();

    /// Initialize texture from the given descriptor (which specifies clamping/wrapping and
    /// filtering) and the given texture info (which describes the dimensions, format, etc.)
    void init( const TextureDescriptor& descriptor, const imageSource::TextureInfo& info, std::shared_ptr<CUmipmappedArray> masterArray );

    /// Check whether the texture has been initialized.
    bool isInitialized() const { return m_isInitialized; }

    /// Get the dimensions of the specified miplevel.
    uint2 getMipLevelDims( unsigned int mipLevel ) const;

    /// Get the CUDA texture object.
    CUtexObject getTextureObject() const { return m_texture; }

    /// Fill the texture mip levels on the device with textureData, which contains all mip levels.
    void fillTexture( CUstream stream, const char* textureData, unsigned int width, unsigned int height, bool bufferPinned ) const;

    /// Get total number of bytes filled
    size_t getNumBytesFilled() const { return m_numBytesFilled; }

    /// Get the mipmapped array backing store for the texture
    std::shared_ptr<CUmipmappedArray> getDenseArray() { return m_array; }

  private:
    bool                              m_isInitialized = false;
    CUcontext                         m_context;
    imageSource::TextureInfo          m_info;
    std::shared_ptr<CUmipmappedArray> m_array;
    CUtexObject                       m_texture{};

    mutable size_t m_numBytesFilled = 0;
};

}  // namespace demandLoading
