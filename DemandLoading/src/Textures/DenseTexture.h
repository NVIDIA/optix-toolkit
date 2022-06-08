//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "Util/Exception.h"

#include <DemandLoading/TextureDescriptor.h>
#include <ImageSource/TextureInfo.h>

#include <vector_types.h>

#include <vector>

namespace demandLoading {

/// DenseTexture encapsulates a standard CUDA texture and its associated CUDA array.
class DenseTexture
{
  public:
    /// Construct DenseTexture for the specified device.
    explicit DenseTexture( unsigned int deviceIndex )
        : m_deviceIndex( deviceIndex )
    {
    }

    /// Destroy the dense texture, reclaiming its resources.
    ~DenseTexture();

    /// Initialize texture from the given descriptor (which specifies clamping/wrapping and
    /// filtering) and the given texture info (which describes the dimensions, format, etc.)
    void init( const TextureDescriptor& descriptor, const imageSource::TextureInfo& info );

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

  private:
    bool                     m_isInitialized = false;
    unsigned int             m_deviceIndex;
    imageSource::TextureInfo m_info;
    CUmipmappedArray         m_array{};
    CUtexObject              m_texture{};

    mutable size_t m_numBytesFilled = 0;
};

}  // namespace demandLoading
