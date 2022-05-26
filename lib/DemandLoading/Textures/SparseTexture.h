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

class SparseArray
{
public:
    SparseArray() = default;
    ~SparseArray();

    void init( unsigned int deviceIndex, const imageSource::TextureInfo& info );

    explicit operator CUmipmappedArray() const { return m_array; }

    CUarray getLevel( unsigned int mipLevel ) const
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

        CUarray mipLevelArray{};
        DEMAND_CUDA_CHECK( cuMipmappedArrayGetLevel( &mipLevelArray, m_array, mipLevel ) );
        return mipLevelArray;
    }

    unsigned int getTileWidth() const { return m_properties.tileExtent.width; }
    unsigned int getTileHeight() const { return m_properties.tileExtent.height; }
    unsigned int getMipTailFirstLevel() const { return m_properties.miptailFirstLevel; }
    size_t       getMipTailSize() const { return m_properties.miptailSize; }
    uint2 getMipLevelDims( unsigned int mipLevel ) const
    {
        DEMAND_ASSERT( mipLevel < m_mipLevelDims.size() );
        return m_mipLevelDims[mipLevel];
    }

    void mapTileAsync( CUstream                     stream,
                       unsigned int                 mipLevel,
                       uint2                        levelOffset,
                       uint2                        levelExtent,
                       CUmemGenericAllocationHandle memHandle,
                       size_t                       offset ) const;
    void unmapTileAsync( CUstream stream, unsigned int mipLevel, uint2 levelOffset, uint2 levelExtent ) const;
    void mapMipTailAsync( CUstream stream, size_t mipTailSize, CUmemGenericAllocationHandle memHandle, size_t offset ) const;
    void unmapMipTailAsync( CUstream stream, size_t mipTailSize ) const;

private:
    // Get the dimensions of the specified miplevel by querying its CUDA array descriptor.
    uint2 queryMipLevelDims( unsigned int mipLevel ) const;

    bool                         m_initialized{};
    unsigned int                 m_deviceIndex{};
    imageSource::TextureInfo     m_info{};
    CUmipmappedArray             m_array{};
    CUDA_ARRAY_SPARSE_PROPERTIES m_properties{};
    std::vector<uint2>           m_mipLevelDims;
};

/// SparseTexture encapsulates a CUDA sparse texture and its associated CUDA array.
class SparseTexture
{
  public:
    /// Construct SparseTexture for the specified device.
    explicit SparseTexture( unsigned int deviceIndex )
        : m_deviceIndex( deviceIndex )
    {
    }

    /// Destroy the sparse texture, reclaiming its resources.
    ~SparseTexture();

    /// Initialize sparse texture from the given descriptor (which specifies clamping/wrapping and
    /// filtering) and the given texture info (which describes the dimensions, format, etc.)
    void init( const TextureDescriptor& descriptor, const imageSource::TextureInfo& info );

    /// Check whether the texture has been initialized.
    bool isInitialized() const { return m_isInitialized; }

    /// Get the dimensions of the specified miplevel.
    uint2 getMipLevelDims( unsigned int mipLevel ) const
    {
        return m_array.getMipLevelDims( mipLevel );
    }

    /// Get the tile width, which depends on the format and number of channels.
    unsigned int getTileWidth() const { return m_array.getTileWidth(); }

    /// Get the tile height, which depends on the format and number of channels.
    unsigned int getTileHeight() const { return m_array.getTileHeight(); } 

    /// Get the miplevel at the start of the "mip tail".  The mip tail consists of the coarsest
    /// miplevels that fit into a single memory page.
    unsigned int getMipTailFirstLevel() const { return m_array.getMipTailFirstLevel(); }

    /// Get the size of the mip tail in bytes.
    size_t getMipTailSize() const { return m_array.getMipTailSize(); } 

    /// Get the CUDA texture object.
    CUtexObject getTextureObject() const { return m_texture; }

    /// Map the given backing storage for the specified tile into the sparse texture and fill it with the given data.
    void fillTile( CUstream                     stream,
                   unsigned int                 mipLevel,
                   unsigned int                 tileX,
                   unsigned int                 tileY,
                   const char*                  tileData,
                   CUmemorytype                 tileMemoryType,
                   size_t                       tileSize,
                   CUmemGenericAllocationHandle tileHandle,
                   size_t                       tileOffset ) const;

    /// Unmap the backing storage for the specified tile.
    void unmapTile( CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const;

    /// Map the given backing storage for mip tail into the sparse texture and fill it with the given data.
    void fillMipTail( CUstream                     stream,
                      const char*                  mipTailData,
                      CUmemorytype                 mipTailMemoryType,
                      size_t                       mipTailSize,
                      CUmemGenericAllocationHandle tileHandle,
                      size_t                       tileOffset ) const;

    /// Unmap the backing storage for the mip tail
    void unmapMipTail( CUstream stream ) const;

    /// Get total number of unmappings
    unsigned int getNumUnmappings() const { return m_numUnmappings; }

    /// Get total number of bytes filled
    size_t getNumBytesFilled() const { return m_numBytesFilled; }

  private:
    bool                         m_isInitialized = false;
    unsigned int                 m_deviceIndex;
    imageSource::TextureInfo     m_info{};
    SparseArray                  m_array;
    CUtexObject                  m_texture{};

    // Get the dimensions of the specified tile, which might be a partial tile.
    uint2 getTileDimensions( unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const;

    // Stats
    mutable unsigned int m_numUnmappings = 0;
    mutable size_t m_numBytesFilled = 0;
};

}  // namespace demandLoading
