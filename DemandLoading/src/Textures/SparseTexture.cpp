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

#include "Textures/SparseTexture.h"
#include "Util/Exception.h"

#include <ImageSource/ImageSource.h>

#include <algorithm>
#include <cmath>

namespace demandLoading {

SparseArray::~SparseArray()
{
    if( m_initialized )
    {
        // It's not necessary to unmap the tiles / mip tail when destroying the array.
        DEMAND_CUDA_CHECK_NOTHROW( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK_NOTHROW( cuMipmappedArrayDestroy( m_array ) );

        m_initialized = false;
    }
}

void SparseArray::init( unsigned int deviceIndex, const imageSource::TextureInfo& info )
{
    if( m_initialized )
        return;

    m_deviceIndex = deviceIndex;
    m_info = info;

    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Work around an invalid read (reported by valgrind) in cuMipmappedArrayCreate when the number
    // of miplevels is less than the start of the mip tail.  See bug 3139148.
    // Note that the texture descriptor clamps the maximum miplevel appropriately, and we'll never
    // map tiles (or the mip tail) beyond the actual maximum miplevel.
    const unsigned int nominalNumMipLevels = imageSource::calculateNumMipLevels( m_info.width, m_info.height );
    DEMAND_ASSERT( m_info.numMipLevels <= nominalNumMipLevels );

    // Create CUDA array
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = m_info.width;
    ad.Height      = m_info.height;
    ad.Format      = m_info.format;
    ad.NumChannels = m_info.numChannels;
    ad.Flags       = CUDA_ARRAY3D_SPARSE;
    DEMAND_CUDA_CHECK( cuMipmappedArrayCreate( &m_array, &ad, nominalNumMipLevels ) );

    // Get sparse texture properties
    DEMAND_CUDA_CHECK( cuMipmappedArrayGetSparseProperties( &m_properties, m_array ) );

    // Precompute array of mip level dimensions (for use in getTileDimensions).
    for( unsigned int mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        m_mipLevelDims.push_back( queryMipLevelDims( mipLevel ) );
    }

    m_initialized = true;
}

uint2 SparseArray::queryMipLevelDims( unsigned int mipLevel ) const
{
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Get CUDA array for the specified level from the mipmapped array.
    DEMAND_ASSERT( mipLevel < m_info.numMipLevels );
    CUarray mipLevelArray = getLevel( mipLevel );

    // Get the array descriptor.
    CUDA_ARRAY_DESCRIPTOR desc;
    DEMAND_CUDA_CHECK( cuArrayGetDescriptor( &desc, mipLevelArray ) );

    return make_uint2( static_cast<unsigned int>( desc.Width ), static_cast<unsigned int>( desc.Height ) );
}

void SparseArray::mapTileAsync( CUstream stream, unsigned int mipLevel, uint2 levelOffset, uint2 levelExtent, CUmemGenericAllocationHandle memHandle, size_t offset ) const
{
    DEMAND_ASSERT( m_initialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Map tile backing storage into array
    CUarrayMapInfo mapInfo{}; 
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_array;

    mapInfo.subresourceType               = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level = mipLevel;

    mapInfo.subresource.sparseLevel.offsetX = levelOffset.x;
    mapInfo.subresource.sparseLevel.offsetY = levelOffset.y;

    mapInfo.subresource.sparseLevel.extentWidth  = levelExtent.x;
    mapInfo.subresource.sparseLevel.extentHeight = levelExtent.y;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = memHandle;
    mapInfo.offset              = offset;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseArray::unmapTileAsync( CUstream stream, unsigned int mipLevel, uint2 levelOffset, uint2 levelExtent ) const
{
    DEMAND_ASSERT( m_initialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_array;

    mapInfo.subresourceType               = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level = mipLevel;

    mapInfo.subresource.sparseLevel.offsetX = levelOffset.x;
    mapInfo.subresource.sparseLevel.offsetY = levelOffset.y;

    mapInfo.subresource.sparseLevel.extentWidth  = levelExtent.x;
    mapInfo.subresource.sparseLevel.extentHeight = levelExtent.y;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = 0ULL;
    mapInfo.offset              = 0ULL;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseArray::mapMipTailAsync( CUstream stream, size_t mipTailSize, CUmemGenericAllocationHandle memHandle, size_t offset ) const
{
    DEMAND_ASSERT( m_initialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_array;

    mapInfo.subresourceType            = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = mipTailSize;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = memHandle;
    mapInfo.offset              = offset;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseArray::unmapMipTailAsync( CUstream stream, size_t mipTailSize ) const
{
    DEMAND_ASSERT( m_initialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = static_cast<CUmipmappedArray>( m_array );

    mapInfo.subresourceType            = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = mipTailSize;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = 0ULL;
    mapInfo.offset              = 0ULL;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseTexture::init( const TextureDescriptor& descriptor, const imageSource::TextureInfo& info )
{
    // Redundant initialization can occur because requests from multiple streams are not yet
    // deduplicated.
    if( m_isInitialized )
        return;

    m_info = info;
    m_array.init( m_deviceIndex, m_info );

    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Create CUDA texture descriptor
    CUDA_TEXTURE_DESC td{};
    td.addressMode[0]      = descriptor.addressMode[0];
    td.addressMode[1]      = descriptor.addressMode[1];
    td.filterMode          = descriptor.filterMode;
    td.flags               = CU_TRSF_NORMALIZED_COORDINATES | descriptor.flags;
    td.maxAnisotropy       = descriptor.maxAnisotropy;
    td.mipmapFilterMode    = descriptor.mipmapFilterMode;
    td.maxMipmapLevelClamp = float( info.numMipLevels - 1 );
    td.minMipmapLevelClamp = 0.f;

    // Create texture object.
    CUDA_RESOURCE_DESC rd{};
    rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    rd.res.mipmap.hMipmappedArray = static_cast<CUmipmappedArray>( m_array );
    DEMAND_CUDA_CHECK( cuTexObjectCreate( &m_texture, &rd, &td, nullptr ) );

    m_isInitialized = true;
};


// Get the dimensions of the specified tile, which might be a partial tile.
uint2 SparseTexture::getTileDimensions( unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const
{
    unsigned int startX = tileX * getTileWidth();
    unsigned int startY = tileY * getTileHeight();
    unsigned int endX   = startX + getTileWidth();
    unsigned int endY   = startY + getTileHeight();

    // TODO: cache the level dimensions.
    uint2 levelDims = getMipLevelDims( mipLevel );
    endX            = std::min( endX, levelDims.x );
    endY            = std::min( endY, levelDims.y );

    return make_uint2( endX - startX, endY - startY );
}


void SparseTexture::fillTile( CUstream                     stream,
                              unsigned int                 mipLevel,
                              unsigned int                 tileX,
                              unsigned int                 tileY,
                              const char*                  tileData,
                              CUmemorytype                 tileMemoryType,
                              size_t                       tileSize,
                              CUmemGenericAllocationHandle tileHandle,
                              size_t                       tileOffset ) const
{
    DEMAND_ASSERT( m_isInitialized );

    const uint2 tileDims{getTileDimensions( mipLevel, tileX, tileY )};
    const uint2 levelOffset{make_uint2( tileX * getTileWidth(), tileY * getTileHeight() )};
    m_array.mapTileAsync(stream, mipLevel, levelOffset, tileDims, tileHandle, tileOffset);

    // Get CUDA array for the specified miplevel.
    CUarray mipLevelArray = m_array.getLevel( mipLevel );

    // Copy tile data into CUDA array
    const unsigned int pixelSize = m_info.numChannels * imageSource::getBytesPerChannel( m_info.format );

    CUDA_MEMCPY2D copyArgs{};
    copyArgs.srcMemoryType = tileMemoryType;
    copyArgs.srcHost       = ( tileMemoryType == CU_MEMORYTYPE_HOST ) ? tileData : nullptr;
    copyArgs.srcDevice     = ( tileMemoryType == CU_MEMORYTYPE_DEVICE ) ? reinterpret_cast<CUdeviceptr>( tileData ) : 0;
    copyArgs.srcPitch      = getTileWidth() * pixelSize;

    copyArgs.dstXInBytes = tileX * getTileWidth() * pixelSize;
    copyArgs.dstY        = tileY * getTileHeight();

    copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyArgs.dstArray      = mipLevelArray;

    copyArgs.WidthInBytes = tileDims.x * pixelSize;
    copyArgs.Height       = tileDims.y;

    DEMAND_CUDA_CHECK( cuMemcpy2DAsync( &copyArgs, stream ) );
    m_numBytesFilled += getTileWidth() * getTileHeight() * pixelSize;
}


void SparseTexture::unmapTile( CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const
{
    DEMAND_ASSERT( m_isInitialized );

    const uint2 levelExtent{getTileDimensions( mipLevel, tileX, tileY )};
    const uint2 levelOffset{make_uint2( tileX * getTileWidth(), tileY * getTileHeight() )};
    m_array.unmapTileAsync( stream, mipLevel, levelOffset, levelExtent );
    m_numUnmappings++;
}


void SparseTexture::fillMipTail( CUstream                     stream,
                                 const char*                  mipTailData,
                                 CUmemorytype                 mipTailMemoryType,
                                 size_t                       mipTailSize,
                                 CUmemGenericAllocationHandle tileHandle,
                                 size_t                       tileOffset ) const
{
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_ASSERT( mipTailSize >= getMipTailSize() );

    m_array.mapMipTailAsync(stream, getMipTailSize(), tileHandle, tileOffset);

    // Fill each level in the mip tail.
    size_t             offset    = 0;
    const unsigned int pixelSize = m_info.numChannels * imageSource::getBytesPerChannel( m_info.format );
    for( unsigned int mipLevel = getMipTailFirstLevel(); mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        CUarray mipLevelArray = m_array.getLevel( mipLevel );
        uint2 levelDims = getMipLevelDims( mipLevel );

        CUDA_MEMCPY2D copyArgs{};
        copyArgs.srcMemoryType = mipTailMemoryType;
        copyArgs.srcHost       = ( mipTailMemoryType == CU_MEMORYTYPE_HOST ) ? mipTailData + offset : nullptr;
        copyArgs.srcDevice     = ( mipTailMemoryType == CU_MEMORYTYPE_DEVICE ) ? reinterpret_cast<CUdeviceptr>( mipTailData + offset ) : 0;
        copyArgs.srcPitch      = levelDims.x * pixelSize;

        copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyArgs.dstArray      = mipLevelArray;

        copyArgs.WidthInBytes = levelDims.x * pixelSize;
        copyArgs.Height       = levelDims.y;

        DEMAND_CUDA_CHECK( cuMemcpy2DAsync( &copyArgs, stream ) );

        offset += levelDims.x * levelDims.y * pixelSize;
    }

    m_numBytesFilled += getMipTailSize();
}

void SparseTexture::unmapMipTail( CUstream stream ) const
{
    DEMAND_ASSERT( m_isInitialized );

    m_array.unmapMipTailAsync(stream, getMipTailSize());
    m_numUnmappings++;
}


SparseTexture::~SparseTexture()
{
    if( m_isInitialized )
    {
        DEMAND_CUDA_CHECK_NOTHROW( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK_NOTHROW( cuTexObjectDestroy( m_texture ) );
    }
}

}  // namespace demandLoading
