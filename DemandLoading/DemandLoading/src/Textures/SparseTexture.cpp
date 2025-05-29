// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Textures/SparseTexture.h"
#include "Util/ContextSaver.h"

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <vector_functions.h> // from CUDA toolkit

#include <algorithm>
#include <cmath>

using namespace imageSource;

namespace demandLoading {

SparseArray::~SparseArray()
{
    if( m_initialized )
    {
        ContextSaver contextSaver;

        // It's not necessary to unmap the tiles / mip tail when destroying the array.
        OTK_ERROR_CHECK_NOTHROW( cuCtxSetCurrent( m_context ) );
        OTK_ERROR_CHECK_NOTHROW( cuMipmappedArrayDestroy( m_array ) );

        m_initialized = false;
    }
}

void SparseArray::init( const imageSource::TextureInfo& info )
{
    if( m_initialized && info == m_info )
        return;

    // Record device index and current CUDA context.
    CUdevice device;
    OTK_ERROR_CHECK( cuCtxGetDevice( &device ) );
    m_deviceIndex = static_cast<unsigned int>( device );
    OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );

    m_info = info;

    // Work around an invalid read (reported by valgrind) in cuMipmappedArrayCreate when the number
    // of miplevels is less than the start of the mip tail.  See bug 3139148.
    // Note that the texture descriptor clamps the maximum miplevel appropriately, and we'll never
    // map tiles (or the mip tail) beyond the actual maximum miplevel.
    const unsigned int nominalNumMipLevels = imageSource::calculateNumMipLevels( m_info.width, m_info.height );
    OTK_ASSERT( m_info.numMipLevels <= nominalNumMipLevels );

    // Create CUDA array
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = m_info.width;
    ad.Height      = m_info.height;
    ad.Format      = m_info.format;
    ad.NumChannels = m_info.numChannels;
    ad.Flags       = CUDA_ARRAY3D_SPARSE;
    OTK_ERROR_CHECK( cuMipmappedArrayCreate( &m_array, &ad, nominalNumMipLevels ) );

    // Get sparse texture properties
    OTK_ERROR_CHECK( cuMipmappedArrayGetSparseProperties( &m_properties, m_array ) );

    // Precompute array of mip level dimensions (for use in getTileDimensions).
    for( unsigned int mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        m_mipLevelDims.push_back( queryMipLevelDims( mipLevel ) );
    }

    m_initialized = true;
}

uint2 SparseArray::queryMipLevelDims( unsigned int mipLevel ) const
{
    // Get CUDA array for the specified level from the mipmapped array.
    OTK_ASSERT( mipLevel < m_info.numMipLevels );
    CUarray mipLevelArray = getLevel( mipLevel );

    // Get the array descriptor.
    CUDA_ARRAY_DESCRIPTOR desc;
    OTK_ERROR_CHECK( cuArrayGetDescriptor( &desc, mipLevelArray ) );

    return make_uint2( static_cast<unsigned int>( desc.Width ), static_cast<unsigned int>( desc.Height ) );
}

void SparseArray::mapTileAsync( CUstream stream, unsigned int mipLevel, uint2 levelOffset, uint2 levelExtent, CUmemGenericAllocationHandle memHandle, size_t offset ) const
{
    OTK_ASSERT( m_initialized );

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

    OTK_ERROR_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseArray::unmapTileAsync( CUstream stream, unsigned int mipLevel, uint2 levelOffset, uint2 levelExtent ) const
{
    OTK_ASSERT( m_initialized );

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

    OTK_ERROR_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseArray::mapMipTailAsync( CUstream stream, size_t mipTailSize, CUmemGenericAllocationHandle memHandle, size_t offset ) const
{
    OTK_ASSERT( m_initialized );

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

    OTK_ERROR_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseArray::unmapMipTailAsync( CUstream stream, size_t mipTailSize ) const
{
    OTK_ASSERT( m_initialized );

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

    OTK_ERROR_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}

void SparseTexture::init( const TextureDescriptor& descriptor, const imageSource::TextureInfo& info, std::shared_ptr<SparseArray> masterArray )
{
    // Redundant initialization can occur because requests from multiple streams are not yet deduplicated.
    if( m_isInitialized && info == m_info )
        return;

    // Record current CUDA context.
    m_info = info;
    OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );

    // Set the array to a new array, or the master array if one was passed in
    m_array = masterArray;
    if( m_array.get() == nullptr )
    {
        m_array.reset( new SparseArray() );
        m_array->init( m_info );
    }

    // Create CUDA texture descriptor
    CUDA_TEXTURE_DESC td{};
    td.addressMode[0]      = descriptor.addressMode[0];
    td.addressMode[1]      = descriptor.addressMode[1];
    td.filterMode          = toCudaFilterMode( descriptor.filterMode );
    td.flags               = CU_TRSF_NORMALIZED_COORDINATES | descriptor.flags;
    td.maxAnisotropy       = descriptor.maxAnisotropy;
    td.mipmapFilterMode    = descriptor.mipmapFilterMode;
    td.maxMipmapLevelClamp = float( m_info.numMipLevels - 1 );
    td.minMipmapLevelClamp = 0.f;

    // Create texture object.
    CUDA_RESOURCE_DESC rd{};
    rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    rd.res.mipmap.hMipmappedArray = static_cast<CUmipmappedArray>( *m_array );
    OTK_ERROR_CHECK( cuTexObjectCreate( &m_texture, &rd, &td, nullptr ) );

    m_isInitialized = true;
}


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


void SparseTexture::mapTile( CUstream stream,
                             unsigned int                 mipLevel,
                             unsigned int                 tileX,
                             unsigned int                 tileY,
                             CUmemGenericAllocationHandle tileHandle,
                             size_t                       tileOffset ) const
{
    OTK_ASSERT( m_isInitialized );
    const uint2 tileDims{getTileDimensions( mipLevel, tileX, tileY )};
    const uint2 levelOffset{make_uint2( tileX * getTileWidth(), tileY * getTileHeight() )};
    m_array->mapTileAsync(stream, mipLevel, levelOffset, tileDims, tileHandle, tileOffset);
}


void SparseTexture::fillTile( CUstream                     stream,
                              unsigned int                 mipLevel,
                              unsigned int                 tileX,
                              unsigned int                 tileY,
                              const char*                  tileData,
                              CUmemorytype                 tileMemoryType,
                              size_t                       /*tileSize*/,
                              CUmemGenericAllocationHandle tileHandle,
                              size_t                       tileOffset ) const
{
    OTK_ASSERT( m_isInitialized );

    const uint2 tileDims{getTileDimensions( mipLevel, tileX, tileY )};
    mapTile( stream, mipLevel, tileX, tileY, tileHandle, tileOffset );

    // Get CUDA array for the specified miplevel.
    CUarray mipLevelArray = m_array->getLevel( mipLevel );
    int blockScale = imageSource::isBcFormat( m_info.format ) ? 4 : 1;

    // Copy tile data into CUDA array
    const unsigned int bitsPerPixel = getBitsPerPixel( m_info );

    CUDA_MEMCPY2D copyArgs{};
    copyArgs.srcMemoryType = tileMemoryType;
    copyArgs.srcHost       = ( tileMemoryType == CU_MEMORYTYPE_HOST ) ? tileData : nullptr;
    copyArgs.srcDevice     = ( tileMemoryType == CU_MEMORYTYPE_DEVICE ) ? reinterpret_cast<CUdeviceptr>( tileData ) : 0;
    copyArgs.srcPitch      = ( blockScale * getTileWidth() * bitsPerPixel ) / BITS_PER_BYTE;

    copyArgs.dstXInBytes   = ( blockScale * tileX * getTileWidth() * bitsPerPixel ) / BITS_PER_BYTE;
    copyArgs.dstY          = tileY * getTileHeight() / blockScale;
    copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyArgs.dstArray      = mipLevelArray;

    copyArgs.WidthInBytes  = ( blockScale * tileDims.x * bitsPerPixel ) / BITS_PER_BYTE;
    copyArgs.Height        = tileDims.y / blockScale;

    OTK_ERROR_CHECK( cuMemcpy2DAsync( &copyArgs, stream ) );
    m_numBytesFilled += ( getTileWidth() * getTileHeight() * bitsPerPixel ) / BITS_PER_BYTE;
}


void SparseTexture::unmapTile( CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const
{
    OTK_ASSERT( m_isInitialized );

    const uint2 levelExtent{getTileDimensions( mipLevel, tileX, tileY )};
    const uint2 levelOffset{make_uint2( tileX * getTileWidth(), tileY * getTileHeight() )};
    m_array->unmapTileAsync( stream, mipLevel, levelOffset, levelExtent );
    m_numUnmappings++;
}


void SparseTexture::mapMipTail( CUstream stream, CUmemGenericAllocationHandle tileHandle, size_t tileOffset ) const
{
    OTK_ASSERT( m_isInitialized );
    m_array->mapMipTailAsync(stream, getMipTailSize(), tileHandle, tileOffset);
}


void SparseTexture::fillMipTail( CUstream                     stream,
                                 const char*                  mipTailData,
                                 CUmemorytype                 mipTailMemoryType,
                                 size_t                       mipTailSize,
                                 CUmemGenericAllocationHandle tileHandle,
                                 size_t                       tileOffset ) const
{
    OTK_ASSERT( m_isInitialized );
    OTK_ASSERT( mipTailSize >= getMipTailSize() );
    (void)mipTailSize;  // silence unused variable warning.

    m_array->mapMipTailAsync(stream, getMipTailSize(), tileHandle, tileOffset);

    int bitsPerPixel = getBitsPerPixel( m_info );
    int blockScale   = isBcFormat( m_info.format ) ? 4 : 1;
    size_t offset    = 0;

    // Fill each level in the mip tail.
    for( unsigned int mipLevel = getMipTailFirstLevel(); mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        CUarray mipLevelArray = m_array->getLevel( mipLevel );
        uint2 levelDims = getMipLevelDims( mipLevel );

        CUDA_MEMCPY2D copyArgs{};
        copyArgs.srcMemoryType = mipTailMemoryType;
        copyArgs.srcHost       = ( mipTailMemoryType == CU_MEMORYTYPE_HOST ) ? mipTailData + offset : nullptr;
        copyArgs.srcDevice     = ( mipTailMemoryType == CU_MEMORYTYPE_DEVICE ) ? reinterpret_cast<CUdeviceptr>( mipTailData + offset ) : 0;
        copyArgs.srcPitch      = ( blockScale * levelDims.x * bitsPerPixel ) / BITS_PER_BYTE;

        copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyArgs.dstArray      = mipLevelArray;

        copyArgs.WidthInBytes  = copyArgs.srcPitch;
        copyArgs.Height        = levelDims.y / blockScale;

        OTK_ERROR_CHECK( cuMemcpy2DAsync( &copyArgs, stream ) );

        offset += ( levelDims.x * levelDims.y * bitsPerPixel ) / BITS_PER_BYTE;
    }

    m_numBytesFilled += getMipTailSize();
}

void SparseTexture::unmapMipTail( CUstream stream ) const
{
    OTK_ASSERT( m_isInitialized );

    m_array->unmapMipTailAsync(stream, getMipTailSize());
    m_numUnmappings++;
}


SparseTexture::~SparseTexture()
{
    if( m_isInitialized )
    {
        ContextSaver contextSaver;
        OTK_ERROR_CHECK_NOTHROW( cuCtxSetCurrent( m_context ) );
        OTK_ERROR_CHECK_NOTHROW( cuTexObjectDestroy( m_texture ) );
    }
}

}  // namespace demandLoading
