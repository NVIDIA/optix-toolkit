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

#include "Textures/DemandTextureImpl.h"
#include "DemandLoaderImpl.h"
#include "Memory/MemoryBlockDesc.h"
#include "PageTableManager.h"
#include "Textures/TextureRequestHandler.h"
#include "Util/Math.h"
#include "Util/Stopwatch.h"

#include <OptiXToolkit/DemandLoading/TileIndexing.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace demandLoading {

DemandTextureImpl::DemandTextureImpl( unsigned int                              id,
                                      unsigned int                              maxNumDevices,
                                      const TextureDescriptor&                  descriptor,
                                      std::shared_ptr<imageSource::ImageSource> image,
                                      DemandLoaderImpl*                         loader )

    : m_id( id )
    , m_descriptor( descriptor )
    , m_image( image )
    , m_masterTexture( nullptr )
    , m_loader( loader )
{
    initPerDeviceTextures( maxNumDevices );
}

DemandTextureImpl::DemandTextureImpl( unsigned int id, DemandTextureImpl* masterTexture, const TextureDescriptor& descriptor, DemandLoaderImpl* loader )
    : m_id( id )
    , m_descriptor( descriptor )
    , m_image( masterTexture->m_image )
    , m_masterTexture( masterTexture )
    , m_loader( loader )
{
    initPerDeviceTextures( static_cast<unsigned int>( masterTexture->m_sparseTextures.size() ) );
}

void DemandTextureImpl::initPerDeviceTextures( unsigned int maxNumDevices )
{
    // Construct per-device sparse and dense textures.  These are just empty shells until they are initialized.  
    // Note that the vectors do not grow after construction, which is important for thread safety.
    m_sparseTextures.reserve( maxNumDevices );
    m_denseTextures.reserve( maxNumDevices );
    for( unsigned int i = 0; i < maxNumDevices; ++i )
    {
        m_sparseTextures.emplace_back( i );
        m_denseTextures.emplace_back( i );
    }

    m_sampler = {0};
}

bool DemandTextureImpl::setImage( const TextureDescriptor& descriptor, std::shared_ptr<imageSource::ImageSource> image )
{
    std::unique_lock<std::mutex> lock( m_initMutex );

    // If the original image was not opened, just replace it
    if( !m_image->isOpen() )
    {
        m_descriptor = descriptor;
        m_image = image;
        return false;
    }

    // If the two images are the same size and format, replace the image and use the existing textures
    imageSource::TextureInfo newInfo;
    image->open( &newInfo );
    DEMAND_ASSERT( m_info.isValid );
    if( ( descriptor == m_descriptor ) && ( newInfo == m_info ) )
    {
        m_descriptor = descriptor;
        m_image      = image;
        return false;
    }

    // If there is a size or format mismatch, create new textures
    // FIXME: This leaks pages in the virtual address space, but currently there is no way to reclaim them.
    m_info       = newInfo;
    m_descriptor = descriptor;
    m_image      = image;

    unsigned int maxNumDevices = static_cast<unsigned int>( m_sparseTextures.size() );
    m_sparseTextures.clear();
    m_denseTextures.clear();
    m_sparseTextures.reserve( maxNumDevices );
    m_denseTextures.reserve( maxNumDevices );

    for( unsigned int i = 0; i < maxNumDevices; ++i )
    {
        m_sparseTextures.emplace_back( i );
        m_denseTextures.emplace_back( i );
    }
    m_sampler       = {0};
    m_isInitialized = false;

    // Return true, the sampler needs to be replaced on the devices.  The cuda arrays and cuda textures 
    // will be updated when the sparse/dense textures for each device are re-initialized.
    return true; 
}

unsigned int DemandTextureImpl::getId() const
{
    return m_id;
}

void DemandTextureImpl::init( unsigned int deviceIndex )
{
    std::unique_lock<std::mutex> lock( m_initMutex );

    if( m_masterTexture && !m_masterTexture->m_sparseTextures[deviceIndex].isInitialized() )
        m_masterTexture->init( deviceIndex );

    // Initialize the sparse or dense texture for the specified device.
    if( useSparseTexture() )
    {
        // Per-device initialization.
        DEMAND_ASSERT( deviceIndex < m_sparseTextures.size() );
        
        // Get the master array (backing store) if there is master texture
        std::shared_ptr<SparseArray> masterArray( nullptr );
        if( m_masterTexture )
            masterArray = m_masterTexture->m_sparseTextures[deviceIndex].getSparseArray();

        SparseTexture& sparseTexture = m_sparseTextures[deviceIndex];
        sparseTexture.init( m_descriptor, m_info, masterArray );

        // Device-independent initialization.
        if( !m_isInitialized )
        {
            m_isInitialized = true;

            // Retain various properties for subsequent use.  (They're the same on all devices.)
            m_tileWidth         = sparseTexture.getTileWidth();
            m_tileHeight        = sparseTexture.getTileHeight();
            m_mipTailFirstLevel = sparseTexture.getMipTailFirstLevel();
            m_mipTailSize       = m_mipTailFirstLevel < m_info.numMipLevels ? sparseTexture.getMipTailSize() : 0;

            // Verify that the tile size agrees with TilePool.
            DEMAND_ASSERT( m_tileWidth * m_tileHeight * imageSource::getBytesPerChannel( m_info.format ) <= TILE_SIZE_IN_BYTES );

            // Record the dimensions of each miplevel.
            const unsigned int numMipLevels = m_info.numMipLevels;
            m_mipLevelDims.resize( numMipLevels );
            for( unsigned int i = 0; i < numMipLevels; ++i )
            {
                m_mipLevelDims[i] = sparseTexture.getMipLevelDims( i );
            }

            initSampler();
        }
    }
    else // dense texture
    {
        // Per-device initialization.
        DEMAND_ASSERT( deviceIndex < m_denseTextures.size() );

        // Get the master array (backing store) if there is master texture
        std::shared_ptr<CUmipmappedArray> masterArray( nullptr );
        if( m_masterTexture )
            masterArray = m_masterTexture->m_denseTextures[deviceIndex].getDenseArray();

        DenseTexture& denseTexture = m_denseTextures[deviceIndex];
        denseTexture.init( m_descriptor, m_info, masterArray );

        // Device-independent initialization.
        if( !m_isInitialized )
        {
            m_isInitialized = true;

            // Set dummy properties (not used for dense textures)
            m_tileWidth         = 64;
            m_tileHeight        = 64;
            m_mipTailFirstLevel = 0;
            m_mipTailSize       = 0;

            // Record the dimensions of each miplevel.
            const unsigned int numMipLevels = m_info.numMipLevels;
            m_mipLevelDims.resize( numMipLevels );
            for( unsigned int i = 0; i < numMipLevels; ++i )
            {
                m_mipLevelDims[i] = denseTexture.getMipLevelDims( i );
                m_mipTailSize += m_mipLevelDims[i].x * m_mipLevelDims[i].y * m_info.numChannels
                                 * imageSource::getBytesPerChannel( m_info.format );
            }

            initSampler();
        }
    }
}

void DemandTextureImpl::initSampler()
{
    // Construct the canonical sampler for this texture, excluding the CUDA texture object, which
    // differs for each device (see getTextureObject).
    
    // Note: m_sampler zeroed out in constructor, so that the udim fields can be initialized before calling initSampler.

    // Descriptions
    m_sampler.desc.numMipLevels     = m_info.numMipLevels;
    m_sampler.desc.logTileWidth     = static_cast<unsigned int>( log2f( static_cast<float>( m_tileWidth ) ) );
    m_sampler.desc.logTileHeight    = static_cast<unsigned int>( log2f( static_cast<float>( m_tileHeight ) ) );
    m_sampler.desc.isSparseTexture  = useSparseTexture() ? 1 : 0;
    m_sampler.desc.wrapMode0        = static_cast<int>( m_descriptor.addressMode[0] );
    m_sampler.desc.wrapMode1        = static_cast<int>( m_descriptor.addressMode[1] );
    m_sampler.desc.mipmapFilterMode = m_descriptor.mipmapFilterMode;
    m_sampler.desc.maxAnisotropy    = m_descriptor.maxAnisotropy;

    // Dimensions
    m_sampler.width             = m_info.width;
    m_sampler.height            = m_info.height;
    m_sampler.mipTailFirstLevel = m_mipTailFirstLevel;

    // Initialize mipLevelSizes
    TextureSampler::MipLevelSizes* mls = m_sampler.mipLevelSizes;
    memset( mls, 0, MAX_TILE_LEVELS * sizeof(TextureSampler::MipLevelSizes ) );

    if( m_sampler.desc.isSparseTexture )
    {
        // Calculate number of tiles for sparse texture
        for( int mipLevel = static_cast<int>( m_sampler.mipTailFirstLevel ); mipLevel >= 0; --mipLevel )
        {
            if( mipLevel < static_cast<int>( m_sampler.mipTailFirstLevel ) )
                mls[mipLevel].mipLevelStart = mls[mipLevel + 1].mipLevelStart + getNumTilesInLevel( mipLevel + 1 );
            else
                mls[mipLevel].mipLevelStart = 0;

            mls[mipLevel].levelWidthInTiles = static_cast<unsigned short>(
                getLevelDimInTiles( m_sampler.width, static_cast<unsigned int>( mipLevel ), m_tileWidth ) );
            mls[mipLevel].levelHeightInTiles = static_cast<unsigned short>(
                getLevelDimInTiles( m_sampler.height, static_cast<unsigned int>( mipLevel ), m_tileHeight ) );
        }
        m_sampler.numPages = mls[0].mipLevelStart + getNumTilesInLevel( 0 );

        // Reserve a range of page table entries, one per tile, associated with the page request
        // handler for this texture.
        if( m_masterTexture )
        {
            m_sampler.startPage = m_masterTexture->m_sampler.startPage;
        }
        else
        {
            m_requestHandler.reset( new TextureRequestHandler( this, m_loader ) );
            m_sampler.startPage = m_loader->getPageTableManager()->reserve( m_sampler.numPages, m_requestHandler.get() );       
        }
    }
    else // Dense texture 
    {
        // Dense textures do not need extra page table entries
        m_sampler.numPages = 0;
        m_sampler.startPage = m_id;
    }
}

const imageSource::TextureInfo& DemandTextureImpl::getInfo() const
{
    DEMAND_ASSERT( m_isInitialized );
    return m_info;
}

const TextureSampler& DemandTextureImpl::getSampler() const
{
    DEMAND_ASSERT( m_isInitialized );
    return m_sampler;
}

const TextureDescriptor& DemandTextureImpl::getDescriptor() const
{
    return m_descriptor;
}

uint2 DemandTextureImpl::getMipLevelDims( unsigned int mipLevel ) const
{
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_ASSERT( mipLevel < m_mipLevelDims.size() );
    return m_mipLevelDims[mipLevel];
}

unsigned int DemandTextureImpl::getTileWidth() const
{
    DEMAND_ASSERT( m_isInitialized );
    return m_tileWidth;
}

unsigned int DemandTextureImpl::getTileHeight() const
{
    DEMAND_ASSERT( m_isInitialized );
    return m_tileHeight;
}

bool DemandTextureImpl::isMipmapped() const
{
    DEMAND_ASSERT( m_info.isValid );
    return getInfo().numMipLevels > getMipTailFirstLevel();
}

bool DemandTextureImpl::useSparseTexture() const
{
    DEMAND_ASSERT( m_info.isValid );

    if( !m_loader->getOptions().useSparseTextures || !m_info.isTiled )
        return false;
    if( !m_loader->getOptions().useSmallTextureOptimization )
        return true;
    return m_info.width * m_info.height > SPARSE_TEXTURE_THRESHOLD;
}

unsigned int DemandTextureImpl::getMipTailFirstLevel() const
{
    DEMAND_ASSERT( m_isInitialized );
    return m_mipTailFirstLevel;
}

CUtexObject DemandTextureImpl::getTextureObject( unsigned int deviceIndex ) const
{
    DEMAND_ASSERT( m_isInitialized );
    if( useSparseTexture() )
    {
        DEMAND_ASSERT( deviceIndex < m_sparseTextures.size() );
        return m_sparseTextures[deviceIndex].getTextureObject();
    }

    DEMAND_ASSERT(deviceIndex < m_denseTextures.size());
    return m_denseTextures[deviceIndex].getTextureObject();
}

unsigned int DemandTextureImpl::getNumTilesInLevel( unsigned int mipLevel ) const
{
    if( mipLevel > m_mipTailFirstLevel || mipLevel >= m_info.numMipLevels )
        return 0;

    unsigned int levelWidthInTiles  = getLevelDimInTiles( m_mipLevelDims[0].x, mipLevel, m_tileWidth );
    unsigned int levelHeightInTiles = getLevelDimInTiles( m_mipLevelDims[0].y, mipLevel, m_tileHeight );

    return calculateNumTilesInLevel( levelWidthInTiles, levelHeightInTiles );
}

void DemandTextureImpl::accumulateStatistics( Statistics& stats, std::set<imageSource::ImageSource*>& images )
{
    // This image has already been accounted for in the statistics.
    if( images.find( m_image.get() ) != images.end() )
        return;

    // Get the size of the texture, and number of bytes read
    images.insert( m_image.get() );
    stats.numTilesRead += m_image->getNumTilesRead();
    stats.numBytesRead += m_image->getNumBytesRead();
    stats.readTime += m_image->getTotalReadTime();

    // Calculate size (total number of bytes) in virtual texture
    const imageSource::TextureInfo &info = m_image->getInfo();
    if( info.isValid )  // texture initialized
    {
        stats.virtualTextureBytes += getTextureSizeInBytes( info );
    }

    // Get the number of bytes filled (transferred) per device
    for( unsigned int i = 0; i < m_sparseTextures.size(); ++i )
    {
        stats.perDevice[i].bytesTransferred += m_sparseTextures[i].getNumBytesFilled();
        stats.perDevice[i].numEvictions += m_sparseTextures[i].getNumUnmappings();
    }

    for( unsigned int i = 0; i < m_denseTextures.size(); ++i )
    {
        stats.perDevice[i].bytesTransferred += m_denseTextures[i].getNumBytesFilled();
        // Count memory used per device for dense texture data
        if( m_denseTextures[i].isInitialized() && m_denseTextures[i].getTextureObject() != 0 )
        {
            imageSource::TextureInfo info = m_image->getInfo();
            stats.perDevice[i].memoryUsed += getTextureSizeInBytes( info );
        }
    }
}

// Tiles can be read concurrently.  The EXRReader currently locks, however, because the OpenEXR 2.x
// tile reading API is stateful.  That should be fixed in OpenEXR 3.0.
bool DemandTextureImpl::readTile( unsigned int mipLevel, unsigned int tileX, unsigned int tileY, char* tileBuffer,
                                  size_t tileBufferSize, CUstream stream ) const
{
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_ASSERT( mipLevel < m_info.numMipLevels );

    // Resize buffer if necessary.
    const unsigned int bytesPerPixel = imageSource::getBytesPerChannel( getInfo().format ) * getInfo().numChannels;
    const unsigned int bytesPerTile  = getTileWidth() * getTileHeight() * bytesPerPixel;
    DEMAND_ASSERT_MSG( bytesPerTile <= tileBufferSize, "Maximum tile size exceeded" );

    return m_image->readTile( tileBuffer, mipLevel, tileX, tileY, getTileWidth(), getTileHeight(), stream );
}

// Tiles can be filled concurrently.
void DemandTextureImpl::fillTile( unsigned int                 deviceIndex,
                                  CUstream                     stream,
                                  unsigned int                 mipLevel,
                                  unsigned int                 tileX,
                                  unsigned int                 tileY,
                                  const char*                  tileData,
                                  CUmemorytype                 tileDataType,
                                  size_t                       tileSize,
                                  CUmemGenericAllocationHandle handle,
                                  size_t                       offset ) const
{
    DEMAND_ASSERT( deviceIndex < m_sparseTextures.size() );
    DEMAND_ASSERT( mipLevel < m_info.numMipLevels );
    DEMAND_ASSERT( tileSize <= TILE_SIZE_IN_BYTES );

    m_sparseTextures[deviceIndex].fillTile( stream, mipLevel, tileX, tileY, tileData, tileDataType, tileSize, handle, offset );
}

// Tiles can be unmapped concurrently.
void DemandTextureImpl::unmapTile( unsigned int deviceIndex, CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const
{
    DEMAND_ASSERT( deviceIndex < m_sparseTextures.size() );
    DEMAND_ASSERT( mipLevel < m_info.numMipLevels );
    m_sparseTextures[deviceIndex].unmapTile( stream, mipLevel, tileX, tileY );
}

bool DemandTextureImpl::readNonMipMappedData( char* buffer, size_t bufferSize, CUstream stream ) const
{
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_ASSERT( m_info.numMipLevels == 1 );
    DEMAND_ASSERT_MSG( m_mipTailSize <= bufferSize, "Provided buffer is too small." );

    return m_image->readMipLevel( buffer, 0, getInfo().width, getInfo().height, stream );
}

bool DemandTextureImpl::readMipTail( char* buffer, size_t bufferSize, CUstream stream ) const
{
    return readMipLevels( buffer, bufferSize, getMipTailFirstLevel(), stream );
}

// Request deduplication will ensure that concurrent calls to readMipTail do not occur.  Note that
// EXRReader currently locks, since it uses the OpenEXR 2.x tile reading API, which is stateful.  
// CoreEXRReader uses OpenEXR 3.0, which fixes the issue.
bool DemandTextureImpl::readMipLevels( char* buffer, size_t bufferSize, unsigned int startLevel, CUstream stream ) const
{
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_ASSERT( startLevel < getInfo().numMipLevels );

    const unsigned int pixelSize = getInfo().numChannels * imageSource::getBytesPerChannel( getInfo().format );
    size_t dataSize = ( m_mipLevelDims[startLevel].x * m_mipLevelDims[startLevel].y * pixelSize * 4 ) / 3;

    DEMAND_ASSERT_MSG( dataSize <= bufferSize, "Provided buffer is too small." );

    return m_image->readMipTail( buffer, startLevel, getInfo().numMipLevels, m_mipLevelDims.data(), pixelSize, stream );
} 

void DemandTextureImpl::fillMipTail( unsigned int                 deviceIndex,
                                     CUstream                     stream,
                                     const char*                  mipTailData,
                                     CUmemorytype                 mipTailDataType,
                                     size_t                       mipTailSize,
                                     CUmemGenericAllocationHandle handle,
                                     size_t                       offset ) const
{
    DEMAND_ASSERT( deviceIndex < m_sparseTextures.size() );
    DEMAND_ASSERT( getMipTailFirstLevel() < m_info.numMipLevels );

    m_sparseTextures[deviceIndex].fillMipTail( stream, mipTailData, mipTailDataType, mipTailSize, handle, offset );
}

void DemandTextureImpl::unmapMipTail( unsigned int deviceIndex, CUstream stream ) const
{
    DEMAND_ASSERT( deviceIndex < m_sparseTextures.size() );
    m_sparseTextures[deviceIndex].unmapMipTail( stream );
}

// Fill the dense texture on the given device.
void DemandTextureImpl::fillDenseTexture( unsigned int deviceIndex, CUstream stream, const char* textureData, unsigned int width, unsigned int height, bool bufferPinned )
{
    DEMAND_ASSERT( deviceIndex < m_denseTextures.size() );
    m_denseTextures[deviceIndex].fillTexture( stream, textureData, width, height, bufferPinned );
}

// Lazily open the associated image source.
void DemandTextureImpl::open()
{
    std::unique_lock<std::mutex> lock( m_initMutex );

    // Open the image if necessary, fetching the dimensions and other info.
    if( !m_isOpen )
    {
        m_image->open( &m_info );
        DEMAND_ASSERT( m_info.isValid );
        m_isOpen = true;
    }
}

// Set this texture as an entry point for a udim texture array.
void DemandTextureImpl::setUdimTexture( unsigned int udimStartPage, unsigned int udim, unsigned int vdim, bool isBaseTexture )
{
    m_sampler.desc.isUdimBaseTexture = isBaseTexture ? 1 : 0;
    m_sampler.udimStartPage          = udimStartPage;
    m_sampler.udim                   = udim;
    m_sampler.vdim                   = vdim;
}

size_t DemandTextureImpl::getMipTailSize() 
{ 
    DEMAND_ASSERT( m_isInitialized );
    return m_mipTailSize; 
}

}  // namespace demandLoading
