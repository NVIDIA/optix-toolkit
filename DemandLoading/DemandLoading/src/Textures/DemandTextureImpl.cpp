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
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>
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

using namespace otk;

namespace demandLoading {

DemandTextureImpl::DemandTextureImpl( unsigned int                              id,
                                      const TextureDescriptor&                  descriptor,
                                      std::shared_ptr<imageSource::ImageSource> image,
                                      DemandLoaderImpl*                         loader )

    : m_id( id )
    , m_descriptor( descriptor )
    , m_image( image )
    , m_masterTexture( nullptr )
    , m_loader( loader )
    , m_sampler{}
{
}

DemandTextureImpl::DemandTextureImpl( unsigned int id, DemandTextureImpl* masterTexture, const TextureDescriptor& descriptor, DemandLoaderImpl* loader )
    : m_id( id )
    , m_descriptor( descriptor )
    , m_image( masterTexture->m_image )
    , m_masterTexture( masterTexture )
    , m_loader( loader )
    , m_sampler{}
{
    masterTexture->addVariantId( id );
}

void DemandTextureImpl::setImage( const TextureDescriptor& descriptor, std::shared_ptr<imageSource::ImageSource> newImage )
{
    std::unique_lock<std::mutex> lock( m_initMutex );

    // If the original image was not opened, avoid opening the new image
    if( !m_image->isOpen() )
    {
        m_descriptor = descriptor;
        m_image = newImage;
        return;
    }

    // Get the info for the new image
    imageSource::TextureInfo newInfo;
    newImage->open( &newInfo );
    OTK_ASSERT( newInfo.isValid );

    // If the new image is a different size or format, the texture will need to be re-initialized
    // FIXME: This leaks pages in the virtual address space, but currently there is no way to reclaim them.
    // There is also the possibility that these stale pages will be requested in the future, so that
    // problem must also be fixed.
    if( !( descriptor == m_descriptor ) || !( newInfo == m_info ) )
    {
        m_isInitialized = false;
        // Reset the sampler so the texture will be reinitialized, keeping only the udim info
        TextureSampler newSampler = {};
        newSampler.udimStartPage = m_sampler.udimStartPage;
        newSampler.udim = m_sampler.udim;
        newSampler.vdim = m_sampler.vdim;
        newSampler.numChannelTextures = m_sampler.numChannelTextures;
        newSampler.desc.isUdimBaseTexture = m_sampler.desc.isUdimBaseTexture;
        m_sampler = newSampler;
    }

    m_info       = newInfo;
    m_descriptor = descriptor;
    m_image      = newImage;
}

unsigned int DemandTextureImpl::getId() const
{
    return m_id;
}

void DemandTextureImpl::init()
{
    std::unique_lock<std::mutex> lock( m_initMutex );

    // Initialize the sparse or dense texture.
    if( useSparseTexture() )
    {
        // Get the master array (backing store) if there is master texture
        std::shared_ptr<SparseArray> masterArray( nullptr );
        if( m_masterTexture && !m_masterTexture->m_sparseTexture.isInitialized() )
            m_masterTexture->init();
        if( m_masterTexture )
            masterArray = m_masterTexture->m_sparseTexture.getSparseArray();

        m_sparseTexture.init( m_descriptor, m_info, masterArray );

        // Device-independent initialization.
        if( !m_isInitialized )
        {
            m_isInitialized = true;

            // Retain various properties for subsequent use.
            m_tileWidth         = m_sparseTexture.getTileWidth();
            m_tileHeight        = m_sparseTexture.getTileHeight();
            m_mipTailFirstLevel = m_sparseTexture.getMipTailFirstLevel();
            m_mipTailSize       = m_mipTailFirstLevel < m_info.numMipLevels ? m_sparseTexture.getMipTailSize() : 0;

            // Verify that the tile size agrees with TilePool.
            OTK_ASSERT( m_tileWidth * m_tileHeight * imageSource::getBytesPerChannel( m_info.format ) <= TILE_SIZE_IN_BYTES );

            // Record the dimensions of each miplevel.
            const unsigned int numMipLevels = m_info.numMipLevels;
            m_mipLevelDims.resize( numMipLevels );
            for( unsigned int i = 0; i < numMipLevels; ++i )
            {
                m_mipLevelDims[i] = m_sparseTexture.getMipLevelDims( i );
            }
            initSampler();
        }
    }
    else // dense texture
    {
        // Get the master array (backing store) if there is master texture
        std::shared_ptr<CUmipmappedArray> masterArray( nullptr );
        if( m_masterTexture && !m_masterTexture->m_denseTexture.isInitialized() )
            m_masterTexture->init();
        if( m_masterTexture )
            masterArray = m_masterTexture->m_denseTexture.getDenseArray();

        m_denseTexture.init( m_descriptor, m_info, masterArray );

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

                m_mipLevelDims[i] = m_denseTexture.getMipLevelDims( i );

                m_mipTailSize += m_mipLevelDims[i].x * m_mipLevelDims[i].y * m_info.numChannels
                                 * imageSource::getBytesPerChannel( m_info.format );
            }
            initSampler();
        }
    }
}

void DemandTextureImpl::initSampler()
{
    // Construct the canonical sampler for this texture, excluding the CUDA texture object
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
            // If the texture is being resized, remove the existing request handler
            if( m_requestHandler != nullptr )
                m_loader->getPageTableManager()->removeRequestHandler( m_requestHandler->getStartPage() );
            m_requestHandler.reset( new TextureRequestHandler( this, m_loader ) );
            m_sampler.startPage = m_loader->getPageTableManager()->reserveUnbackedPages( m_sampler.numPages, m_requestHandler.get() );
        }
    }
    else // Dense texture 
    {
        // Dense textures do not need extra page table entries
        m_sampler.numPages = 0;
        m_sampler.startPage = m_id;
    }

    // Fill in the hasCascade, cascadeLevel, and filterMode values in the sampler
    m_sampler.hasCascade = m_image->hasCascade();
    m_sampler.cascadeLevel = static_cast<unsigned short>( getCascadeLevel( m_sampler.width, m_sampler.height ) );
    m_sampler.filterMode = m_descriptor.filterMode;
}

const imageSource::TextureInfo& DemandTextureImpl::getInfo() const
{
    OTK_ASSERT( m_isInitialized );
    return m_info;
}

const TextureSampler& DemandTextureImpl::getSampler() const
{
    OTK_ASSERT( m_isInitialized );
    return m_sampler;
}

const TextureDescriptor& DemandTextureImpl::getDescriptor() const
{
    return m_descriptor;
}

uint2 DemandTextureImpl::getMipLevelDims( unsigned int mipLevel ) const
{
    OTK_ASSERT( m_isInitialized );
    OTK_ASSERT( mipLevel < m_mipLevelDims.size() );
    return m_mipLevelDims[mipLevel];
}

unsigned int DemandTextureImpl::getTileWidth() const
{
    OTK_ASSERT( m_isInitialized );
    return m_tileWidth;
}

unsigned int DemandTextureImpl::getTileHeight() const
{
    OTK_ASSERT( m_isInitialized );
    return m_tileHeight;
}

bool DemandTextureImpl::isMipmapped() const
{
    OTK_ASSERT( m_info.isValid );
    return getInfo().numMipLevels > getMipTailFirstLevel();
}

bool DemandTextureImpl::useSparseTexture() const
{
    OTK_ASSERT( m_info.isValid );

    if( !m_loader->getOptions().useSparseTextures || !m_info.isTiled )
        return false;
    if( !m_loader->getOptions().useSmallTextureOptimization )
        return true;
    return m_info.width * m_info.height > SPARSE_TEXTURE_THRESHOLD;
}

unsigned int DemandTextureImpl::getMipTailFirstLevel() const
{
    OTK_ASSERT( m_isInitialized );
    return m_mipTailFirstLevel;
}

CUtexObject DemandTextureImpl::getTextureObject() const
{
    OTK_ASSERT( m_isInitialized );
    if( useSparseTexture() )
        return m_sparseTexture.getTextureObject();
    return m_denseTexture.getTextureObject();
}

unsigned int DemandTextureImpl::getNumTilesInLevel( unsigned int mipLevel ) const
{
    if( mipLevel > m_mipTailFirstLevel || mipLevel >= m_info.numMipLevels )
        return 0;

    unsigned int levelWidthInTiles  = getLevelDimInTiles( m_mipLevelDims[0].x, mipLevel, m_tileWidth );
    unsigned int levelHeightInTiles = getLevelDimInTiles( m_mipLevelDims[0].y, mipLevel, m_tileHeight );

    return calculateNumTilesInLevel( levelWidthInTiles, levelHeightInTiles );
}

void DemandTextureImpl::accumulateStatistics( Statistics& stats )
{
    stats.numTilesRead += m_image->getNumTilesRead();
    stats.numBytesRead += m_image->getNumBytesRead();
    stats.readTime += m_image->getTotalReadTime();

    const imageSource::TextureInfo &info = m_image->getInfo();
    if( info.isValid )
        stats.virtualTextureBytes += getTextureSizeInBytes( info );

    stats.bytesTransferredToDevice += m_sparseTexture.getNumBytesFilled();
    stats.numEvictions += m_sparseTexture.getNumUnmappings();

    stats.bytesTransferredToDevice += m_denseTexture.getNumBytesFilled();
    if( m_denseTexture.isInitialized() && m_denseTexture.getTextureObject() != 0 )
        stats.deviceMemoryUsed += getTextureSizeInBytes( info );
}

// Tiles can be read concurrently.
bool DemandTextureImpl::readTile( unsigned int mipLevel, unsigned int tileX, unsigned int tileY, char* tileBuffer,
                                  size_t tileBufferSize, CUstream stream ) const
{
    OTK_ASSERT( m_isInitialized );
    OTK_ASSERT( mipLevel < m_info.numMipLevels );

    // Resize buffer if necessary.
    const unsigned int bytesPerPixel = imageSource::getBytesPerChannel( getInfo().format ) * getInfo().numChannels;
    const unsigned int bytesPerTile  = getTileWidth() * getTileHeight() * bytesPerPixel;
    OTK_ASSERT_MSG( bytesPerTile <= tileBufferSize, "Maximum tile size exceeded" );
    (void)bytesPerTile;  // silence unused variable warning
    (void)tileBufferSize;

    return m_image->readTile( tileBuffer, mipLevel, { tileX, tileY, getTileWidth(), getTileHeight() }, stream );
}

// Tiles can be filled concurrently.
void DemandTextureImpl::fillTile( CUstream                     stream,
                                  unsigned int                 mipLevel,
                                  unsigned int                 tileX,
                                  unsigned int                 tileY,
                                  const char*                  tileData,
                                  CUmemorytype                 tileDataType,
                                  size_t                       tileSize,
                                  CUmemGenericAllocationHandle handle,
                                  size_t                       offset ) const
{
    OTK_ASSERT( mipLevel < m_info.numMipLevels );
    OTK_ASSERT( tileSize <= TILE_SIZE_IN_BYTES );

    m_sparseTexture.fillTile( stream, mipLevel, tileX, tileY, tileData, tileDataType, tileSize, handle, offset );
}

void DemandTextureImpl::mapTile( CUstream                     stream,
                                 unsigned int                 mipLevel,
                                 unsigned int                 tileX,
                                 unsigned int                 tileY,
                                 CUmemGenericAllocationHandle tileHandle,
                                 size_t                       tileOffset ) const
{
    OTK_ASSERT( mipLevel < m_info.numMipLevels );
    m_sparseTexture.mapTile( stream, mipLevel, tileX, tileY, tileHandle, tileOffset );
}

// Tiles can be unmapped concurrently.
void DemandTextureImpl::unmapTile( CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const
{
    OTK_ASSERT( mipLevel < m_info.numMipLevels );
    m_sparseTexture.unmapTile( stream, mipLevel, tileX, tileY );
}

bool DemandTextureImpl::readNonMipMappedData( char* buffer, size_t bufferSize, CUstream stream ) const
{
    OTK_ASSERT( m_isInitialized );
    OTK_ASSERT( m_info.numMipLevels == 1 );
    OTK_ASSERT_MSG( m_mipTailSize <= bufferSize, "Provided buffer is too small." );
    (void)bufferSize;  // silence unused variable warning.

    return m_image->readMipLevel( buffer, 0, getInfo().width, getInfo().height, stream );
}

bool DemandTextureImpl::readMipTail( char* buffer, size_t bufferSize, CUstream stream ) const
{
    return readMipLevels( buffer, bufferSize, getMipTailFirstLevel(), stream );
}

bool DemandTextureImpl::readMipLevels( char* buffer, size_t bufferSize, unsigned int startLevel, CUstream stream ) const
{
    OTK_ASSERT( m_isInitialized );
    OTK_ASSERT( startLevel < getInfo().numMipLevels );

    const unsigned int pixelSize = getInfo().numChannels * imageSource::getBytesPerChannel( getInfo().format );
    size_t dataSize = ( m_mipLevelDims[startLevel].x * m_mipLevelDims[startLevel].y * pixelSize * 4 ) / 3;
    OTK_ASSERT_MSG( dataSize <= bufferSize, "Provided buffer is too small." );
    (void)dataSize;  // silence unused variable warning
    (void)bufferSize;

    return m_image->readMipTail( buffer, startLevel, getInfo().numMipLevels, m_mipLevelDims.data(), pixelSize, stream );
}

void DemandTextureImpl::fillMipTail( CUstream                     stream,
                                     const char*                  mipTailData,
                                     CUmemorytype                 mipTailDataType,
                                     size_t                       mipTailSize,
                                     CUmemGenericAllocationHandle handle,
                                     size_t                       offset ) const
{
    OTK_ASSERT( getMipTailFirstLevel() < m_info.numMipLevels );

    m_sparseTexture.fillMipTail( stream, mipTailData, mipTailDataType, mipTailSize, handle, offset );
}

void DemandTextureImpl::mapMipTail( CUstream stream, CUmemGenericAllocationHandle tileHandle, size_t tileOffset )
{
    m_sparseTexture.mapMipTail( stream, tileHandle, tileOffset );
}

void DemandTextureImpl::unmapMipTail( CUstream stream ) const
{
    m_sparseTexture.unmapMipTail( stream );
}

// Fill the dense texture on the given stream.
void DemandTextureImpl::fillDenseTexture( CUstream stream, const char* textureData, unsigned int width, unsigned int height, bool bufferPinned )
{
    m_denseTexture.fillTexture( stream, textureData, width, height, bufferPinned );
}

// Lazily open the associated image source.
void DemandTextureImpl::open()
{
    std::unique_lock<std::mutex> lock( m_initMutex );

    if (m_masterTexture)
        m_masterTexture->open();

    // Open the image if necessary, fetching the dimensions and other info.
    if( !m_isOpen )
    {
        m_image->open( &m_info );
        OTK_ASSERT( m_info.isValid );
        m_isOpen = true;
    }
}

// Set this texture as an entry point for a udim texture array.
void DemandTextureImpl::setUdimTexture( unsigned int udimStartPage, unsigned int udim, unsigned int vdim, unsigned int numChannelTextures, bool isBaseTexture )
{
    m_sampler.desc.isUdimBaseTexture = isBaseTexture ? 1 : 0;
    m_sampler.udimStartPage          = udimStartPage;
    m_sampler.udim                   = static_cast<unsigned short>( udim );
    m_sampler.vdim                   = static_cast<unsigned short>( vdim );
    m_sampler.numChannelTextures     = numChannelTextures;
}

size_t DemandTextureImpl::getMipTailSize() 
{ 
    OTK_ASSERT( m_isInitialized );
    return m_mipTailSize; 
}

}  // namespace demandLoading
