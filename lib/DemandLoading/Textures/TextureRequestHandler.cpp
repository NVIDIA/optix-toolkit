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

#include "Textures/TextureRequestHandler.h"
#include "DemandLoaderImpl.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "Util/NVTXProfiling.h"

#include <DemandLoading/TileIndexing.h>

#include "TransferBufferDesc.h"

namespace demandLoading {

void TextureRequestHandler::fillRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    // Try to make sure there are free tiles to handle the request
    m_loader->freeStagedTiles( deviceIndex, stream );

    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to fill the same tile (or the mip tail).
    unsigned int index = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), index);

    // Do nothing if the request has already been filled.
    if( m_loader->getPagingSystem( deviceIndex )->isResident( pageId ) )
        return;

    // Decide if we need to fill a mip tail or a tile
    if( pageId == m_startPage && m_texture->isMipmapped() )
        fillMipTailRequest( deviceIndex, stream, pageId );
    else
        fillTileRequest( deviceIndex, stream, pageId );
}

void TextureRequestHandler::fillTileRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Get the texture sampler.  This is thread safe because the sampler is invariant once it's created,
    // and tile requests never occur before the sampler is created.
    const TextureSampler& sampler = m_texture->getSampler();

    // Unpack tile index into miplevel and tile coordinates.
    const unsigned int tileIndex = pageId - m_startPage;
    unsigned int       mipLevel;
    unsigned int       tileX;
    unsigned int       tileY;
    unpackTileIndex( sampler, tileIndex, mipLevel, tileX, tileY );

    // Allocate a tile in device memory
    TilePool*     tilePool    = m_loader->getDeviceMemoryManager( deviceIndex )->getTilePool();
    TileBlockDesc tileLocator = tilePool->allocate( sizeof( TileBuffer ) );
    if( !tileLocator.isValid() )
        return;

    // Allocate a transfer buffer.
    TransferBufferDesc transferBuffer =
        m_loader->allocateTransferBuffer( deviceIndex, m_texture->getImageSource()->getFillType(), sizeof( TileBuffer ), stream );
    if( transferBuffer.size == 0 )
    {
        tilePool->freeBlock( tileLocator );
        return;
    }

    // Read the tile (possibly from disk) into the transfer buffer.  
    try
    {
        m_texture->readTile( mipLevel, tileX, tileY, transferBuffer.buffer, transferBuffer.size, stream );
    }
    catch( const std::exception& e )
    {
        std::stringstream ss;
        ss << "readTile call failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        throw Exception( ss.str().c_str() );
    }

    // Copy data from transfer buffer to the sparse texture on the device
    CUmemGenericAllocationHandle handle;
    size_t                       offset;
    tilePool->getHandle( tileLocator, &handle, &offset );

    m_texture->fillTile( deviceIndex, stream, mipLevel, tileX, tileY, transferBuffer.buffer, transferBuffer.memoryType,
                         sizeof( TileBuffer ), handle, offset );

    const unsigned int lruVal = 0;
    m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, lruVal, tileLocator.getData() );

    m_loader->freeTransferBuffer( transferBuffer, stream );
}

void TextureRequestHandler::fillMipTailRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    const size_t mipTailSize  = m_texture->getMipTailSize();
    
    // Allocate device memory for the mip tail from TilePool.
    TilePool*     tilePool  = m_loader->getDeviceMemoryManager( deviceIndex )->getTilePool();
    TileBlockDesc tileBlock = tilePool->allocate( mipTailSize );
    if( !tileBlock.isValid() )
        return;

    // Allocate a transfer buffer.
    TransferBufferDesc transferBuffer =
        m_loader->allocateTransferBuffer( deviceIndex, m_texture->getImageSource()->getFillType(), mipTailSize, stream );
    if( transferBuffer.size == 0 )
    {
        tilePool->freeBlock( tileBlock );
        return;
    }

    // Read the mip tail into the transfer buffer.
    try
    {
        m_texture->readMipTail( transferBuffer.buffer, mipTailSize, stream );
    }
    catch( const std::exception& e )
    {
        std::stringstream ss;
        ss << "readMipTail call failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        throw Exception( ss.str().c_str() );
    }

    CUmemGenericAllocationHandle handle;
    size_t                       offset;
    tilePool->getHandle( tileBlock, &handle, &offset );

    // Copy data from the transfer buffer to the sparse texture on the device
    m_texture->fillMipTail( deviceIndex, stream, transferBuffer.buffer, transferBuffer.memoryType, mipTailSize, handle, offset );

    // Add a mapping for the mip tail, which will be sent to the device in pushMappings().
    unsigned int lruVal = 0;
    m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, lruVal, tileBlock.getData() );

    m_loader->freeTransferBuffer( transferBuffer, stream );
}

void TextureRequestHandler::unmapTileResource( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to fill the same tile (or the mip tail).
    unsigned int tileIndex = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), tileIndex);

    // If the page has already been remapped, don't unmap it
    PagingSystem* pagingSystem = m_loader->getPagingSystem( deviceIndex );
    if( pagingSystem->isResident( pageId ) )
        return;

    DemandTextureImpl* texture = getTexture();
    
    // Unmap the tile or mip tail
    if( tileIndex == 0 )
    {
        texture->unmapMipTail( deviceIndex, stream );
    }
    else
    {
        unsigned int mipLevel;
        unsigned int tileX;
        unsigned int tileY;
        unpackTileIndex( texture->getSampler(), tileIndex, mipLevel, tileX, tileY );
        texture->unmapTile( deviceIndex, stream, mipLevel, tileX, tileY );
    }
}


}  // namespace demandLoading
