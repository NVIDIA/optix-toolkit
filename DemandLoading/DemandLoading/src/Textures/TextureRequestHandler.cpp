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
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "TransferBufferDesc.h"
#include "Util/NVTXProfiling.h"

#include <OptiXToolkit/DemandLoading/TileIndexing.h>

using namespace otk;

namespace demandLoading {

void TextureRequestHandler::fillRequest( CUstream stream, unsigned int pageId )
{
   loadPage( stream, pageId, false );
}

void TextureRequestHandler::loadPage( CUstream stream, unsigned int pageId, bool reloadIfResident )
{
    // Try to make sure there are free tiles to handle the request
    m_loader->freeStagedTiles( stream );

    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to fill the same tile (or the mip tail).
    unsigned int index = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), index);

    // Do nothing if the page is resident and the flag says not to reload it.
    unsigned long long pageEntry;
    bool resident =  m_loader->getPagingSystem()->isResident( pageId, &pageEntry );
    if( resident && !reloadIfResident )
        return;

    // Get the TileBlockHandle from the page table if the page is resident
    TileBlockHandle bh{ 0, 0 };
    if( resident )
    {
        bh.block = TileBlockDesc( pageEntry );
        bh.handle = m_loader->getDeviceMemoryManager()->getTileBlockHandle( bh.block );
    } 

    // Decide if we need to fill a mip tail or a tile
    if( pageId == m_startPage && m_texture->isMipmapped() )
        fillMipTailRequest( stream, pageId, bh );
    else
        fillTileRequest( stream, pageId, bh );
}

void TextureRequestHandler::fillTileRequest( CUstream stream, unsigned int pageId, TileBlockHandle bh )
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

    // Make sure to have device memory for the tile
    bool useNewBlock = bh.block.isBad();
    if( useNewBlock )
    {
        bh = m_loader->getDeviceMemoryManager()->allocateTileBlock( TILE_SIZE_IN_BYTES );
        if( bh.block.isBad() )
        {
            // If the allocation failed, set max memory to current size and turn on eviction.
            m_loader->setMaxTextureMemory( m_loader->getDeviceMemoryManager()->getTextureTileMemory() );
            return;
        }
    }

    // Allocate a transfer buffer.
    TransferBufferDesc transferBuffer = m_loader->allocateTransferBuffer( m_texture->getFillType(), TILE_SIZE_IN_BYTES, stream );
    if( transferBuffer.memoryBlock.size == 0 )
    {
        m_loader->getDeviceMemoryManager()->freeTileBlock( bh.block );
        return;
    }

    // Read the tile (possibly from disk) into the transfer buffer.
    bool satisfied;
    try
    {
        satisfied = m_texture->readTile( mipLevel, tileX, tileY, reinterpret_cast<char*>( transferBuffer.memoryBlock.ptr ),
                                         transferBuffer.memoryBlock.size, stream );
    }
    catch( const std::exception& e )
    {
        std::stringstream ss;
        ss << "readTile call failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        throw std::runtime_error( ss.str().c_str() );
    }

    if( satisfied )
    {
        // Copy data from transfer buffer to the sparse texture on the device
        m_texture->fillTile( stream,
                             mipLevel, tileX, tileY,                                     // Tile to fill
                             reinterpret_cast<char*>( transferBuffer.memoryBlock.ptr ),  // Src buffer
                             transferBuffer.memoryType, TILE_SIZE_IN_BYTES,              // Src type and size
                             bh.handle, bh.block.offset()                                // Dest
                             );

        // Add a mapping for the tile, which will be sent to the device in pushMappings().
        if( useNewBlock )
        {
            m_loader->setPageTableEntry( pageId, true, static_cast<unsigned long long>( bh.block.data ) );
        }
    }

    m_loader->freeTransferBuffer( transferBuffer, stream );
}

void TextureRequestHandler::fillMipTailRequest( CUstream stream, unsigned int pageId, TileBlockHandle bh )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    const size_t mipTailSize  = m_texture->getMipTailSize();

    // Allocate device texture memory for mip tail.
    DeviceMemoryManager* deviceMemoryManager = m_loader->getDeviceMemoryManager();

    // Make sure to have device memory for the tile
    bool useNewBlock = bh.block.isBad();
    if( useNewBlock )
    {
        bh = deviceMemoryManager->allocateTileBlock( mipTailSize );
        if( bh.block.isBad() )
        {
            // If the allocation failed, set max memory to current size and turn on eviction.
            m_loader->setMaxTextureMemory( m_loader->getDeviceMemoryManager()->getTextureTileMemory() );
            return;
        }
    }

    // Allocate a transfer buffer.
    TransferBufferDesc transferBuffer = m_loader->allocateTransferBuffer( m_texture->getFillType(), mipTailSize, stream );
    if( transferBuffer.memoryBlock.size == 0 )
    {
        deviceMemoryManager->freeTileBlock( bh.block );
        return;
    }

    // Read the mip tail into the transfer buffer.
    bool satisfied;
    try
    {
        satisfied = m_texture->readMipTail( reinterpret_cast<char*>( transferBuffer.memoryBlock.ptr ), mipTailSize, stream );
    }
    catch( const std::exception& e )
    {
        std::stringstream ss;
        ss << "readMipTail call failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        throw std::runtime_error( ss.str().c_str() );
    }

    if( satisfied )
    {
        // Copy data from the transfer buffer to the sparse texture on the device
        m_texture->fillMipTail( stream,
                                reinterpret_cast<char*>( transferBuffer.memoryBlock.ptr ),  // Src buffer
                                transferBuffer.memoryType, mipTailSize,                     // Src type and size
                                bh.handle, bh.block.offset()                                // Dest
                                );

        // Add a mapping for the mip tail, which will be sent to the device in pushMappings().

        if( useNewBlock )
        {
            m_loader->setPageTableEntry( pageId, true, static_cast<unsigned long long>( bh.block.data ) );
        }
    }

    m_loader->freeTransferBuffer( transferBuffer, stream );
}

void TextureRequestHandler::unmapTileResource( CUstream stream, unsigned int pageId )
{
    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to fill the same tile (or the mip tail).
    unsigned int tileIndex = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), tileIndex );

    // If the page has already been remapped, don't unmap it
    PagingSystem* pagingSystem = m_loader->getPagingSystem();
    if( pagingSystem->isResident( pageId ) )
        return;

    DemandTextureImpl* texture = getTexture();
    
    // Unmap the tile or mip tail
    if( tileIndex == 0 )
    {
        texture->unmapMipTail( stream );
    }
    else
    {
        unsigned int mipLevel;
        unsigned int tileX;
        unsigned int tileY;
        unpackTileIndex( texture->getSampler(), tileIndex, mipLevel, tileX, tileY );
        texture->unmapTile( stream, mipLevel, tileX, tileY );
    }
}

unsigned int TextureRequestHandler::getTextureTilePageId( unsigned int mipLevel, unsigned int tileX, unsigned int tileY )
{
    const demandLoading::TextureSampler& sampler = getTexture()->getSampler();
    unsigned int pageId = sampler.startPage + sampler.mipLevelSizes[mipLevel].mipLevelStart;
    pageId += getPageOffsetFromTileCoords( tileX, tileY, sampler.mipLevelSizes[mipLevel].levelWidthInTiles );
    return pageId;
}

}  // namespace demandLoading
