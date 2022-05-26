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

#include "Memory/TilePool.h"

#include "Memory/Buffers.h"
#include "Util/Exception.h"
#include "Util/Math.h"

#include <algorithm>

namespace demandLoading {

TilePool::TilePool( unsigned int deviceIndex, size_t maxTexMem )
    : m_deviceIndex( deviceIndex )
    , m_arenaSize( TileArena::getRecommendedSize( m_deviceIndex ) )
    , m_maxTexMem( maxTexMem )
{
    // Try to always keep at least one arena worth of tiles free.
    m_desiredFreeTiles = static_cast<unsigned int>( m_arenaSize / sizeof( TileBuffer ) );
}

TilePool::~TilePool()
{
    for( TileArena& arena : m_arenas )
    {
        arena.destroy();
    }
}

TileBlockDesc TilePool::allocate( size_t numBytes )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    const unsigned int requestedTiles = static_cast<unsigned int>( ( numBytes + sizeof( TileBuffer ) - 1 ) / sizeof( TileBuffer ) );
    const unsigned int numFreeBlocks = static_cast<unsigned int>( m_freeTileBlocks.size() );

    // Try to find a free block with enough space.
    // Use end of list for single tile request.
    // Search from beginning for multiple tile request
    unsigned idx = 0;
    if( requestedTiles == 1 && numFreeBlocks > 0 )
    {
        idx = numFreeBlocks - 1;
    }
    else
    {
        // only search a few blocks because multi-tile blocks are put at beginning of list.
        const unsigned int maxSearch = 10; 
        for( idx = 0; idx < maxSearch && idx < numFreeBlocks; ++idx )
        {
            if( m_freeTileBlocks[idx].numTiles >= requestedTiles )
                break;
        }
        if( idx == maxSearch )
            idx = numFreeBlocks;
    }

    // Create a new arena if necessary, and put it on the free list.
    if( idx >= numFreeBlocks )
    {
        // Overallocate for multi-blocks by a little bit if necessary, since we don't coalesce blocks.
        const float  largeBlockThreshold = 1.1f;
        const size_t texMemUsage         = m_arenas.size() * m_arenaSize;
        if( ( texMemUsage > m_maxTexMem && requestedTiles == 1 ) || ( texMemUsage > m_maxTexMem * largeBlockThreshold ) )
            return TileBlockDesc{};  // empty block

        m_arenas.push_back( TileArena::create( m_deviceIndex, m_arenaSize ) );
        const unsigned int   arenaId       = static_cast<unsigned int>( m_arenas.size() - 1 );
        const unsigned short tilesPerArena = static_cast<unsigned short>( m_arenaSize / sizeof( TileBuffer ) );
        m_freeTileBlocks.push_front( TileBlockDesc{arenaId, 0, tilesPerArena} );
        idx = 0;
    }

    // Reduce size of block in free list
    unsigned short tileId = m_freeTileBlocks[idx].tileId;
    m_freeTileBlocks[idx].tileId += requestedTiles;
    m_freeTileBlocks[idx].numTiles -= requestedTiles;

    // Make the tile block to return
    TileBlockDesc tileBlock = TileBlockDesc{m_freeTileBlocks[idx].arenaId, tileId, static_cast<unsigned short>( requestedTiles )};

    // Remove the block from the free list if empty
    if( m_freeTileBlocks[idx].numTiles == 0 )
        m_freeTileBlocks.erase( m_freeTileBlocks.begin() + idx );

    return tileBlock;
}

void TilePool::getHandle( TileBlockDesc tileBlock, CUmemGenericAllocationHandle* handle, size_t* offset )
{
    DEMAND_ASSERT( tileBlock.arenaId < static_cast<unsigned int>( m_arenas.size() ) );

    std::unique_lock<std::mutex> lock( m_mutex );
    *handle = m_arenas[tileBlock.arenaId].getHandle();
    *offset = tileBlock.tileId * sizeof( TileBuffer );
}

void TilePool::freeBlock( TileBlockDesc block )
{
    DEMAND_ASSERT( block.arenaId < static_cast<unsigned int>( m_arenas.size() ) );

    std::unique_lock<std::mutex> lock( m_mutex );

    // TODO: Add the capability to coalesce free blocks
    unsigned int idx = ( block.numTiles > 1 ) ? 0 : static_cast<unsigned int>( m_freeTileBlocks.size() );
    m_freeTileBlocks.insert( m_freeTileBlocks.begin() + idx, block );
}

size_t TilePool::getTotalDeviceMemory() const
{
    std::unique_lock<std::mutex> lock( m_mutex );
    return m_arenas.size() * m_arenaSize;
}

size_t TilePool::getTotalFreeTiles() const
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Add up the (approximate) number of free tiles in the freeTileBlocks list
    size_t freeTiles = static_cast<unsigned int>( m_freeTileBlocks.size() );
    if( m_freeTileBlocks.size() > 0 )
        freeTiles += m_freeTileBlocks[0].numTiles - 1;

    // Add tiles for arenas that have not been allocated
    if( m_maxTexMem / m_arenaSize >= m_arenas.size() )
    {
        size_t numAvailableArenas = m_maxTexMem / m_arenaSize - m_arenas.size();
        size_t tilesPerArena      = m_arenaSize / sizeof( TileBuffer );
        freeTiles += tilesPerArena * numAvailableArenas;
    }

    return freeTiles;
}

}  // namespace demandLoading
