// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda.h>

namespace otk {

const uint64_t BAD_ADDR = ~0ull;

/// Allocation description, returned by a suballocator or memory pool
struct MemoryBlockDesc
{
    uint64_t ptr;
    uint64_t size : 48;
    uint64_t description : 16;

    MemoryBlockDesc( uint64_t ptr_ = 0, uint64_t size_ = 0, uint64_t description_ = 0 )
        : ptr( ptr_ )
        , size( size_ )
        , description( description_ )
    {}

    bool isGood() { return ptr != BAD_ADDR; }
    bool isBad() { return ptr == BAD_ADDR; }
};

/// Align a value
inline uint64_t alignVal( uint64_t p, uint64_t alignment )
{
    uint64_t misalignment = p % alignment;
    if( misalignment != 0 )
        p += alignment - misalignment;
    return p;
}

const uint32_t TILE_SIZE_IN_BYTES = 64 * 1024;

/// Describe a block of texture tiles
union TileBlockDesc
{
    // silence warning "ISO C++ prohibits anonymous structs"
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif    
    struct
    {
        uint32_t arenaId;
        uint16_t tileId;
        uint16_t numTiles;
    };
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    
    uint64_t data = 0;

    TileBlockDesc( uint64_t data_ )
    {
        uint64_t* d = (uint64_t*)this;
        *d = data_;
    }
    TileBlockDesc( uint32_t arenaId_, uint16_t tileId_, uint16_t numTiles_ )
        : arenaId{arenaId_}
        , tileId{tileId_}
        , numTiles{numTiles_}
    {
    }

    bool         isGood() { return numTiles != 0; }
    bool         isBad() { return numTiles == 0; }
    unsigned int offset() { return tileId * TILE_SIZE_IN_BYTES; }
};

/// Bundle a TileBlockDesc with its handle
struct TileBlockHandle
{
    CUmemGenericAllocationHandle handle;
    TileBlockDesc                block;
};

}  // namespace otk
