//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <stddef.h>
#include <stdint.h>

#include <cuda.h>

namespace otk {

const uint64_t DEFAULT_ALLOC_SIZE = 2 * 1024 * 1024;
const uint64_t BAD_ADDR           = ~0ull;

/// Allocation description, returned by a suballocator or memory pool
struct MemoryBlockDesc
{
    uint64_t ptr;
    uint64_t size : 48;
    uint64_t description : 16;

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
        : data{data_}
    {
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
