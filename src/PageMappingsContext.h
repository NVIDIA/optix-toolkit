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

#include "Memory/MemoryBlockDesc.h"

#include <OptiXToolkit/DemandLoading/DeviceContext.h>  // for PageMapping
#include <OptiXToolkit/DemandLoading/Options.h>

namespace demandLoading {

struct PageMappingsContext
{
    PageMapping* filledPages;
    unsigned int numFilledPages;
    unsigned int maxFilledPages;

    unsigned int* invalidatedPages;
    unsigned int  numInvalidatedPages;
    unsigned int  maxInvalidatedPages;

    void clear()
    {
        numFilledPages      = 0;
        numInvalidatedPages = 0;
    }

    // Return the size required for the struct + filledPages + invalidatedPages
    static uint64_t getAllocationSize( const Options& options )
    {
        uint64_t allocSize = alignVal( sizeof( PageMappingsContext ), alignof( PageMapping ) );
        allocSize += options.maxFilledPages * sizeof( PageMapping );
        allocSize += options.maxInvalidatedPages * sizeof( unsigned int );
        return allocSize;
    }

    // Initialize the struct and array pointers, assuming that the this pointer points
    // to a free memory block of sufficient size, as calculated in getAllocationSize.
    void init( const Options& options )
    {
        char* start = reinterpret_cast<char*>( this );
        char* filledPagesStart = start + alignVal( sizeof( PageMappingsContext ), sizeof( PageMapping ) );
        char* invalidatedPagesStart = filledPagesStart + options.maxFilledPages * sizeof(PageMapping);

        filledPages    = reinterpret_cast<PageMapping*>( filledPagesStart );
        numFilledPages = 0;
        maxFilledPages = options.maxFilledPages;

        invalidatedPages = reinterpret_cast<unsigned int*>( invalidatedPagesStart );
        numInvalidatedPages = 0;
        maxInvalidatedPages = options.maxInvalidatedPages;
    }
};

}  // namespace demandLoading
