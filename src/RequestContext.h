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

#include <OptiXToolkit/DemandLoading/DeviceContext.h>  // for StalePage
#include <OptiXToolkit/DemandLoading/Options.h>

#include "Memory/MemoryBlockDesc.h"

namespace demandLoading {

struct RequestContext
{
    unsigned int* requestedPages;
    unsigned int  maxRequestedPages;

    StalePage*   stalePages;
    unsigned int maxStalePages;

    unsigned int*             arrayLengths;
    static const unsigned int numArrayLengths = 2;

    // Get the size required for the RequestContext struct + requestedPages + stalePages + arrayLengths.
    static uint64_t getAllocationSize( const Options& options )
    {
        uint64_t allocSize = alignVal( sizeof( RequestContext ), alignof( RequestContext ) );
        allocSize += options.maxRequestedPages * sizeof( unsigned int );
        allocSize += options.maxStalePages * sizeof( StalePage );
        allocSize += numArrayLengths * sizeof( unsigned int );
        return allocSize;
    }

    // Initialize the struct and array pointers, assuming that the this pointer points to a free
    // memory block of sufficient size, as calculated in getAllocationSize.
    void init( const Options& options )
    {
        char* start               = reinterpret_cast<char*>( this );
        char* requestedPagesStart = start + alignVal( sizeof( RequestContext ), sizeof( RequestContext ) );
        char* stalePagesStart     = requestedPagesStart + options.maxRequestedPages * sizeof( unsigned int );
        char* arrayLengthsStart   = stalePagesStart + options.maxStalePages * sizeof( StalePage );

        maxRequestedPages = options.maxRequestedPages;
        requestedPages    = reinterpret_cast<unsigned int*>( requestedPagesStart );
        maxStalePages     = options.maxStalePages;
        stalePages        = reinterpret_cast<StalePage*>( stalePagesStart );
        arrayLengths      = reinterpret_cast<unsigned int*>( arrayLengthsStart );
    }
};

}  // namespace demandLoading
