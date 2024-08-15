// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>  // for StalePage
#include <OptiXToolkit/DemandLoading/Options.h>

#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

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
        uint64_t allocSize = otk::alignVal( sizeof( RequestContext ), alignof( RequestContext ) );
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
        char* requestedPagesStart = start + otk::alignVal( sizeof( RequestContext ), sizeof( RequestContext ) );
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
