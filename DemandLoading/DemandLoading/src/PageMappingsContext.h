// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>  // for PageMapping
#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

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
        uint64_t allocSize = otk::alignVal( sizeof( PageMappingsContext ), alignof( PageMapping ) );
        allocSize += options.maxFilledPages * sizeof( PageMapping );
        allocSize += options.maxInvalidatedPages * sizeof( unsigned int );
        return allocSize;
    }

    // Initialize the struct and array pointers, assuming that the this pointer points
    // to a free memory block of sufficient size, as calculated in getAllocationSize.
    void init( const Options& options )
    {
        char* start = reinterpret_cast<char*>( this );
        char* filledPagesStart = start + otk::alignVal( sizeof( PageMappingsContext ), sizeof( PageMapping ) );
        char* invalidatedPagesStart = filledPagesStart + options.maxFilledPages * sizeof(PageMapping);

        filledPages    = reinterpret_cast<PageMapping*>( filledPagesStart );
        numFilledPages = 0;
        maxFilledPages = options.maxFilledPages;

        invalidatedPages = reinterpret_cast<unsigned int*>( invalidatedPagesStart );
        numInvalidatedPages = 0;
        maxInvalidatedPages = options.maxInvalidatedPages;
    }

    // Copy given PageMappingsContext.
    void copy( const PageMappingsContext& other )
    {
        OTK_ASSERT( numFilledPages == 0 );
        OTK_ASSERT( maxFilledPages >= other.numFilledPages );
        std::copy( filledPages, filledPages + other.numFilledPages, other.filledPages );
        numFilledPages = other.numFilledPages;

        OTK_ASSERT( numInvalidatedPages == 0 );
        OTK_ASSERT( maxInvalidatedPages >= other.numInvalidatedPages );
        std::copy( invalidatedPages, invalidatedPages + other.numInvalidatedPages, other.invalidatedPages );
        numInvalidatedPages = other.numInvalidatedPages;
    }        
};

}  // namespace demandLoading
