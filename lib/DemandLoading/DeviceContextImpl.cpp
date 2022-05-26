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

#include "DeviceContextImpl.h"

namespace demandLoading {

void DeviceContextImpl::reservePerDeviceData( BulkDeviceMemory* memory, const Options& options )
{
    // We allocate only enough room in the page table for the texture samplers.  The rest of the
    // page table is unused, because we don't currently need to know the mappings for individual
    // sparse texture tiles (that's handled by the virtual memory system).
    memory->reserve<unsigned long long>( options.numPageTableEntries );

    memory->reserveBytes( sizeofResidenceBits( options ), BIT_VECTOR_ALIGNMENT );

    unsigned int lruTableSize = options.useLruTable ? ( options.numPages + 1 ) / 2 : 0;  // half byte per page
    memory->reserveBytes( lruTableSize, alignof( unsigned int ) );
}

void DeviceContextImpl::allocatePerDeviceData( BulkDeviceMemory* memory, const Options& options )
{
    pageTable.data     = memory->allocate<unsigned long long>( options.numPageTableEntries );
    pageTable.capacity = options.numPageTableEntries;
    maxNumPages        = options.numPages;

    residenceBits = memory->allocateBytes<unsigned int>( sizeofResidenceBits( options ), BIT_VECTOR_ALIGNMENT );

    unsigned int lruTableSize = options.useLruTable ? ( options.numPages + 1 ) / 2 : 0;  // half byte per page
    lruTable                  = memory->allocateBytes<unsigned int>( lruTableSize, alignof( unsigned int ) );

    DEMAND_ASSERT( isAligned( pageTable.data, alignof( unsigned long long ) ) );
    DEMAND_ASSERT( isAligned( residenceBits, BIT_VECTOR_ALIGNMENT ) );
    DEMAND_ASSERT( isAligned( lruTable, alignof( unsigned int ) ) );
}

void DeviceContextImpl::setPerDeviceData( const DeviceContext& other )
{
    pageTable     = other.pageTable;
    maxNumPages   = other.maxNumPages;
    residenceBits = other.residenceBits;
    lruTable      = other.lruTable;
}

void DeviceContextImpl::reservePerStreamData( BulkDeviceMemory* memory, const Options& options )
{
    memory->reserveBytes( sizeofReferenceBits( options ), BIT_VECTOR_ALIGNMENT );
    memory->reserve<unsigned int>( options.maxRequestedPages );
    memory->reserve<StalePage>( options.maxStalePages );
    memory->reserve<unsigned int>( options.maxEvictablePages );
    memory->reserve<unsigned int>( NUM_ARRAYS );
    memory->reserve<PageMapping>( options.maxFilledPages );
    memory->reserve<unsigned int>( options.maxInvalidatedPages );
}

void DeviceContextImpl::allocatePerStreamData( BulkDeviceMemory* memory, const Options& options )
{
    referenceBits = memory->allocateBytes<unsigned int>( sizeofReferenceBits( options ), BIT_VECTOR_ALIGNMENT );

    requestedPages.data     = memory->allocate<unsigned int>( options.maxRequestedPages );
    requestedPages.capacity = options.maxRequestedPages;

    stalePages.data     = memory->allocate<StalePage>( options.maxStalePages );
    stalePages.capacity = options.maxStalePages;

    evictablePages.data     = memory->allocate<unsigned int>( options.maxEvictablePages );
    evictablePages.capacity = options.maxEvictablePages;

    arrayLengths.data     = memory->allocate<unsigned int>( NUM_ARRAYS );
    arrayLengths.capacity = NUM_ARRAYS;

    filledPages.data     = memory->allocate<PageMapping>( options.maxFilledPages );
    filledPages.capacity = options.maxFilledPages;

    invalidatedPages.data     = memory->allocate<unsigned int>( options.maxInvalidatedPages );
    invalidatedPages.capacity = options.maxInvalidatedPages;

    DEMAND_ASSERT( isAligned( referenceBits, BIT_VECTOR_ALIGNMENT ) );
    DEMAND_ASSERT( isAligned( requestedPages.data, alignof( unsigned int ) ) );
    DEMAND_ASSERT( isAligned( stalePages.data, alignof( StalePage ) ) );
    DEMAND_ASSERT( isAligned( evictablePages.data, alignof( unsigned int ) ) );
    DEMAND_ASSERT( isAligned( arrayLengths.data, alignof( unsigned int ) ) );
    DEMAND_ASSERT( isAligned( filledPages.data, alignof( PageMapping ) ) );
    DEMAND_ASSERT( isAligned( invalidatedPages.data, alignof( unsigned int ) ) );
}

}  // namespace demandLoading
