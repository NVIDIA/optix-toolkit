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

using namespace otk;

namespace demandLoading {

template <typename Type>
inline Type* allocItems( MemoryPool<DeviceAllocator, HeapSuballocator>* memPool, size_t numItems, size_t alignment = 0 )
{
    alignment             = alignment ? alignment : alignof( Type );
    MemoryBlockDesc block = memPool->alloc( numItems * sizeof( Type ), alignment );
    DEMAND_CUDA_CHECK( cuMemsetD8( static_cast<CUdeviceptr>( block.ptr ), 0, numItems * sizeof( Type ) ) );
    return reinterpret_cast<Type*>( block.ptr );
}

void DeviceContextImpl::allocatePerDeviceData( MemoryPool<DeviceAllocator, HeapSuballocator>* memPool, const Options& options )
{
    // Note: We allocate only enough room in the device-side page table for the texture samplers
    // and texture base colors.  Mappings for individual sparse texture tiles are stored on the
    // host to allow eviction, but are not currently needed on the device.

    pageTable.data     = allocItems<unsigned long long>( memPool, options.numPageTableEntries );
    pageTable.capacity = options.numPageTableEntries;
    maxNumPages        = options.numPages;

    const unsigned int sizeofResidenceBitsInInts = ( options.numPages + 31 ) / 32;
    residenceBits = allocItems<unsigned int>( memPool, sizeofResidenceBitsInInts, BIT_VECTOR_ALIGNMENT );

    // Allocate half byte per page (8 pages per int32)
    const unsigned int sizeofLruTableInInts = options.useLruTable ? ( options.numPages + 7 ) / 8 : 0; 
    lruTable = allocItems<unsigned int>( memPool, sizeofLruTableInInts );

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

void DeviceContextImpl::allocatePerStreamData( MemoryPool<DeviceAllocator, HeapSuballocator>* memPool, const Options& options )
{
    const unsigned int sizeofReferenceBitsInInts = ( options.numPages + 31 ) / 32;
    referenceBits = allocItems<unsigned int>( memPool, sizeofReferenceBitsInInts, BIT_VECTOR_ALIGNMENT );

    requestedPages.data     = allocItems<unsigned int>( memPool, options.maxRequestedPages );
    requestedPages.capacity = options.maxRequestedPages;

    stalePages.data     = allocItems<StalePage>( memPool, options.maxStalePages );
    stalePages.capacity = options.maxStalePages;

    evictablePages.data     = allocItems<unsigned int>( memPool, options.maxEvictablePages );
    evictablePages.capacity = options.maxEvictablePages;

    arrayLengths.data     = allocItems<unsigned int>( memPool, ArrayLengthsIndex::NUM_ARRAY_LENGTHS );
    arrayLengths.capacity = ArrayLengthsIndex::NUM_ARRAY_LENGTHS;

    filledPages.data     = allocItems<PageMapping>( memPool, options.maxFilledPages );
    filledPages.capacity = options.maxFilledPages;

    invalidatedPages.data     = allocItems<unsigned int>( memPool, options.maxInvalidatedPages );
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
