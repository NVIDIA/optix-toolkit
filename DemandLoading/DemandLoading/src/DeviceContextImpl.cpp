// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DeviceContextImpl.h"

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

using namespace otk;

namespace demandLoading {

template <typename Type>
inline Type* allocItems( MemoryPool<DeviceAllocator, HeapSuballocator>* memPool, size_t numItems, size_t alignment = 0 )
{
    alignment             = alignment ? alignment : alignof( Type );
    MemoryBlockDesc block = memPool->alloc( numItems * sizeof( Type ), alignment );
    OTK_ERROR_CHECK( cuMemsetD8( static_cast<CUdeviceptr>( block.ptr ), 0, numItems * sizeof( Type ) ) );
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
    maxTextures        = options.maxTextures;

    const unsigned int sizeofResidenceBitsInInts = ( options.numPages + 31 ) / 32;
    residenceBits = allocItems<unsigned int>( memPool, sizeofResidenceBitsInInts, BIT_VECTOR_ALIGNMENT );

    // Allocate half byte per page (8 pages per int32)
    const unsigned int sizeofLruTableInInts = options.useLruTable ? ( options.numPages + 7 ) / 8 : 0; 
    lruTable = allocItems<unsigned int>( memPool, sizeofLruTableInInts );

    OTK_ASSERT( isAligned( pageTable.data, alignof( unsigned long long ) ) );
    OTK_ASSERT( isAligned( residenceBits, BIT_VECTOR_ALIGNMENT ) );
    OTK_ASSERT( isAligned( lruTable, alignof( unsigned int ) ) );
}

void DeviceContextImpl::setPerDeviceData( const DeviceContext& other )
{
    pageTable     = other.pageTable;
    maxNumPages   = other.maxNumPages;
    maxTextures   = other.maxTextures;
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

    OTK_ASSERT( isAligned( referenceBits, BIT_VECTOR_ALIGNMENT ) );
    OTK_ASSERT( isAligned( requestedPages.data, alignof( unsigned int ) ) );
    OTK_ASSERT( isAligned( stalePages.data, alignof( StalePage ) ) );
    OTK_ASSERT( isAligned( evictablePages.data, alignof( unsigned int ) ) );
    OTK_ASSERT( isAligned( arrayLengths.data, alignof( unsigned int ) ) );
    OTK_ASSERT( isAligned( filledPages.data, alignof( PageMapping ) ) );
    OTK_ASSERT( isAligned( invalidatedPages.data, alignof( unsigned int ) ) );
}

}  // namespace demandLoading
