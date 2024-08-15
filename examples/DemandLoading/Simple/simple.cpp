// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/DemandLoading/DemandLoader.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

#include "Simple.h"

using namespace demandLoading;

// Check status returned by a CUDA call.
inline void check( cudaError_t status )
{
    if( status != cudaSuccess )
        throw std::runtime_error( cudaGetErrorString( status ) );
}

// Global count of requests processed.
std::atomic<int> g_numRequestsProcessed( 0 );

void* toPageEntry( unsigned int value )
{
    void*         result{};
    std::intptr_t source = value;
    std::memcpy( &result, &source, sizeof( result ) );
    return result;
}

// This callback is invoked by the demand loading library when a page request is processed.
bool loadResourceCallback( cudaStream_t /*stream*/, unsigned int pageId, void* /*context*/, void** pageTableEntry )
{
    ++g_numRequestsProcessed;
    *pageTableEntry = toPageEntry( pageId );
    return true;
}

void validatePageTableEntries( const std::vector<PageTableEntry>& pageTableEntries, unsigned int currentPage, unsigned int batchSize )
{
    for( unsigned int i = 0; i < batchSize; ++i )
    {
        if( pageTableEntries[i] != currentPage )
            printf( "Mismatch pageTable[%u]=%llu != %u\n", i, pageTableEntries[i], currentPage );
        ++currentPage;
    }
}

int main()
{
    // Initialize CUDA.
    check( cudaFree( nullptr ) );

    // Create a stream to be used for asynchronous operations
    unsigned int deviceIndex = 0;
    check( cudaSetDevice( deviceIndex ) );
    cudaStream_t stream;
    check( cudaStreamCreate( &stream ) );

    // Create DemandLoader
    DemandLoader* loader = createDemandLoader( Options() );

    // Create a resource, using the given callback to handle page requests.
    const unsigned int numPages  = 128;
    unsigned int       startPage = loader->createResource( numPages, loadResourceCallback, nullptr );

    // Process all the pages of the resource in batches.
    const unsigned int batchSize   = 32;
    unsigned int       numLaunches = 0;

    void* devPageTableEntries{};
    cudaMalloc( &devPageTableEntries, sizeof( PageTableEntry ) * numPages );

    std::vector<PageTableEntry> pageTableEntries;

    for( unsigned int currentPage = startPage; currentPage < startPage + numPages; )
    {
        // Prepare for launch, obtaining DeviceContext.
        DeviceContext context;
        loader->launchPrepare( stream, context );

        // Launch the kernel.
        launchPageRequester( stream, context, currentPage, currentPage + batchSize,
                             static_cast<PageTableEntry*>( devPageTableEntries ) );
        ++numLaunches;

        // Initiate request processing, which returns a Ticket.
        Ticket ticket = loader->processRequests( stream, context );

        // Wait for any page requests to be processed.
        ticket.wait();

        // Advance the loop counter only when there were no page requests.
        if( ticket.numTasksTotal() == 0 )
        {
            pageTableEntries.resize( batchSize );
            cudaMemcpy( pageTableEntries.data(), devPageTableEntries, batchSize * sizeof( PageTableEntry ), cudaMemcpyDeviceToHost );
            validatePageTableEntries( pageTableEntries, currentPage, batchSize );
            currentPage += batchSize;
        }
    }

    printf( "Processed %i requests in %i launches.\n", g_numRequestsProcessed.load(), numLaunches );

    // Clean up
    cudaFree( devPageTableEntries );
    check( cudaStreamDestroy( stream ) );
    destroyDemandLoader( loader );

    return 0;
}
