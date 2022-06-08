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

#include <DemandLoading/DemandLoader.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <exception>
#include <stdexcept>

using namespace demandLoading;

// Launch kernel.  Defined in optixDemandLoadSimple.cu
void launchPageRequester( cudaStream_t stream, const DeviceContext& context, unsigned int pageBegin, unsigned int pageEnd );

// Check status returned by a CUDA call.
inline void check( cudaError_t status )
{
    if( status != cudaSuccess )
        throw std::runtime_error( cudaGetErrorString( status ) );
}

// Global count of requests processed.
std::atomic<int> g_numRequestsProcessed( 0 );

// This callback is invoked by the demand loading library when a page request is processed.
void* callback( unsigned int deviceIndex, cudaStream_t stream, unsigned int pageIndex )
{
    ++g_numRequestsProcessed;
    return nullptr;
}

int main()
{
    // Create DemandLoader
    DemandLoader*             loader = createDemandLoader( Options() );

    // Create a resource, using the given callback to handle page requests.
    const unsigned int numPages  = 128;
    unsigned int       startPage = loader->createResource( numPages, callback );

    // Create a stream on the first supported device, which is used for asynchronous operations.
    unsigned int deviceIndex = loader->getDevices().at( 0 );
    check( cudaSetDevice( deviceIndex ) );
    cudaStream_t stream;
    check( cudaStreamCreate( &stream ) );

    // Process all the pages of the resource in batches.
    const unsigned int batchSize   = 32;
    unsigned int       numLaunches = 0;
    for( unsigned int currentPage = startPage; currentPage < startPage + numPages; )
    {
        // Prepare for launch, obtaining DeviceContext.
        DeviceContext      context;
        loader->launchPrepare( deviceIndex, stream, context );

        // Launch the kernel.
        launchPageRequester( stream, context, currentPage, currentPage + batchSize );
        ++numLaunches;

        // Initiate request processing, which returns a Ticket.
        Ticket ticket = loader->processRequests( deviceIndex, stream, context );

        // Wait for any page requests to be processed.
        ticket.wait();

        // Advance the loop counter only when there were no page requests.
        if( ticket.numTasksTotal() == 0 )
            currentPage += batchSize;
    }

    printf( "Processed %i requests in %i launches.\n", g_numRequestsProcessed.load(), numLaunches );

    // Clean up
    check( cudaStreamDestroy( stream ) );
    destroyDemandLoader( loader );

    return 0;
}
