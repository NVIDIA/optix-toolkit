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

#include <OptiXToolkit/DemandLoading/Paging.h>

#include "Simple.h"

using namespace demandLoading;

__global__ static void pageRequester( DeviceContext context, unsigned int pageBegin, unsigned int pageEnd, PageTableEntry* pageTableEntries )
{
    unsigned int numPages = pageEnd - pageBegin;
    unsigned int index    = blockIdx.x * blockDim.x + threadIdx.x;
    if( index >= numPages )
        return;
    unsigned int pageId = pageBegin + index;

    bool               isResident;
    unsigned long long entry = pagingMapOrRequest( context, pageId, &isResident );
    if( isResident )
        pageTableEntries[index] = entry;
}

__host__ void launchPageRequester( cudaStream_t stream, const DeviceContext& context, unsigned int pageBegin, unsigned int pageEnd, PageTableEntry* pageTableEntries )
{
    unsigned int threadsPerBlock = 32;
    unsigned int numPages        = pageEnd - pageBegin;
    unsigned int numBlocks       = ( numPages + threadsPerBlock - 1 ) / threadsPerBlock;

    // The DeviceContext is passed by value to the kernel, so it is copied to device memory when the kernel is launched.
    pageRequester<<<numBlocks, threadsPerBlock, 0U, stream>>>( context, pageBegin, pageEnd, pageTableEntries );
}
