// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
