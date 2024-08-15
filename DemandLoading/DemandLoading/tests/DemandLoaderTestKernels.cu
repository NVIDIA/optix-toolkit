// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandLoaderTestKernels.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <OptiXToolkit/DemandLoading/Texture2D.h>

using namespace demandLoading;

__global__ static void pageRequester( DeviceContext context, unsigned int pageId, bool* isResident, unsigned long long* pageTableEntry )
{
    *pageTableEntry = pagingMapOrRequest( context, pageId, isResident );
}

__host__ void launchPageRequester( CUstream stream, const DeviceContext& context, unsigned int pageId, bool* devIsResident, unsigned long long* pageTableEntry )
{
    pageRequester<<<1, 1, 0U, stream>>>( context, pageId, devIsResident, pageTableEntry );
    OTK_ERROR_CHECK( cudaStreamSynchronize( stream ) );
    OTK_ERROR_CHECK( cudaGetLastError() );
}


__global__ static void pageBatchRequester( DeviceContext context, unsigned int pageBegin, unsigned int pageEnd, PageTableEntry* pageTableEntries )
{
    unsigned int numPages = pageEnd - pageBegin;
    unsigned int index    = blockIdx.x * blockDim.x + threadIdx.x;
    if( index >= numPages )
        return;
    unsigned int pageId = pageBegin + index;

    bool           isResident;
    PageTableEntry entry = pagingMapOrRequest( context, pageId, &isResident );
    if( isResident )
    {
        pageTableEntries[index] = entry;
    }
}

__host__ void launchPageBatchRequester( CUstream stream, const DeviceContext& context, unsigned int pageBegin, unsigned int pageEnd, PageTableEntry* pageTableEntries )
{
    unsigned int threadsPerBlock = 32;
    unsigned int numPages        = pageEnd - pageBegin;
    unsigned int numBlocks       = ( numPages + threadsPerBlock - 1 ) / threadsPerBlock;

    // The DeviceContext is passed by value to the kernel, so it is copied to device memory when the kernel is launched.
    pageBatchRequester<<<numBlocks, threadsPerBlock, 0U, stream>>>( context, pageBegin, pageEnd, pageTableEntries );
}
