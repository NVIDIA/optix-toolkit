// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "PagingSystemTestKernels.h"
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <OptiXToolkit/DemandLoading/Paging.h>

using namespace demandLoading;

__global__ static void pageRequester( DeviceContext       context,
                                      unsigned int        numPages,
                                      const unsigned int* pageIds,
                                      unsigned long long* outputPages,
                                      bool*               pagesResident )
{
    unsigned int index = blockIdx.x + threadIdx.x;
    if( index >= numPages )
        return;

    outputPages[index] = pagingMapOrRequest( context, pageIds[index], &pagesResident[index] );
}

__host__ void launchPageRequester( CUstream             stream,
                                   const DeviceContext& context,
                                   unsigned int         numPages,
                                   const unsigned int*  pageIds,
                                   unsigned long long*  outputPages,
                                   bool*                pagesResident )
{
    unsigned int threadsPerBlock = 32;
    unsigned int numBlocks       = ( numPages + threadsPerBlock - 1 ) / threadsPerBlock;
    pageRequester<<<numBlocks, threadsPerBlock, 0U, stream>>>( context, numPages, pageIds, outputPages, pagesResident );
    OTK_ERROR_CHECK( cudaStreamSynchronize( stream ) );
    OTK_ERROR_CHECK( cudaGetLastError() );
}
