//
//  Copyright (c) 2021, NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include "PagingSystemTestKernels.h"
#include "Util/Exception.h"

#include <DemandLoading/Paging.h>

using namespace demandLoading;

__global__ static void pageRequester( DeviceContext context, unsigned int numPages, const unsigned int* pageIds, unsigned long long* outputPages )
{
    unsigned int index = blockIdx.x + threadIdx.x;
    if( index >= numPages )
        return;

    bool isResident;
    outputPages[index] = pagingMapOrRequest( context, pageIds[index], &isResident );
}

__host__ void launchPageRequester( CUstream             stream,
                                   const DeviceContext& context,
                                   unsigned int         numPages,
                                   const unsigned int*  pageIds,
                                   unsigned long long*  outputPages )
{
    unsigned int threadsPerBlock = 32;
    unsigned int numBlocks       = ( numPages + threadsPerBlock - 1 ) / threadsPerBlock;
    pageRequester<<<numBlocks, threadsPerBlock, 0U, stream>>>( context, numPages, pageIds, outputPages );
    DEMAND_CUDA_CHECK( cudaGetLastError() );
}
