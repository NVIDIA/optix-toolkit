// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "PagingSystemKernels.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <algorithm>

namespace demandLoading {

void launchKernel( CUmodule module, const char* symbol, unsigned int numBlocks, unsigned int numThreadsPerBlock, CUstream stream, void** params )
{
    CUfunction fn{};
    OTK_ERROR_CHECK( cuModuleGetFunction( &fn, module, symbol ) );
    OTK_ERROR_CHECK( cuLaunchKernel( fn, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0U, stream, params, nullptr ) );  // NOLINT(readability-suspicious-call-argument)
}

inline unsigned int roundNearest32( unsigned int value )
{
    return ( value + 31 ) & 0xFFFFFFE0;  // Round to nearest multiple of 32
}

inline unsigned int roundUp( unsigned int value, unsigned int size )
{
    return ( value + size - 1 ) / size;
}

void launchPullRequests( CUmodule             module,
                         CUstream             stream,
                         const DeviceContext& context,
                         unsigned int         launchNum,
                         unsigned int         lruThreshold,
                         unsigned int         startPage,
                         unsigned int         endPage )
{
    const unsigned int numPagesPerThread  = std::max( 32U, roundNearest32( context.maxNumPages / 65536U ) );
    const unsigned int numThreadsPerBlock = 64;
    const unsigned int numPagesPerBlock   = numPagesPerThread * numThreadsPerBlock;
    const unsigned int numBlocks          = roundUp( context.maxNumPages, numPagesPerBlock );

    void* kernelParams[5]{const_cast<DeviceContext*>( &context ), &launchNum, &lruThreshold, &startPage, &endPage};
    launchKernel( module, "_ZN13demandLoading18devicePullRequestsENS_13DeviceContextEjjjj", numBlocks,
                  numThreadsPerBlock, stream, kernelParams );
}

void launchPushMappings( CUmodule module, CUstream stream, const DeviceContext& constContext, int filledPageCount )
{
    const int numPagesPerThread   = 2;
    const int numThreadsPerBlock  = 128;
    const int numPagesPerBlock    = numPagesPerThread * numThreadsPerBlock;
    const int numFilledPageBlocks = roundUp( filledPageCount, numPagesPerBlock );

    DeviceContext& context = const_cast<DeviceContext&>( constContext );

    void* kernelParams[]{&context.pageTable.data, &context.pageTable.capacity, &context.residenceBits,
                         &context.lruTable,       &context.filledPages.data,   &filledPageCount};
    launchKernel( module, "_ZN13demandLoading18devicePushMappingsEPyjPjS1_PNS_11PageMappingEi", numFilledPageBlocks,
                  numThreadsPerBlock, stream, kernelParams );
}

void launchInvalidatePages( CUmodule module, CUstream stream, const DeviceContext& constContext, int invalidatedPageCount )
{
    const int numPagesPerThread        = 2;
    const int numThreadsPerBlock       = 128;
    const int numPagesPerBlock         = numPagesPerThread * numThreadsPerBlock;
    const int numInvalidatedPageBlocks = roundUp( invalidatedPageCount, numPagesPerBlock );

    DeviceContext& context = const_cast<DeviceContext&>( constContext );

    void* kernelParams[]{&context.residenceBits, &context.invalidatedPages.data, &invalidatedPageCount};
    launchKernel( module, "_ZN13demandLoading21deviceInvalidatePagesEPjS0_i", numInvalidatedPageBlocks,
                  numThreadsPerBlock, stream, kernelParams );
}

}  // namespace demandLoading
