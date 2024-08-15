// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/ErrorCheck.h>

#include <cuda.h>

namespace demandLoading {

/// Verify that the current CUDA context matches the given context.
inline void checkCudaContext( CUcontext expected )
{
    CUcontext current;
    OTK_ERROR_CHECK( cuCtxGetCurrent( &current ) );
    OTK_ASSERT( current == expected );
}

/// Verify that the current CUDA context matches the context associated with the given stream.
inline void checkCudaContext( CUstream stream )
{
    if( stream )
    {
        CUcontext context;
        OTK_ERROR_CHECK( cuStreamGetCtx( stream, &context ) );
        checkCudaContext( context );
    }
}

} // namespace demandLoading
