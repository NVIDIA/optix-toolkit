// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/CudaContext.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include <optix_stubs.h>

namespace demandPbrtScene {

CudaContext::CudaContext( unsigned int deviceIndex )
{
    // Initialize CUDA
    m_deviceIndex = deviceIndex;
    OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
    OTK_ERROR_CHECK( cuCtxGetCurrent( &m_cudaContext ) );
    OTK_ERROR_CHECK( cuStreamCreate( &m_stream, CU_STREAM_DEFAULT ) );

    // ...and also initialize OptiX.  To enable testing other classes with
    // a mock OptiX API, initialize OptiX here because this will fill in
    // the global function table.
    OTK_ERROR_CHECK( optixInit() );
}

void CudaContext::setCurrent()
{
    OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
}

}  // namespace demandPbrtScene
