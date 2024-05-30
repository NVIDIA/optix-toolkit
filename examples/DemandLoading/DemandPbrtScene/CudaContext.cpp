//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "CudaContext.h"

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
