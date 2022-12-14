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

#pragma once

#ifndef ENABLE_NVTX_PROFILING
// If NVTX profiling is not enabled, do not include the NVTX headers
// and turn the macros into no-ops.
#define SCOPED_NVTX_RANGE_A( string )
#define SCOPED_NVTX_RANGE_FUNCTION_NAME()
#else

#include <nvtx3/nvToolsExt.h>

#if defined( __GNUC__ )
#define FUNC_NAME __func__
#elif defined( _MSC_VER )
#define FUNC_NAME __FUNCTION__
#else
#define FUNC_NAME __func__
#endif

namespace demandLoading {

#define SCOPED_TOKENPASTE( x, y ) x##y
#define SCOPED_TOKENPASTE2( x, y ) SCOPED_TOKENPASTE( x, y )
#define SCOPED_NVTX_RANGE_A( string )                                                                                  \
    demandLoading::ScopedNVTXRangeA SCOPED_TOKENPASTE2( scopedNVTXRange_, __LINE__ )( string );
#define SCOPED_NVTX_RANGE_FUNCTION_NAME() SCOPED_NVTX_RANGE_A( FUNC_NAME )

struct ScopedNVTXRangeA
{
    ScopedNVTXRangeA( const char* message ) { nvtxRangePushA( message ); }
    ~ScopedNVTXRangeA() { nvtxRangePop(); }
};

}

#endif
