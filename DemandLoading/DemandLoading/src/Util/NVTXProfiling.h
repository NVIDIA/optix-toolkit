// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
