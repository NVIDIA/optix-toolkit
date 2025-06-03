// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>

namespace otk {

/// \brief Configures the OptiX module compile options based on the build type.
inline void configModuleCompileOptions( OptixModuleCompileOptions& options )
{
#ifndef NDEBUG
    options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#if OPTIX_VERSION >= 70400
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

#else
    options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#if OPTIX_VERSION >= 70400
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

#endif
}

#if OPTIX_VERSION < 70700
/// \brief Configures the OptiX pipeline link options based on the build type.
inline void configPipelineLinkOptions( OptixPipelineLinkOptions& options )
{
#ifdef NDEBUG
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
}
#else
/// \brief Configures the OptiX pipeline link options based on the build type.
inline void configPipelineLinkOptions( OptixPipelineLinkOptions& /*options*/ )
{
}    
#endif

}  // namespace otk
