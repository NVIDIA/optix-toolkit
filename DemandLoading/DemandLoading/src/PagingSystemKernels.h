// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>

#include <cuda.h>

namespace demandLoading {

void launchPullRequests( CUmodule module,
                         CUstream             stream,
                         const DeviceContext& context /*on host*/,
                         unsigned int         launchNum,
                         unsigned int         lruThreshold,
                         unsigned int         startPage, unsigned int         endPage /*inclusive*/ );

void launchPushMappings( CUmodule module, CUstream stream, const DeviceContext& context /*on host*/
                         , int filledPageCount );

void launchInvalidatePages( CUmodule module, CUstream stream, const DeviceContext& context /*on host*/
                            , int invalidatedPageCount );

}  // namespace demandLoading
