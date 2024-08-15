// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

namespace demandLoading {
struct DeviceContext;
}

void launchPageRequester( CUstream                            stream,
                          const demandLoading::DeviceContext& context,
                          unsigned int                        numPages,
                          const unsigned int*                 pageIds,
                          unsigned long long*                 outputPages,
                          bool*                               pagesResident );
