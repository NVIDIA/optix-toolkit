// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

namespace demandLoading {
struct DeviceContext;
}

using PageTableEntry = unsigned long long;

void launchPageRequester( CUstream                            stream,
                          const demandLoading::DeviceContext& context,
                          unsigned int                        pageId,
                          bool*                               devIsResident,
                          PageTableEntry*                     pageTableEntry );

void launchPageBatchRequester( CUstream                            stream,
                               const demandLoading::DeviceContext& context,
                               unsigned int                        pageBegin,
                               unsigned int                        pageEnd,
                               PageTableEntry*                     pageTableEntries );
