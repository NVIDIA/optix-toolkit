// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

using PageTableEntry = unsigned long long;

void launchPageRequester( cudaStream_t                        stream,
                          const demandLoading::DeviceContext& context,
                          unsigned int                        pageBegin,
                          unsigned int                        pageEnd,
                          PageTableEntry*                     pageTableEntries );
