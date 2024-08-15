// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file Resource.h
/// Definitions for demand-loaded resources.

#include <cuda.h>

#include <functional>

namespace demandLoading {

/// ResourceCallback is a user-provided function that fills requests for pages in arbitrary
/// demand-loaded buffers.  It takes as arguments a stream and an integer page index.  It returns
/// the new page table entry for the requested page, which is typically a device pointer (but it can
/// be an arbitrary 64-bit value).
using ResourceCallback = std::function<bool( CUstream stream, unsigned int pageIndex, void *context, void** pageTableEntry )>;

}  // namespace demandLoading
