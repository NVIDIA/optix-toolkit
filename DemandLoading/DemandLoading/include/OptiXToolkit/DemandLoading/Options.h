// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file Options.h
/// Demand loading configuration options.

#include <cstddef>
#include <limits>
#include <string>

namespace demandLoading {

/// Demand loading configuration options.  \see createDemandLoader
// clang-format off
struct Options
{
    // Page table size
    unsigned int numPages            = 64 * 1024 * 1024;  ///< total virtual pages (4 TB of 64k texture tiles)
    unsigned int numPageTableEntries = 1024 * 1024;  ///< num page table entries, used for texture samplers and base colors.

    // Demand loading
    unsigned int maxRequestedPages = 8192;  ///< max requests to pull from device in processRequests
    unsigned int maxFilledPages    = 8192;  ///< num slots to push mappings back to device in processRequests

    // Demand load textures
    unsigned int maxTextures         = 256 * 1024;  ///< The maximum demand load textures that can be defined
    bool useSparseTextures           = true;   ///< whether to use sparse or dense textures
    bool useSmallTextureOptimization = false;  ///< whether to use dense textures for very small textures
    bool useCascadingTextureSizes    = false;  ///< whether to use cascading texture sizes
    bool coalesceWhiteBlackTiles     = false;  ///< whether to use the same backing store for all white/black tiles

    // Memory limits
    size_t maxTexMemPerDevice = 0;  ///< texture to allocate per device (in MB) before starting eviction (0 is unlimited)
    size_t maxPinnedMemory = 64 * 1024 * 1024;  ///< max pinned memory to use for data transfer between host and device

    // Eviction
    unsigned int maxStalePages       = 8192;  ///< max stale (resident but not used) pages to pull from device in processRequests
    unsigned int maxEvictablePages   = 0;     ///< not used
    unsigned int maxInvalidatedPages = 8192;  ///< max slots to push invalidated pages back to device in processRequests
    unsigned int maxStagedPages      = 8192;  ///< num staged pages (pages flagged as non-resident, ready to be evicted) to maintain.
    unsigned int maxRequestQueueSize = 8192;  ///< max size for host-side request queue (filled over multiple processRequests cycles)
    bool useLruTable                 = true;  ///< Whether to use LRU table, or randomized eviction
    bool evictionActive              = true;  ///< whether eviction is active. (turning it off speeds up texture ops)

    // Concurrency
    unsigned int maxThreads = 0;  ///< max threads for processing requests. (0 means std::thread::hardware_concurrency)

    // Trace file
    std::string traceFile;  ///< trace filename (disabled if empty).
};
// clang-format on

}
