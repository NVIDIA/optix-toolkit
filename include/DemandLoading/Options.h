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
    bool         useSparseTextures = true;  ///< whether to use sparse or dense textures

    // Memory limits
    size_t maxTexMemPerDevice = 0;  ///< texture to allocate per device (in MB) before starting eviction (0 is unlimited)
    size_t maxPinnedMemory = 64 * 1024 * 1024;  ///< max pinned memory to use for data transfer between host and device

    // Eviction
    unsigned int maxStalePages       = 8192;  ///< max stale (resident but not used) pages to pull from device in processRequests
    unsigned int maxEvictablePages   = 0;     ///< not used
    unsigned int maxInvalidatedPages = 8192;  ///< max slots to push invalidated pages back to device in processRequests
    unsigned int maxStagedPages      = 8192;  ///< num staged pages (pages flagged as non-resident, ready to be evicted) to maintain.
    unsigned int maxRequestQueueSize = 32768; ///< max size for host-side request queue (filled over multiple processRequests cycles)
    bool useLruTable                 = true;  ///< Whether to use LRU table, or randomized eviction
    bool evictionActive              = true;  ///< whether eviction is active. (turning it off speeds up texture ops)

    // Concurrency
    unsigned int maxThreads = 0;  ///< max threads for processing requests. (0 means std::thread::hardware_concurrency)
    unsigned int maxActiveStreams = 4;  ///< number of active CUDA streams across all devices.

    // Trace file
    std::string traceFile = "";  ///< trace filename (disabled if empty).
};
// clang-format on

}
