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

#include <cstddef>

/// \file Statistics.h
/// Demand loading statistics.

namespace demandLoading {

/// Demand loading statistics.  \see DemandLoader::getStatistics
struct Statistics
{
    /// Time in seconds spent processing page requests.
    double requestProcessingTime;

    /// Total number of tiles read by all ImageSources.
    size_t numTilesRead;

    /// Number of bytes read from disk by all ImageSources.
    size_t numBytesRead;

    /// Number of textures 
    size_t numTextures;

    /// Number of bytes in all textures, if they were all completely loaded
    size_t virtualTextureBytes;

    /// Total time in seconds spent reading image data by all ImageSources.  This is
    /// the cumulative time and does not take into account simultaneous reads, e.g. by multiple threads.
    double readTime;

    /// Amount of device memory allocated per device.
    size_t memoryUsedPerDevice[16];

    /// Amount of texture image data transferred to each device
    size_t bytesTransferredPerDevice[16];

    /// Number of tiles evicted by demand loading system
    unsigned int numEvictionsPerDevice[16];
};

}  // namespace demandLoading
