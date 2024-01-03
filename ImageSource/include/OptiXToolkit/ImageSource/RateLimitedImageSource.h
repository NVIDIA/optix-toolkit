//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file RateLimitedImageSource.h

#include <OptiXToolkit/ImageSource/WrappedImageSource.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

namespace imageSource {

/// RateLimitedImageSource adapts an ImageSource to cease operations after a time limit has been
/// exceeded.  Useful for achieving a target framerate in an interactive application.  
class RateLimitedImageSource : public WrappedImageSource
{
  public:
    /// The time limit is specified in microseconds.  (Since it must be atomic, we can't easily use
    /// std::chrono::duration.)
    using Microseconds = std::int64_t;

    /// Rate limit the given ImageSource by ceasing operations whenever the given duration (in
    /// microseconds) is less or equal to zero.  After each non-trivial operation, the duration is
    /// atomically decremented by its elapsed time.
    RateLimitedImageSource( std::shared_ptr<ImageSource> imageSource, std::shared_ptr<std::atomic<Microseconds>> duration );

    /// Destructor
    ~RateLimitedImageSource() override = default;

    /// Delegate to the wrapped ImageSource and update the time remaining.  The open call proceeds
    /// regardless of the current time limit.
    void open( TextureInfo* info ) override;

    /// Delegate to the wrapped ImageSource and update the time remaining, unless there is no time
    /// remaining, in which case nothing is done and false is returned.
    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream ) override;

    /// Delegate to the wrapped ImageSource and update the time remaining, unless there is no time
    /// remaining, in which case nothing is done and false is returned.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override;

    /// Delegate to the wrapped ImageSource and update the time remaining, unless there is no time
    /// remaining, in which case nothing is done and false is returned.
    bool readMipTail( char*        dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      unsigned int pixelSizeInBytes,
                      CUstream     stream ) override;

    /// Delegate to the wrapped ImageSource and update the time remaining, unless there is no time
    /// remaining, in which case nothing is done and false is returned.
    bool readBaseColor( float4& dest ) override;

  private:
    std::shared_ptr<std::atomic<Microseconds>> m_duration;
};

}  // namespace imageSource
