// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
                      CUstream     stream ) override;

    /// Delegate to the wrapped ImageSource and update the time remaining, unless there is no time
    /// remaining, in which case nothing is done and false is returned.
    bool readBaseColor( float4& dest ) override;

  private:
    std::shared_ptr<std::atomic<Microseconds>> m_duration;
};

}  // namespace imageSource
