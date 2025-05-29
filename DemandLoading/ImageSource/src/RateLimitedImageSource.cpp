// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/RateLimitedImageSource.h>

namespace imageSource {

class Timer
{
  public:
    Timer()
        : m_start( std::chrono::high_resolution_clock::now() )
    {
    }

    /// Returns the duration in microseconds since the timer was constructed.
    int64_t elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - m_start ).count();
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

RateLimitedImageSource::RateLimitedImageSource( std::shared_ptr<ImageSource> imageSource, std::shared_ptr<std::atomic<Microseconds>> duration )
    : WrappedImageSource( imageSource )
    , m_duration( duration )
{
}

void RateLimitedImageSource::open( TextureInfo* info )
{
    Timer timer;
    WrappedImageSource::open( info );
    *m_duration -= timer.elapsed();
}

/// Delegates to the wrapped ImageSource and decrements the time remaining, unless the
/// time limit has been exceeded, in which case nothing is done and false is returned.
bool RateLimitedImageSource::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream )
{
    if( m_duration->load() <= Microseconds( 0 ) )
        return false;

    Timer timer;
    bool  result = WrappedImageSource::readTile( dest, mipLevel, tile, stream);
    *m_duration -= timer.elapsed();
    return result;
}

/// Delegates to the wrapped ImageSource and decrements the time remaining, unless the
/// time limit has been exceeded, in which case nothing is done and false is returned.
bool RateLimitedImageSource::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    if( m_duration->load() <= Microseconds( 0 ) )
        return false;

    Timer timer;
    bool  result = WrappedImageSource::readMipLevel( dest, mipLevel, expectedWidth, expectedHeight, stream );
    *m_duration -= timer.elapsed();
    return result;
}

/// Delegates to the wrapped ImageSource and decrements the time remaining, unless the
/// time limit has been exceeded, in which case nothing is done and false is returned.
bool RateLimitedImageSource::readMipTail( char*        dest,
                                          unsigned int mipTailFirstLevel,
                                          unsigned int numMipLevels,
                                          const uint2* mipLevelDims,
                                          CUstream     stream )
{
    if( m_duration->load() <= Microseconds( 0 ) )
        return false;

    Timer timer;
    bool  result = WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
    *m_duration -= timer.elapsed();
    return result;
}

/// Delegates to the wrapped ImageSource and decrements the time remaining, unless the
/// time limit has been exceeded, in which case nothing is done and false is returned.
bool RateLimitedImageSource::readBaseColor( float4& dest )
{
    if( m_duration->load() <= Microseconds( 0 ) )
        return false;

    Timer timer;
    bool  result = WrappedImageSource::readBaseColor( dest );
    *m_duration -= timer.elapsed();
    return result;
}

}  // namespace imageSource
