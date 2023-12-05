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
bool RateLimitedImageSource::readTile( char*        dest,
                                       unsigned int mipLevel,
                                       unsigned int tileX,
                                       unsigned int tileY,
                                       unsigned int tileWidth,
                                       unsigned int tileHeight,
                                       CUstream     stream )
{
    if( m_duration->load() <= Microseconds( 0 ) )
        return false;

    Timer timer;
    bool  result = WrappedImageSource::readTile( dest, mipLevel, tileX, tileY, tileWidth, tileHeight, stream );
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
                                          unsigned int pixelSizeInBytes,
                                          CUstream     stream )
{
    if( m_duration->load() <= Microseconds( 0 ) )
        return false;

    Timer timer;
    bool  result = WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, pixelSizeInBytes, stream );
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
