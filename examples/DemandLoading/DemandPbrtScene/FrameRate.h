//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "Timer.h"

#include <numeric>
#include <vector>

namespace demandPbrtScene {

/// Moving average of frame rate.
class FrameRate
{
  public:
    FrameRate()
    {
        // The vector of frame times never exceeds its initial capacity.
        m_frameTimes.reserve( 20 );
    }

    void reset()
    {
        m_frameTimes.clear();
        m_nextIndex = 0;
        m_timer.reset();
    }        

    void update()
    {
        // The vector grows until it reaches its initial capacity.
        m_frameTimes.resize( m_nextIndex + 1 );

        // Record the elapsed time since the last update.
        m_frameTimes[m_nextIndex] = m_timer.getSeconds();
        m_timer.reset();

        // Advance the index, modulo the capacity.
        m_nextIndex = ( m_nextIndex + 1 ) % m_frameTimes.capacity();
    }

    double getFPS() const
    {
        // Return the frame count divided by the sum of the frame times.
        double sum = std::accumulate( m_frameTimes.begin(), m_frameTimes.end(), 0.0 );
        return m_frameTimes.size() / sum;
    }        
    

  private:
    std::vector<double>       m_frameTimes;
    unsigned int              m_nextIndex{0};
    Timer                     m_timer;
};

}  // namespace demandPbrtScene
