// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
