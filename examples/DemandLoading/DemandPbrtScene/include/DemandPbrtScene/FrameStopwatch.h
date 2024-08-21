// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <chrono>

namespace demandPbrtScene {

class FrameStopwatch
{
  public:
    FrameStopwatch( bool interactive )
        : m_interactive( interactive )
    {
    }

    void start() { m_frameStart = Clock::now(); }

    bool expired() const
    {
        // infinite frame budget when rendering to a file
        if( !m_interactive )
        {
            return false;
        }

        const Clock::duration duration = Clock::now() - m_frameStart;
        return duration > m_frameTime;
    }


  private:
    using Clock = std::chrono::steady_clock;

    bool m_interactive;

    // desire 60 fps
    Clock::duration                m_frameTime{ std::chrono::milliseconds( 1000 / 60 ) };
    std::chrono::time_point<Clock> m_frameStart{};
};

}  // namespace demandPbrtScene
